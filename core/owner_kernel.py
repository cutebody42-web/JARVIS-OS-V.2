"""Non-LLM authorization and in-memory exact-consent ledger.

This is an application boundary against untrusted data/model proposals, not a
sandbox against hostile Python already running in this process.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import secrets
import threading
import time
from types import MappingProxyType

from core.authority_contracts import (
    ActionRequest, AuthorizationDecision as A, AuthorizationResult, CapabilityGrant,
    ConfirmationPolicy, ConsentTicket, DestructiveLevel, PolicyContext, RiskLevel,
    canonical_arguments,
)
from core.capability_registry import REGISTRY


POLICY_VERSION = "nexus-owner-v2.1"
MAX_CONSENT_SECONDS = 300
MAX_PENDING = 128


@dataclass
class _TicketState:
    ticket: ConsentTicket
    deadline: float
    consumed: bool = False
    revoked: bool = False


class OwnerKernel:
    def __init__(self, context: PolicyContext, *, registry=REGISTRY,
                 clock=time.monotonic, wall_clock=time.time):
        self.__context = context
        self.__registry = MappingProxyType(dict(registry))
        self.__clock = clock
        self.__wall_clock = wall_clock
        self.__lock = threading.RLock()
        self.__pending: dict[str, tuple[ActionRequest, float]] = {}
        self.__tickets: dict[str, _TicketState] = {}
        self.__closed = False
        self.__owner_key = object()

    @property
    def context(self):
        return self.__context

    def _bootstrap_owner_control(self):
        """Application composition only. Never registered as a tool or prompt API."""
        return OwnerControl(self, self.__owner_key)

    def _result(self, request, decision, rule, reason, ticket_id=None):
        return AuthorizationResult(decision, rule, reason, request.capability_id,
                                   request.arguments_digest, ticket_id)

    def _validate(self, request):
        if self.__closed:
            return self._result(request, A.DENY, "session.closed", "Owner session is closed.")
        entry = self.__registry.get(request.capability_id)
        if entry is None:
            return self._result(request, A.DENY, "capability.unknown", "Capability is not registered.")
        cap = entry.capability
        if cap.disabled_reason or cap.confirmation_policy is ConfirmationPolicy.FORBIDDEN:
            return self._result(request, A.DENY, "capability.disabled", cap.disabled_reason or "Forbidden capability.")
        if cap.privilege_requirement != "user":
            return self._result(request, A.DENY, "privilege.forbidden", "Models cannot request privilege elevation.")
        if self.context.environment not in cap.environments:
            return self._result(request, A.DENY, "surface.unavailable", "Capability is unavailable in this environment.")
        try:
            if canonical_arguments(entry.normalize(request.arguments)) != request.arguments_json:
                raise ValueError("Request is not normalized.")
            if request.capability_id in {"workspace.read", "workspace.create_text"}:
                # Both root and relative path are bound by the trusted context.
                root = self.context.workspace_root
                target = (root / request.arguments["path"]).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError("Outside workspace.")
                if cap.access == "mutate" and target.is_relative_to(Path(__file__).resolve().parents[1]):
                    raise ValueError("Application/policy source is not an artifact workspace.")
        except (TypeError, ValueError, OSError, RuntimeError):
            return self._result(request, A.DENY, "arguments.invalid", "Arguments or workspace scope are not admitted.")
        return None

    def _prune(self):
        now = self.__clock()
        self.__pending = {k: v for k, v in self.__pending.items() if v[1] > now}
        self.__tickets = {k: v for k, v in self.__tickets.items()
                          if v.deadline > now and not v.revoked and not v.consumed}

    def authorize(self, request: ActionRequest, *, ticket_id: str | None = None) -> AuthorizationResult:
        with self.__lock:
            self._prune()
            invalid = self._validate(request)
            if invalid:
                return invalid
            cap = self.__registry[request.capability_id].capability
            requires = (cap.confirmation_policy is ConfirmationPolicy.EXACT or cap.external_side_effect
                        or cap.sensitive_data_access or cap.destructive_level is not DestructiveLevel.NONE
                        or cap.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL})
            if ticket_id is not None:
                state = self.__tickets.get(ticket_id)
                if state is None or state.consumed or state.revoked:
                    return self._result(request, A.DENY, "consent.invalid", "Consent is absent, expired, reused or revoked.")
                ticket = state.ticket
                grant = ticket.grant
                if (ticket.request_id != request.request_id or grant.capability_id != request.capability_id
                        or grant.arguments_digest != request.arguments_digest
                        or grant.owner_id != self.context.owner_id or grant.session_id != self.context.session_id
                        or ticket.policy_version != POLICY_VERSION):
                    return self._result(request, A.DENY, "consent.mismatch", "Consent does not match this exact request.")
                # Atomic consumption before invocation, even if the handler fails.
                state.consumed = True
                self.__pending.pop(request.request_id, None)
                return self._result(request, A.ALLOW, "owner.exact_consent", "Exact owner consent consumed.", ticket_id)
            if requires:
                if len(self.__pending) >= MAX_PENDING and request.request_id not in self.__pending:
                    return self._result(request, A.DENY, "consent.capacity", "Too many pending owner decisions.")
                current = self.__pending.get(request.request_id)
                if current is not None and current[0] != request:
                    return self._result(request, A.DENY, "request.changed", "A pending request cannot change arguments.")
                self.__pending.setdefault(request.request_id, (request, self.__clock() + MAX_CONSENT_SECONDS))
                return self._result(request, A.REQUIRE_CONFIRMATION, "owner.confirmation_required",
                                    "Exact owner approval is required; model text cannot approve this action.")
            return self._result(request, A.ALLOW, "registry.safe_operation", "Reviewed low-risk operation admitted.")

    def _require_owner(self, key):
        if key is not self.__owner_key or self.__closed:
            raise PermissionError("An active application owner channel is required.")

    def _pending_for_owner(self, key):
        with self.__lock:
            self._require_owner(key)
            self._prune()
            return tuple(request for request, _ in self.__pending.values())

    def _approve(self, key, request_id, expected_digest, ttl_seconds):
        with self.__lock:
            self._require_owner(key)
            self._prune()
            if type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= MAX_CONSENT_SECONDS:
                raise ValueError("Consent lifetime must be between 0 and 300 seconds.")
            request, deadline = self.__pending.get(request_id, (None, None))
            if request is None or request.arguments_digest != expected_digest or self._validate(request):
                raise PermissionError("No matching, valid pending request.")
            # Re-approval replaces rather than multiplies live tickets for a request.
            for state in self.__tickets.values():
                if state.ticket.request_id == request_id:
                    state.revoked = True
            ttl = min(float(ttl_seconds), deadline - self.__clock())
            if ttl <= 0:
                raise PermissionError("Pending request expired.")
            now = self.__wall_clock()
            grant = CapabilityGrant(secrets.token_urlsafe(24), self.context.owner_id,
                                    self.context.session_id, request.capability_id,
                                    request.arguments_digest, now + ttl, "owner.exact_consent")
            ticket = ConsentTicket(secrets.token_urlsafe(24), request_id, grant, now, now + ttl, POLICY_VERSION)
            self.__tickets[ticket.ticket_id] = _TicketState(ticket, self.__clock() + ttl)
            return ticket

    def _revoke(self, key, ticket_id):
        with self.__lock:
            self._require_owner(key)
            state = self.__tickets.get(ticket_id)
            if state is None or state.consumed:
                return False
            state.revoked = True
            return True

    def _cancel_request(self, key, request_id):
        with self.__lock:
            self._require_owner(key)
            found = self.__pending.pop(request_id, None) is not None
            for state in self.__tickets.values():
                if state.ticket.request_id == request_id:
                    state.revoked = True
            return found

    def close(self):
        with self.__lock:
            self.__closed = True
            self.__pending.clear()
            self.__tickets.clear()


class OwnerControl:
    """Retain on the authenticated host side; never provide this object to models."""
    def __init__(self, kernel, key):
        self.__kernel, self.__key = kernel, key

    def pending(self) -> tuple[ActionRequest, ...]:
        return self.__kernel._pending_for_owner(self.__key)

    def approve(self, request_id: str, expected_digest: str, *, ttl_seconds=120) -> ConsentTicket:
        return self.__kernel._approve(self.__key, request_id, expected_digest, ttl_seconds)

    def revoke(self, ticket_id: str) -> bool:
        return self.__kernel._revoke(self.__key, ticket_id)

    def cancel(self, request_id: str) -> bool:
        return self.__kernel._cancel_request(self.__key, request_id)

    def request(self, request_id: str) -> ActionRequest:
        for request in self.pending():
            if request.request_id == request_id:
                return request
        raise PermissionError("No pending request in this owner session.")
