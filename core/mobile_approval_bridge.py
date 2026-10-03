"""Exact bridge from OwnerKernel confirmation to paired-phone biometric approval.

The model never receives approval authority. A REQUIRE_CONFIRMATION receipt is
bound to its exact request id and normalized-argument digest. The paired phone
may approve that digest after local biometric verification; only then does the
authenticated host mint a short-lived OwnerKernel consent ticket and execute
the already-pending request.

The mapping is intentionally process-local. After a restart, an old durable
phone approval cannot resurrect a lost OwnerKernel pending request.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any

from core.action_contracts import ActionReceipt, ActionStatus
from core.action_gateway import OwnerRuntime
from core.nexus.owner_approval import OwnerApprovalManager


@dataclass(frozen=True)
class BridgedApproval:
    approval_id: str
    request_id: str
    action_digest: str
    tool: str


@dataclass(frozen=True)
class ApprovalExecution:
    approval_id: str
    request_id: str
    state: str
    message: str
    action_id: str | None = None


class MobileApprovalBridge:
    """Resume exact OwnerKernel requests only after fingerprint-approved consent."""

    def __init__(
        self,
        approvals: OwnerApprovalManager,
        runtime: OwnerRuntime,
        *,
        consent_ttl_seconds: int = 30,
        recent_limit: int = 20,
    ):
        if not isinstance(approvals, OwnerApprovalManager):
            raise TypeError("approvals must be OwnerApprovalManager")
        if not isinstance(runtime, OwnerRuntime):
            raise TypeError("runtime must be OwnerRuntime")
        if isinstance(consent_ttl_seconds, bool) or not 1 <= consent_ttl_seconds <= 120:
            raise ValueError("consent_ttl_seconds must be 1..120")
        if isinstance(recent_limit, bool) or not 1 <= recent_limit <= 100:
            raise ValueError("recent_limit must be 1..100")
        self.approvals = approvals
        self.runtime = runtime
        self.consent_ttl_seconds = int(consent_ttl_seconds)
        self._guard = threading.RLock()
        self._reconcile_guard = threading.Lock()
        self._pending: dict[str, BridgedApproval] = {}
        self._by_request: dict[str, str] = {}
        self._recent = deque(maxlen=recent_limit)

    @property
    def pending(self) -> tuple[BridgedApproval, ...]:
        with self._guard:
            return tuple(self._pending.values())

    @property
    def recent(self) -> tuple[ApprovalExecution, ...]:
        with self._guard:
            return tuple(self._recent)

    def queue_receipt(
        self,
        receipt: ActionReceipt,
        *,
        summary: str | None = None,
        ttl_seconds: int = 180,
    ) -> BridgedApproval:
        if not isinstance(receipt, ActionReceipt):
            raise TypeError("receipt must be ActionReceipt")
        if receipt.result.status is not ActionStatus.REQUIRE_CONFIRMATION:
            raise ValueError("only confirmation-required receipts can be bridged")
        if receipt.schema_version != 2 or not receipt.request_id or not receipt.normalized_argument_digest:
            raise ValueError("approval bridge requires an exact v2 authorization receipt")

        with self._guard:
            existing_id = self._by_request.get(receipt.request_id)
            if existing_id is not None:
                existing = self._pending.get(existing_id)
                if existing is not None:
                    if existing.action_digest != receipt.normalized_argument_digest:
                        raise PermissionError("pending approval request changed digest")
                    return existing

            label = (summary or f"Approve JARVIS action: {receipt.tool}").strip()
            approval = self.approvals.create(
                label[:500],
                receipt.normalized_argument_digest,
                ttl_seconds=ttl_seconds,
            )
            bridge = BridgedApproval(
                approval.approval_id,
                receipt.request_id,
                receipt.normalized_argument_digest,
                receipt.tool,
            )
            self._pending[approval.approval_id] = bridge
            self._by_request[receipt.request_id] = approval.approval_id
            return bridge

    def _finish(self, bridge: BridgedApproval, result: ApprovalExecution) -> None:
        with self._guard:
            self._pending.pop(bridge.approval_id, None)
            self._by_request.pop(bridge.request_id, None)
            self._recent.appendleft(result)

    def reconcile(self) -> tuple[ApprovalExecution, ...]:
        """Process terminal phone decisions once; pending decisions remain untouched."""
        # UI refresh and the background scheduler may arrive together. Only one
        # reconciler may consume/approve/execute a request; a losing caller must
        # not cancel the first caller's in-flight authorized action.
        with self._reconcile_guard:
            return self._reconcile_pending()

    def _reconcile_pending(self) -> tuple[ApprovalExecution, ...]:
        with self._guard:
            items = tuple(self._pending.values())

        completed: list[ApprovalExecution] = []
        for bridge in items:
            try:
                approval = self.approvals.get(bridge.approval_id)
            except KeyError:
                self.runtime.owner.cancel(bridge.request_id)
                result = ApprovalExecution(
                    bridge.approval_id,
                    bridge.request_id,
                    "cancelled",
                    "Biometric approval disappeared; exact action was cancelled.",
                )
                self._finish(bridge, result)
                completed.append(result)
                continue

            if approval.state == "pending":
                continue

            if approval.state in {"rejected", "expired"}:
                self.runtime.owner.cancel(bridge.request_id)
                result = ApprovalExecution(
                    bridge.approval_id,
                    bridge.request_id,
                    approval.state,
                    f"Owner approval {approval.state}; no action executed.",
                )
                self._finish(bridge, result)
                completed.append(result)
                continue

            if approval.state != "approved":
                continue

            try:
                # Consume the phone approval first. It can never be replayed.
                self.approvals.consume(
                    bridge.approval_id,
                    action_digest=bridge.action_digest,
                )
                # Then mint a very short exact OwnerKernel ticket bound to the
                # request id + digest already held by the authenticated host.
                ticket = self.runtime.owner.approve(
                    bridge.request_id,
                    bridge.action_digest,
                    ttl_seconds=self.consent_ttl_seconds,
                )
                receipt = self.runtime.execute_approved(
                    bridge.request_id,
                    ticket.ticket_id,
                )
                result = ApprovalExecution(
                    bridge.approval_id,
                    bridge.request_id,
                    receipt.result.status.value,
                    receipt.result.message,
                    action_id=receipt.action_id,
                )
            except Exception as exc:
                # Fail closed. Revoke/cancel what remains; never manufacture a
                # second phone approval or replay a possibly-effectful request.
                try:
                    self.runtime.owner.cancel(bridge.request_id)
                except Exception:
                    pass
                result = ApprovalExecution(
                    bridge.approval_id,
                    bridge.request_id,
                    "failed",
                    f"Approved action could not be safely resumed ({type(exc).__name__}).",
                )

            self._finish(bridge, result)
            completed.append(result)

        return tuple(completed)

    def status(self) -> dict[str, Any]:
        return {
            "pending_exact_actions": len(self.pending),
            "recent": [
                {
                    "approval_id": item.approval_id,
                    "request_id": item.request_id,
                    "state": item.state,
                    "message": item.message,
                    "action_id": item.action_id,
                }
                for item in self.recent
            ],
        }
