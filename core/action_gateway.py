"""Single action invocation boundary: schema -> authority -> adapter -> receipt."""

from core.console import configure_utf8_console

configure_utf8_console()

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4
from weakref import WeakValueDictionary

from core.action_contracts import (ActionReceipt, ActionStatus, Evidence, ToolResult,
                                   VerifierResult, VerificationStatus)
from core.authority_contracts import (
    ActionRequest, AuthorizationDecision as A, AuthorizationResult, PolicyContext,
    canonical_arguments,
)
from core.capability_registry import REGISTRY, resolve_tool
from core.owner_kernel import OwnerKernel, POLICY_VERSION
from core.qa_mode import guard_tool_call, qa_block_message


_runtime_scope = ContextVar("nexus_owner_runtime", default=None)
_resource_locks = WeakValueDictionary()
_resource_lock_guard = threading.Lock()


def _lock_for(resource, context):
    if resource is None:
        # A fresh lock imposes no serialization across unrelated reads.
        return threading.RLock()
    key = resource
    if resource == "workspace":
        key += ":" + str(context.workspace_root)
    elif resource in {"memory", "email"}:
        key += ":" + context.owner_id
    with _resource_lock_guard:
        return _resource_locks.setdefault(key, threading.RLock())


class ActionGateway:
    def __init__(self, kernel: OwnerKernel):
        self.__kernel = kernel
        self.__receipts = deque(maxlen=256)
        self.__receipt_lock = threading.Lock()

    @property
    def context(self):
        return self.__kernel.context

    @property
    def receipts(self):
        with self.__receipt_lock:
            return tuple(self.__receipts)

    def close(self):
        self.__kernel.close()

    def run_tool(self, tool, arguments, *, task_id=None, step_id="1", route="model",
                 cancel_flag=None, additional_denial="",
                 additional_denial_rule="preflight.denied", runtime=None):
        invalid = ""
        try:
            if type(tool) is not str or not 0 < len(tool) <= 120:
                raise ValueError("Invalid tool identity.")
            cid, args = resolve_tool(tool, arguments)
            request = ActionRequest(cid or "unknown." + tool, canonical_arguments(args))
        except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError):
            tool = tool if type(tool) is str and len(tool) <= 120 else "invalid_tool"
            request = ActionRequest("invalid.arguments", canonical_arguments({"rejected": True}))
            invalid = "Invalid or unregistered arguments; no action was invoked."
        return self.run_request(
            request, tool=tool, task_id=task_id, step_id=step_id,
            route=route, cancel_flag=cancel_flag,
            additional_denial=invalid or additional_denial,
            additional_denial_rule=(
                "arguments.invalid" if invalid else additional_denial_rule
            ),
            runtime=runtime,
        )

    def run_request(self, request: ActionRequest, *, tool=None, task_id=None, step_id="1",
                    route="owner", ticket_id=None, cancel_flag=None,
                    additional_denial="", additional_denial_rule="preflight.denied",
                    runtime=None):
        start = datetime.now(timezone.utc).isoformat()
        tick = time.perf_counter()
        if request.capability_id == "task.submit" and route not in {"live", "owner"}:
            additional_denial = "Only the live host can enqueue a model task; workers cannot spawn workers."
        entry = REGISTRY.get(request.capability_id)
        resource = entry.capability.resource_lock if entry else None
        with _lock_for(resource, self.context):
            cancelled = cancel_flag is not None and cancel_flag.is_set()
            # QA can only restrict authority. Normalize the outbound-send alias so
            # even an owner-approved resumption remains blocked in QA mode.
            qa_tool = "email_control" if request.capability_id == "email.send" else (tool or request.capability_id)
            qa_args = {**request.arguments, "action": "approve"} if request.capability_id == "email.send" else request.arguments
            if request.capability_id in {"workspace.read", "workspace.create_text"}:
                # NEXUS paths are relative to the owner workspace, not cwd.
                # Translate only the QA inspection copy; consent and execution
                # retain the exact original normalized arguments and digest.
                qa_tool = "file_controller"
                path = request.arguments.get("path")
                qa_args = {"path": str(self.context.workspace_root / path) if type(path) is str else ""}
            qa = guard_tool_call(qa_tool, qa_args)
            if cancelled or additional_denial or not qa.allowed:
                reason = "Task cancelled." if cancelled else additional_denial or qa_block_message(qa)
                auth = AuthorizationResult(
                    A.DENY,
                    "execution.cancelled" if cancelled else additional_denial_rule,
                    reason,
                    request.capability_id,
                    request.arguments_digest,
                )
            else:
                auth = self.__kernel.authorize(request, ticket_id=ticket_id)
            if cancelled:
                result = ToolResult(ActionStatus.CANCELLED, auth.reason, error_code="cancelled")
            elif auth.decision is A.DENY:
                result = ToolResult(ActionStatus.DENIED, auth.reason, error_code=auth.policy_rule)
            elif auth.decision is A.REQUIRE_CONFIRMATION:
                result = ToolResult(ActionStatus.REQUIRE_CONFIRMATION, auth.reason, error_code=auth.policy_rule)
            else:
                try:
                    from core.action_adapters import invoke
                    from core.tenant import tenant_scope
                    owner_id = self.context.owner_id
                    with tenant_scope(None if owner_id == "local-owner" else owner_id), runtime_scope(runtime) if runtime is not None else _empty_scope():
                        result = invoke(request.capability_id, request.arguments, self.context, runtime)
                    if not isinstance(result, ToolResult):
                        result = ToolResult(ActionStatus.UNVERIFIED, "Adapter returned no verified result.", error_code="adapter_unverified")
                except Exception as exc:
                    result = ToolResult(ActionStatus.FAILED, f"Action failed ({type(exc).__name__}).", error_code=type(exc).__name__)
        finish = datetime.now(timezone.utc).isoformat()
        receipt = ActionReceipt(
            str(uuid4()), task_id or str(uuid4()), str(step_id), tool or request.capability_id,
            request.arguments_json, route, start, finish, (time.perf_counter() - tick) * 1000,
            result, schema_version=2, request_id=request.request_id,
            authorization_decision=auth.decision, capability_id=request.capability_id,
            policy_rule=auth.policy_rule, normalized_argument_digest=request.arguments_digest,
            confirmation_ticket_id=auth.confirmation_ticket_id,
            authorization_evidence=(Evidence("nexus.owner_kernel", f"{POLICY_VERSION}: {auth.policy_rule}: {auth.decision.value}", start),),
            # Envelope for the static adapter's existing postcondition evidence.
            # This does not invent an independent external-service verifier.
            verifier_result=VerifierResult(
                VerificationStatus.PASS if result.status is ActionStatus.SUCCEEDED else VerificationStatus.UNKNOWN,
                "nexus.trusted_adapter", result.evidence),
        )
        with self.__receipt_lock:
            self.__receipts.append(receipt)
        return receipt


@contextmanager
def _empty_scope():
    yield


@contextmanager
def runtime_scope(runtime):
    token = _runtime_scope.set(runtime)
    try:
        yield
    finally:
        _runtime_scope.reset(token)


class OwnerRuntime:
    """Host composition: model dispatch uses gateway; authenticated host uses owner."""
    def __init__(self, context: PolicyContext, *, clock=time.monotonic, wall_clock=time.time):
        kernel = OwnerKernel(context, clock=clock, wall_clock=wall_clock)
        self.gateway = ActionGateway(kernel)
        self.owner = kernel._bootstrap_owner_control()

    def execute_approved(self, request_id, ticket_id, *, cancel_flag=None):
        request = self.owner.request(request_id)
        return self.gateway.run_request(request, ticket_id=ticket_id, cancel_flag=cancel_flag,
                                        route="owner", runtime=self)


def create_runtime(*, owner_id=None, session_id=None, workspace_root=None, environment="cloud"):
    from core.tenant import get_current_user_id
    owner_id = owner_id or get_current_user_id() or "local-owner"
    session_id = session_id or str(uuid4())
    if workspace_root is None:
        owner_dir = hashlib.sha256(owner_id.encode()).hexdigest()[:24]
        workspace_root = Path(tempfile.gettempdir()) / "nexus-workspaces" / owner_dir / session_id
    return OwnerRuntime(PolicyContext(owner_id, session_id, Path(workspace_root), environment))


def current_runtime():
    return _runtime_scope.get() or create_runtime()


def guarded_entrypoint(tool):
    """Retire ambient legacy entry points. Dispatch admitted actions through adapters.

    The legacy body is retained for review, not used as an authorization bypass.
    No model-readable 'approved' kwarg, transcript or pending global draft is read.
    """
    def decorate(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            parameters = kwargs.get("parameters", args[0] if args else {})
            runtime = current_runtime()
            receipt = runtime.gateway.run_tool(tool, parameters or {}, route="legacy", runtime=runtime)
            return render_receipt(receipt)
        return guarded
    return decorate


def render_receipt(receipt):
    result = receipt.result
    if result.status is ActionStatus.SUCCEEDED:
        return result.message
    return f"{result.status.value}: {result.message}"
