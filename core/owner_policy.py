"""Phase 1 compatibility query backed by the Sovereign Owner Kernel.

A boolean query is not an execution permit. Runtime actions use ActionGateway.
"""
from dataclasses import dataclass
from core.authority_contracts import ActionRequest, AuthorizationDecision, canonical_arguments
from core.capability_registry import resolve_tool
from core.action_gateway import create_runtime
from core.owner_kernel import OwnerKernel

@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


def authorize_action(tool: str, parameters: dict) -> PolicyDecision:
    try:
        cid, args = resolve_tool(tool, parameters)
        runtime = create_runtime()
        result = OwnerKernel(runtime.gateway.context).authorize(
            ActionRequest(cid or "unknown", canonical_arguments(args)))
        return PolicyDecision(result.decision is AuthorizationDecision.ALLOW, result.reason)
    except (ValueError, TypeError):
        return PolicyDecision(False, "Invalid action arguments.")
