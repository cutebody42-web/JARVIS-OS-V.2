"""Native read route plus conservative adapters for legacy read-only actions."""

from datetime import datetime, timezone
import json
import time
from uuid import uuid4

from core.action_contracts import ActionReceipt, ActionStatus, Evidence, ToolResult
from core.owner_policy import authorize_action
from core.qa_mode import guard_tool_call


def _dispatch(tool: str, parameters: dict) -> ToolResult:
    if tool == "system_time":
        value = datetime.now().astimezone().isoformat(timespec="seconds")
        return ToolResult(ActionStatus.SUCCEEDED, f"Local time: {value}", (
            Evidence("python.datetime.system_clock", value, datetime.now(timezone.utc).isoformat()),
        ))
    if tool == "web_search":
        from actions.web_search import web_search
        raw = web_search(parameters=parameters, player=None)
    elif tool == "weather_report":
        from actions.weather_report import weather_action
        raw = weather_action(parameters=parameters, player=None)
    else:
        raise ValueError("No registered action handler.")
    # Returning normally, or returning 'Done', does not verify a postcondition.
    return ToolResult(ActionStatus.UNVERIFIED, str(raw or "No result returned."), error_code="legacy_unverified")


def run_action(tool: str, parameters: dict, *, task_id: str, step_id: str,
               route: str, cancel_flag=None, allowed_tools=None) -> ActionReceipt:
    start = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    # JSON roundtrip is both validation and a snapshot: callers cannot mutate
    # authorized parameters during dispatch or retroactively alter a receipt.
    encoded = json.dumps(parameters, sort_keys=True, ensure_ascii=False, allow_nan=False)
    snapshot = json.loads(encoded)
    if cancel_flag is not None and cancel_flag.is_set():
        result = ToolResult(ActionStatus.CANCELLED, "Task cancelled.", error_code="cancelled")
    elif allowed_tools is not None and tool not in allowed_tools:
        result = ToolResult(
            ActionStatus.DENIED,
            "Capability is not allowed by the active persona.",
            error_code="persona_denied",
        )
    else:
        owner = authorize_action(tool, snapshot)
        qa = guard_tool_call(tool, snapshot) if owner.allowed else None
        if not owner.allowed or (qa is not None and not qa.allowed):
            reason = owner.reason if not owner.allowed else qa.reason
            result = ToolResult(ActionStatus.DENIED, reason, error_code="policy_denied")
        else:
            try:
                result = _dispatch(tool, snapshot)
            except Exception as exc:
                # Do not echo exception strings that can contain keys or URLs.
                result = ToolResult(ActionStatus.FAILED, f"Action failed ({type(exc).__name__}).",
                                    error_code=type(exc).__name__)
    return ActionReceipt(str(uuid4()), task_id, str(step_id), str(tool), encoded, route,
                         start, datetime.now(timezone.utc).isoformat(),
                         (time.perf_counter() - tick) * 1000, result)
