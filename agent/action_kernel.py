"""Native read route plus conservative adapters for legacy read-only actions."""

from datetime import datetime, timezone

from core.action_contracts import ActionReceipt, ActionStatus, Evidence, ToolResult
from core.action_gateway import current_runtime
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
        # The legacy weather_action opens a desktop browser. Use a query-only
        # adapter here; its prose is still unverified, not an observed forecast.
        from actions.web_search import web_search
        raw = web_search(parameters={"query": "Current weather in " + parameters["city"]}, player=None)
    else:
        raise ValueError("No registered action handler.")
    # Returning normally, or returning 'Done', does not verify a postcondition.
    return ToolResult(ActionStatus.UNVERIFIED, str(raw or "No result returned."), error_code="legacy_unverified")


def run_action(tool: str, parameters: dict, *, task_id: str, step_id: str,
               route: str, cancel_flag=None, owner_runtime=None,
               allowed_tools=None) -> ActionReceipt:
    """Compatibility entry point; all dispatch now goes through the owner kernel."""
    runtime = owner_runtime or current_runtime()
    # Preserve QA as an additional restrictive gate, never a source of grants.
    qa = guard_tool_call(tool, parameters) if isinstance(parameters, dict) else None
    denial = qa.reason if qa is not None and not qa.allowed else ""
    if allowed_tools is not None and tool not in frozenset(allowed_tools):
        denial = "Capability is not admitted by the active persona."
    return runtime.gateway.run_tool(
        tool, parameters, task_id=task_id, step_id=step_id, route=route,
        cancel_flag=cancel_flag, additional_denial=denial,
        runtime=runtime,
    )
