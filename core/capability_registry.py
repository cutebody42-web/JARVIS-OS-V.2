"""Reviewed capability inventory. Metadata is code, never inferred from tool text.

Legacy composite tools remain explicitly denied until their nested operations can
be individually scoped. Adding an LLM declaration does not register authority.
"""

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from collections.abc import Callable
import re

from core.authority_contracts import (
    ConfirmationPolicy as C, DestructiveLevel as D, ExecutionSurface as S,
    RiskLevel as R, ToolCapability, canonical_arguments,
)


def fields(args, required=(), optional=()):
    if not set(required) <= set(args) or set(args) - set(required) - set(optional):
        raise ValueError("Missing or unregistered arguments; authority fields are not tool arguments.")
    return dict(args)


def text(value, limit=4000, *, empty=False):
    if type(value) is not str or (not empty and not value.strip()) or len(value) > limit or "\x00" in value:
        raise ValueError("Invalid bounded text argument.")
    return value


def one_text(key, limit):
    def normalize(args):
        fields(args, (key,))
        return {key: text(args[key], limit)}
    return normalize


def no_arguments(args):
    return fields(args)


def workspace_arguments(args, *, write=False):
    args = fields(args, ("path", "content") if write else ("path",), ("name",))
    path = text(args["path"], 1024)
    if args.get("name"):
        path += "/" + text(args["name"], 200)
    parts = path.split("/")
    if ("\\" in path or PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
            or any(p in {"", ".", ".."} or p.startswith(".") or ":" in p for p in parts)):
        raise ValueError("Use an explicit relative workspace path without traversal or hidden files.")
    if PurePosixPath(path).suffix.lower() not in {".txt", ".md", ".csv", ".json"}:
        raise ValueError("Only inert text artifacts are admitted in the workspace.")
    result = {"path": path}
    if write:
        result["content"] = text(args["content"], 32000, empty=True)
    return result


def email_send(args):
    args = fields(args, ("to", "subject", "body"), ("cc", "bcc", "provider"))
    if args.get("provider", "gmail") != "gmail":
        raise ValueError("Only the exact Gmail API adapter is admitted.")
    result = {"provider": "gmail", "subject": text(args["subject"], 998),
              "body": text(args["body"], 32000)}
    if "\r" in result["subject"] or "\n" in result["subject"]:
        raise ValueError("Email headers cannot contain newlines.")
    for key in ("to", "cc", "bcc"):
        addresses = text(args.get(key, ""), 2000, empty=key != "to")
        # Exact mailbox addresses only; no contact lookup or display-name guessing.
        normalized = [a.strip() for a in addresses.split(",") if a.strip()]
        if key == "to" and not normalized:
            raise ValueError("At least one explicit recipient is required.")
        if any(not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", a)
               for a in normalized):
            raise ValueError("Explicit mailbox addresses are required.")
        result[key] = ", ".join(normalized)
    return result


def email_read(args):
    args = fields(args, ("message_id",), ("provider",))
    if args.get("provider", "gmail") != "gmail":
        raise ValueError("Unsupported provider.")
    return {"message_id": text(args["message_id"], 200), "provider": "gmail"}


def memory_write(args):
    fields(args, ("category", "key", "value"))
    if args["category"] not in {"identity", "preferences", "projects", "relationships", "wishes", "notes"}:
        raise ValueError("Memory is data; policy/config categories are forbidden.")
    if type(args["key"]) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", args["key"]):
        raise ValueError("Invalid memory key.")
    return {"category": args["category"], "key": args["key"], "value": text(args["value"])}


def queue_submit(args):
    fields(args, ("goal",), ("priority",))
    priority = args.get("priority", "normal")
    if priority not in {"low", "normal", "high"}:
        raise ValueError("Unknown priority.")
    return {"goal": text(args["goal"], 12000), "priority": priority}


def task_status(args):
    fields(args, (), ("action", "task_id"))
    action = args.get("action", "get")
    if action not in {"get", "all"}:
        raise ValueError("Unknown task action.")
    return {"action": action, "task_id": text(args.get("task_id", ""), 100, empty=True)}


def task_cancel(args):
    fields(args, ("task_id",))
    return {"task_id": text(args["task_id"], 100)}


@dataclass(frozen=True)
class RegisteredCapability:
    capability: ToolCapability
    normalize: Callable


def capability(cid, access, risk, external, sensitive, destructive, privilege,
               reversible, verifier, confirmation, surface, lock, normalize,
               *, environments=frozenset({"cloud", "desktop"}), disabled=""):
    return RegisteredCapability(ToolCapability(
        cid, access, risk, external, sensitive, destructive, privilege, reversible,
        verifier, confirmation, surface, lock, environments, disabled), normalize)


_ENTRIES = (
    capability("system.time", "read", R.LOW, False, False, D.NONE, "user", True, True, C.NONE, S.NATIVE, None, no_arguments),
    # Query egress can carry private data; read-only networking is not authority
    # to disclose memory/file/email content to a third party.
    capability("web.search", "read", R.MODERATE, True, False, D.NONE, "user", False, True, C.EXACT, S.NETWORK, None, one_text("query", 2000)),
    capability("weather.read", "read", R.MODERATE, True, False, D.NONE, "user", False, True, C.EXACT, S.NETWORK, None, one_text("city", 200)),
    capability("workspace.read", "read", R.MODERATE, False, True, D.NONE, "user", True, True, C.EXACT, S.NATIVE, "workspace", workspace_arguments),
    capability("workspace.create_text", "mutate", R.HIGH, False, False, D.NONE, "user", True, True, C.EXACT, S.NATIVE, "workspace", lambda a: workspace_arguments(a, write=True)),
    capability("email.send", "mutate", R.HIGH, True, True, D.IRREVERSIBLE, "user", False, True, C.EXACT, S.NETWORK, "email", email_send),
    capability("email.read", "read", R.MODERATE, False, True, D.NONE, "user", True, True, C.EXACT, S.NETWORK, "email", email_read),
    capability("memory.write", "mutate", R.MODERATE, False, True, D.RECOVERABLE, "user", True, True, C.EXACT, S.INTERNAL, "memory", memory_write),
    capability("task.submit", "mutate", R.MODERATE, True, False, D.NONE, "user", True, True, C.EXACT, S.INTERNAL, None, queue_submit),
    capability("task.status", "read", R.LOW, False, True, D.NONE, "user", True, True, C.EXACT, S.INTERNAL, None, task_status),
    capability("task.cancel", "mutate", R.MODERATE, False, False, D.RECOVERABLE, "user", False, True, C.EXACT, S.INTERNAL, None, task_cancel),
)

# Conservative, explicit declarations for legacy bundles. A consent ticket cannot
# enable these: they need bounded adapters/target verifiers, not a blanket grant.
_DEFERRED = (
    ("open_app", "desktop.open_app", S.DESKTOP, False, False, D.NONE, "user", True, "desktop"),
    ("check_messages", "message.read_legacy", S.DESKTOP, False, True, D.NONE, "user", True, "desktop"),
    ("prepare_message_reply", "message.draft_legacy", S.DOM, True, True, D.RECOVERABLE, "user", True, "desktop"),
    ("send_message", "message.send", S.DOM, True, True, D.IRREVERSIBLE, "user", False, "desktop"),
    ("reminder", "reminder.schedule", S.INTERNAL, False, True, D.RECOVERABLE, "user", True, "timers"),
    ("youtube_video", "media.youtube", S.DOM, True, False, D.NONE, "user", True, "desktop"),
    ("media_control", "media.control", S.DESKTOP, True, False, D.NONE, "user", True, "desktop"),
    ("screen_process", "sensors.vision", S.DESKTOP, True, True, D.NONE, "user", False, "desktop"),
    ("computer_settings", "system.settings", S.DESKTOP, True, True, D.IRREVERSIBLE, "admin", False, "desktop"),
    ("browser_control", "browser.legacy", S.DOM, True, True, D.IRREVERSIBLE, "user", False, "desktop"),
    ("desktop_control", "desktop.legacy", S.DESKTOP, True, True, D.IRREVERSIBLE, "user", False, "desktop"),
    ("code_helper", "code.generated", S.NATIVE, True, True, D.IRREVERSIBLE, "user", False, "workspace"),
    ("dev_agent", "code.developer_agent", S.NATIVE, True, True, D.IRREVERSIBLE, "user", False, "workspace"),
    ("computer_control", "desktop.input", S.DESKTOP, True, True, D.IRREVERSIBLE, "user", False, "desktop"),
    ("game_updater", "software.updater", S.NATIVE, True, True, D.IRREVERSIBLE, "admin", False, "workspace"),
    ("flight_finder", "travel.legacy", S.NETWORK, True, True, D.NONE, "user", True, "desktop"),
    ("graphics_quality", "ui.graphics", S.DESKTOP, False, False, D.RECOVERABLE, "user", True, "desktop"),
    ("jarvis_ui_control", "ui.control", S.DESKTOP, False, False, D.RECOVERABLE, "user", True, "desktop"),
    ("deep_research", "research.composite", S.NETWORK, True, True, D.RECOVERABLE, "user", True, "workspace"),
    ("create_presentation", "presentation.composite", S.NATIVE, True, True, D.RECOVERABLE, "user", True, "workspace"),
    ("file_processor", "file.processor_legacy", S.NATIVE, True, True, D.IRREVERSIBLE, "user", False, "workspace"),
    ("shutdown_jarvis", "system.self_shutdown", S.INTERNAL, False, False, D.RECOVERABLE, "user", True, "desktop"),
    ("specialized_runner", "worker.arbitrary_callable", S.NATIVE, True, True, D.IRREVERSIBLE, "user", False, "workspace"),
)

REGISTRY = MappingProxyType({
    **{e.capability.capability_id: e for e in _ENTRIES},
    **{cid: capability(cid, "mutate", R.CRITICAL, external, sensitive, destructive,
                       privilege, reversible, True, C.FORBIDDEN, surface, lock,
                       lambda a: dict(a), disabled="Legacy composite/ambient action is not admitted; migrate its bounded operations first.")
       for _, cid, surface, external, sensitive, destructive, privilege, reversible, lock in _DEFERRED},
})

TOOL_CAPABILITIES = MappingProxyType({
    "system_time": "system.time", "web_search": "web.search", "weather_report": "weather.read",
    "save_memory": "memory.write", "agent_task": "task.submit",
    "task_status": MappingProxyType({"get": "task.status", "all": "task.status", "cancel": "task.cancel"}),
    "file_controller": MappingProxyType({"read": "workspace.read", "create_file": "workspace.create_text", "write": "workspace.create_text"}),
    "email_control": MappingProxyType({"send": "email.send", "approve": "email.send", "read": "email.read"}),
    **{tool: cid for tool, cid, *_ in _DEFERRED},
})


def resolve_tool(tool: str, arguments: dict):
    canonical_arguments(arguments)  # reject non-data before any schema operation
    mapping = TOOL_CAPABILITIES.get(tool)
    if mapping is None:
        return None, dict(arguments)
    if isinstance(mapping, str):
        cid, args = mapping, dict(arguments)
    else:
        args = dict(arguments)
        action = args.pop("action", "get" if tool == "task_status" else None)
        if type(action) is not str:
            raise ValueError("An exact registered action is required.")
        cid = mapping.get(action)
        if cid is None:
            return None, dict(arguments)
        if cid == "task.status":
            args["action"] = action
    return cid, REGISTRY[cid].normalize(args)


def inventory() -> list[dict]:
    rows = []
    for cid, entry in REGISTRY.items():
        row = asdict(entry.capability)
        row["environments"] = sorted(row["environments"])
        row["tools"] = [name for name, mapping in TOOL_CAPABILITIES.items()
                        if mapping == cid or (not isinstance(mapping, str) and cid in mapping.values())]
        rows.append(row)
    return rows
