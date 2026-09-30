"""Small reviewed adapters. Legacy strings never become success evidence."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat

from core.action_contracts import ActionStatus, Evidence, ToolResult


def _verified(message, source, observation):
    return ToolResult(ActionStatus.SUCCEEDED, message, (
        Evidence(source, observation, datetime.now(timezone.utc).isoformat()),
    ))


@contextmanager
def _workspace_parent(context, relative, *, create=False):
    """Linux cloud adapter: no-follow directory handles prevent symlink traversal.

    Windows reparse-point handling is not certified; do not emulate success there.
    Directories must exist except for the app-owned workspace root.
    """
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise OSError("Secure workspace adapter unavailable on this platform.")
    root = context.workspace_root
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # Open every root component, not just the final directory, without symlinks.
    fd = os.open(root.anchor, flags)
    try:
        for index, component in enumerate(root.parts[1:] + tuple(relative.split("/")[:-1])):
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create or index >= len(root.parts) - 1:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, relative.split("/")[-1]
    finally:
        os.close(fd)


def _workspace_read(args, context):
    with _workspace_parent(context, args["path"]) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 65536:
                raise ValueError("Only bounded regular, non-hardlinked workspace files may be read.")
            data = stream.read(65537)
            if len(data) > 65536:
                raise ValueError("File grew beyond the read limit.")
    content = data.decode("utf-8")
    return _verified(content or "Empty workspace file.", "native.workspace.read",
                     f"{args['path']}: sha256={hashlib.sha256(data).hexdigest()}; bytes={len(data)}")


def _workspace_create(args, context):
    data = args["content"].encode("utf-8")
    with _workspace_parent(context, args["path"], create=True) as (parent, name):
        # Exclusive create: approval never permits overwriting an existing file.
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "w+b") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            observed = stream.read()
            if observed != data:
                raise OSError("Post-write verification failed.")
        os.fsync(parent)  # persist the new directory entry before journaling success
    return _verified(f"Created workspace file: {args['path']}", "native.workspace.readback",
                     f"{args['path']}: sha256={hashlib.sha256(observed).hexdigest()}; bytes={len(observed)}")


def invoke(capability_id, arguments, context, runtime):
    # Static canonical dispatch: no globals()/eval/import names from a model.
    if capability_id in {"system.time", "web.search", "weather.read"}:
        from agent.action_kernel import _dispatch
        tool = {"system.time": "system_time", "web.search": "web_search", "weather.read": "weather_report"}[capability_id]
        return _dispatch(tool, arguments)
    if capability_id == "workspace.read":
        return _workspace_read(arguments, context)
    if capability_id == "workspace.create_text":
        return _workspace_create(arguments, context)
    if capability_id in {"email.send", "email.read"}:
        from core.tenant import tenant_scope
        from actions.email_control import _send_gmail, _gmail_read
        with tenant_scope(None if context.owner_id == "local-owner" else context.owner_id):
            raw = (_send_gmail(arguments) if capability_id == "email.send"
                   else _gmail_read(arguments["message_id"]))
        # Legacy adapter loses provider receipt IDs. Do not infer success from prose.
        return ToolResult(ActionStatus.UNVERIFIED, str(raw or "No provider receipt returned."), error_code="legacy_unverified")
    if capability_id == "memory.write":
        from core.tenant import tenant_scope
        from memory.memory_manager import update_memory
        with tenant_scope(None if context.owner_id == "local-owner" else context.owner_id):
            update_memory({arguments["category"]: {arguments["key"]: {"value": arguments["value"]}}})
        return ToolResult(ActionStatus.UNVERIFIED, "Memory write returned; durable storage not independently verified.", error_code="legacy_unverified")
    if capability_id == "task.submit":
        from agent.task_queue import get_queue, TaskPriority
        priority = {"low": TaskPriority.LOW, "normal": TaskPriority.NORMAL, "high": TaskPriority.HIGH}[arguments["priority"]]
        queue = get_queue()
        task_id = queue.submit(arguments["goal"], priority=priority, owner_runtime=runtime)
        return _verified(f"Task queued (not completed): {task_id}", "native.task_queue", f"Queued task {task_id}")
    if capability_id in {"task.status", "task.cancel"}:
        import json
        from agent.task_queue import get_queue
        queue = get_queue()
        owner_id = context.owner_id
        if capability_id == "task.cancel":
            value = queue.cancel(arguments["task_id"], owner_id=owner_id)
            return _verified(f"Cancellation requested: {value}", "native.task_queue", f"Cancellation recorded: {value}")
        value = (queue.get_all_statuses(owner_id=owner_id) if arguments["action"] == "all"
                 else queue.get_status(arguments["task_id"], owner_id=owner_id))
        return _verified(json.dumps(value, ensure_ascii=False), "native.task_queue", "Read owner-scoped task state.")
    raise ValueError("No admitted adapter for capability.")
