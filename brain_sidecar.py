"""Headless entry point embedded beside the Tauri JARVIS shell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import sys

import psutil
import uvicorn

from api.jarvis_local_server import LocalBrainHost, create_local_brain_app
from api.nexus_sync_server import create_sync_app
from core.companion_gateway import resolve_companion_gateway


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JARVIS local brain sidecar")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-token", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--cloud-boost", action="store_true")
    return parser


def _watch_parent(parent_pid: int, host: LocalBrainHost) -> None:
    while True:
        if parent_pid <= 0 or not psutil.pid_exists(parent_pid):
            host.close()
            os._exit(0)
        time.sleep(2.0)


def _companion_server_listening(server: uvicorn.Server) -> bool:
    return bool(
        server.started
        and not server.should_exit
        and any(listener.is_serving() for listener in getattr(server, "servers", ()))
    )


def _serve_companion(app, host: str, port: int, brain_host: LocalBrainHost) -> None:
    brain_host.set_companion_listener(None)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    brain_host.set_companion_listener(lambda: _companion_server_listening(server))
    try:
        server.run()
    finally:
        brain_host.set_companion_listener(None)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--update-self-test":
        if len(arguments) != 2:
            return 2
        try:
            from core.build_info import read_build_info
            from core.ollama_bootstrap import load_brain_manifest
            info = read_build_info()
            load_brain_manifest()
            output = Path(arguments[1])
            if not output.is_absolute() or output.is_symlink():
                return 2
            with output.open("x", encoding="utf-8") as handle:
                json.dump({"ok": True, **info}, handle)
            return 0
        except Exception:
            return 1
    if arguments and arguments[0] == "--install-update":
        if len(arguments) != 4:
            return 2
        try:
            parent_pid = int(arguments[3])
            if parent_pid <= 0 or not Path(arguments[1]).is_absolute():
                return 2
        except (TypeError, ValueError):
            return 2
        try:
            from core.update_manager import run_windows_update_helper, UpdateState
            outcome = run_windows_update_helper(arguments[1], arguments[2], parent_pid)
            return 0 if outcome.state is UpdateState.APPLIED else 1
        except Exception:
            return 1
    args = _parser().parse_args(arguments)
    if not 1024 <= args.port <= 65535:
        raise ValueError("--port must be between 1024 and 65535")
    if len(args.ui_token) < 32:
        raise ValueError("--ui-token is too short")

    gateway = resolve_companion_gateway()
    host = LocalBrainHost(
        ui_token=args.ui_token,
        state_dir=Path(args.state_dir) if args.state_dir else None,
        allow_cloud=args.cloud_boost,
        companion_endpoint=gateway.endpoint if gateway is not None else None,
        parent_pid=args.parent_pid,
    )

    threading.Thread(
        target=_watch_parent,
        args=(args.parent_pid, host),
        name="jarvis-parent-watch",
        daemon=True,
    ).start()

    if gateway is not None:
        companion_app = create_sync_app(
            host.node,
            brain=host.brain,
            approvals=host.approvals,
            run_scheduler=True,
        )
        threading.Thread(
            target=_serve_companion,
            args=(companion_app, gateway.bind_host, gateway.port, host),
            name="jarvis-companion-gateway",
            daemon=True,
        ).start()

    app = create_local_brain_app(host)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
