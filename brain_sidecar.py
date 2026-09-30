"""Headless entry point embedded beside the Tauri JARVIS shell."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time

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


def _watch_parent(parent_pid: int) -> None:
    while True:
        if parent_pid <= 0 or not psutil.pid_exists(parent_pid):
            os._exit(0)
        time.sleep(2.0)


def _serve_companion(app, host: str, port: int) -> None:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    uvicorn.Server(config).run()


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not 1024 <= args.port <= 65535:
        raise ValueError("--port must be between 1024 and 65535")
    if len(args.ui_token) < 32:
        raise ValueError("--ui-token is too short")

    threading.Thread(
        target=_watch_parent,
        args=(args.parent_pid,),
        name="jarvis-parent-watch",
        daemon=True,
    ).start()

    gateway = resolve_companion_gateway()
    host = LocalBrainHost(
        ui_token=args.ui_token,
        state_dir=Path(args.state_dir) if args.state_dir else None,
        allow_cloud=args.cloud_boost,
        companion_endpoint=gateway.endpoint if gateway is not None else None,
    )

    if gateway is not None:
        companion_app = create_sync_app(
            host.node,
            brain=host.brain,
            run_scheduler=True,
        )
        threading.Thread(
            target=_serve_companion,
            args=(companion_app, gateway.bind_host, gateway.port),
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
