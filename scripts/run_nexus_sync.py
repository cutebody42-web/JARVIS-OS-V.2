"""Run one explicit NEXUS device-sync node."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path

import uvicorn

from api.nexus_sync_server import create_sync_app
from core.nexus.sync_node import NexusSyncNode


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run NEXUS signed device sync.")
    value.add_argument("--state-dir", required=True)
    value.add_argument("--device-id", required=True)
    value.add_argument("--bind", default="127.0.0.1",
                       help="Literal local address. Prefer the device's Tailscale IP.")
    value.add_argument("--port", type=int, default=8765)
    value.add_argument("--allow-any-interface", action="store_true")
    value.add_argument("--no-scheduler", action="store_true")
    return value


def _validate_bind(value: str, allow_any: bool) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError("--bind must be a literal local IP address") from exc
    if address.is_unspecified and not allow_any:
        raise ValueError(
            "Refusing an all-interface bind without --allow-any-interface; "
            "prefer the exact Tailscale interface address."
        )
    return str(address)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in 1..65535")
    host = _validate_bind(args.bind, args.allow_any_interface)
    node = NexusSyncNode(Path(args.state_dir), args.device_id)
    app = create_sync_app(node, run_scheduler=not args.no_scheduler)
    uvicorn.run(app, host=host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
