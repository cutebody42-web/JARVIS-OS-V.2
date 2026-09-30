"""Owner CLI for NEXUS device identity, trust and revocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.nexus.sync_node import NexusSyncNode


def _node(args) -> NexusSyncNode:
    return NexusSyncNode(Path(args.state_dir), args.device_id)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Manage explicit NEXUS peer trust.")
    value.add_argument("--state-dir", required=True)
    value.add_argument("--device-id", required=True)
    sub = value.add_subparsers(dest="command", required=True)

    sub.add_parser("identity")

    trust = sub.add_parser("trust")
    trust.add_argument("--peer-id", required=True)
    trust.add_argument("--public-key", required=True)
    trust.add_argument("--endpoint", required=True)

    revoke = sub.add_parser("revoke")
    revoke.add_argument("--peer-id", required=True)

    sub.add_parser("list")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    node = _node(args)

    if args.command == "identity":
        print(json.dumps({
            "device_id": node.identity.device_id,
            "public_key": node.identity.public_key,
            "fingerprint": node.signer.fingerprint,
        }, indent=2))
        return 0

    if args.command == "trust":
        peer = node.trust_peer(args.peer_id, args.public_key, args.endpoint)
        print(json.dumps({
            "trusted": True,
            "peer_id": peer.peer_id,
            "endpoint": peer.endpoint,
        }, indent=2))
        return 0

    if args.command == "revoke":
        print(json.dumps({
            "peer_id": args.peer_id,
            "revoked": node.revoke_peer(args.peer_id),
        }, indent=2))
        return 0

    if args.command == "list":
        print(json.dumps([
            {
                "peer_id": peer.peer_id,
                "endpoint": peer.endpoint,
                "revoked": peer.revoked,
            }
            for peer in node.registry.active_peers()
        ], indent=2))
        return 0

    raise RuntimeError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
