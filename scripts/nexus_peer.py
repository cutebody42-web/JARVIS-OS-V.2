"""Owner CLI for NEXUS device identity, trust and revocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.nexus.pairing import PairingManager, PairingOffer
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

    offer = sub.add_parser("pair-offer")
    offer.add_argument("--endpoint", required=True)
    offer.add_argument("--ttl", type=int, default=300)

    request = sub.add_parser("pair-request")
    request.add_argument("--offer-json", required=True)
    request.add_argument("--endpoint", required=True)

    sub.add_parser("pair-pending")

    approve = sub.add_parser("pair-approve")
    approve.add_argument("--pairing-id", required=True)

    cancel = sub.add_parser("pair-cancel")
    cancel.add_argument("--pairing-id", required=True)

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

    if args.command == "pair-offer":
        offer = node.pairing.create_offer(args.endpoint, ttl_seconds=args.ttl)
        print(json.dumps(offer.public_payload(), indent=2))
        return 0

    if args.command == "pair-request":
        try:
            offer_value = json.loads(args.offer_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--offer-json must contain valid pairing JSON") from exc
        offer = PairingOffer.from_dict(offer_value)
        request = PairingManager.build_request(
            offer,
            node.identity.device_id,
            node.signer,
            args.endpoint,
        )
        print(json.dumps(request.to_dict(), indent=2))
        return 0

    if args.command == "pair-pending":
        print(json.dumps([
            {
                "pairing_id": item.pairing_id,
                "candidate_device": item.candidate_device,
                "candidate_public_key": item.candidate_public_key,
                "candidate_endpoint": item.candidate_endpoint,
                "created_at": item.created_at,
            }
            for item in node.pairing.pending()
        ], indent=2))
        return 0

    if args.command == "pair-approve":
        peer = node.pairing.approve(args.pairing_id)
        print(json.dumps({
            "approved": True,
            "peer_id": peer.peer_id,
            "endpoint": peer.endpoint,
        }, indent=2))
        return 0

    if args.command == "pair-cancel":
        print(json.dumps({
            "pairing_id": args.pairing_id,
            "cancelled": node.pairing.cancel(args.pairing_id),
        }, indent=2))
        return 0

    raise RuntimeError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
