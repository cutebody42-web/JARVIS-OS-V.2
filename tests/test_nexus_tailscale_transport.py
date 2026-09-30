"""Offline tests for signed NEXUS HTTP transport over a Tailscale path."""

from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone

from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier
from core.nexus.sync_daemon import SyncDaemon
from core.nexus.tailscale_transport import (
    DEVICE_HEADER,
    NONCE_HEADER,
    PROTOCOL_HEADER,
    SIGNATURE_HEADER,
    SYNC_PATH,
    TIMESTAMP_HEADER,
    NonceReplayCache,
    PeerEndpoint,
    PeerRegistry,
    SignedSyncHttpEndpoint,
    SyncAuthenticationError,
    TailscaleHttpTransport,
    generate_pairwise_secret,
)


class MemoryKeyStore:
    def __init__(self, keys):
        self.keys = dict(keys)

    def get_secret(self, peer_id):
        return self.keys[peer_id]


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self.content = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class EndpointSession:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.calls = []

    def post(self, url, data, headers, timeout):
        self.calls.append((url, bytes(data), dict(headers), timeout))
        status, response_headers, response_body = self.endpoint.handle(
            headers=headers,
            body=bytes(data),
        )
        return FakeResponse(status, response_headers, response_body)


class Node:
    def __init__(self, root: Path, device_id: str):
        self.store = EventStore(root, device_id)
        self.applier = MergeApplier(self.store)
        self.sync = SyncDaemon(self.store, self.applier)

    def author(self, entity_id="memory:1", value="hello"):
        event = self.store.create_event(
            "memory.upsert",
            entity_id,
            {"value": value},
            entity_type="memory.fact",
        )
        self.applier.apply_local_event(event)
        return event


class TailscaleTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hp = Node(root / "hp", "hp")
        self.dell = Node(root / "dell", "dell")
        self.secret = generate_pairwise_secret()
        self.hp_keys = MemoryKeyStore({"dell": self.secret})
        self.dell_keys = MemoryKeyStore({"hp": self.secret})
        self.hp_registry = PeerRegistry({
            "dell": PeerEndpoint("dell", "dell.tailnet.ts.net", port=8443)
        })
        self.dell_registry = PeerRegistry({
            "hp": PeerEndpoint("hp", "hp.tailnet.ts.net", port=8443)
        })
        fixed = datetime(2026, 9, 30, 12, 30, tzinfo=timezone.utc)
        self.clock = lambda: fixed
        self.endpoint = SignedSyncHttpEndpoint(
            daemon=self.dell.sync,
            registry=self.dell_registry,
            key_store=self.dell_keys,
            replay_cache=NonceReplayCache(),
            clock=self.clock,
        )
        self.session = EndpointSession(self.endpoint)
        self.transport = TailscaleHttpTransport(
            local_device_id="hp",
            registry=self.hp_registry,
            key_store=self.hp_keys,
            session=self.session,
            clock=self.clock,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_signed_delivery(self):
        event = self.hp.author(value="physics")

        report = self.hp.sync.sync_peer("dell", self.transport)

        self.assertEqual(report.acked, 1)
        self.assertEqual(report.remaining, 0)
        self.assertEqual(
            self.dell.applier.get_snapshot(event.entity_id).value,
            "physics",
        )
        self.assertTrue(self.session.calls[0][0].endswith(SYNC_PATH))

    def test_replaying_identical_signed_request_is_rejected(self):
        self.hp.author(value="one")
        self.hp.sync.sync_peer("dell", self.transport)
        _url, body, headers, _timeout = self.session.calls[0]

        status, _, _ = self.endpoint.handle(headers=headers, body=body)

        self.assertEqual(status, 401)

    def test_tampered_body_is_rejected_before_merge(self):
        self.hp.author(entity_id="memory:tamper", value="safe")
        self.hp.sync.sync_peer("dell", self.transport)
        _url, body, headers, _timeout = self.session.calls[0]

        tampered = body.replace(b'"safe"', b'"evil"')
        status, _, _ = self.endpoint.handle(headers=headers, body=tampered)

        self.assertEqual(status, 401)
        self.assertEqual(
            self.dell.applier.get_snapshot("memory:tamper").value,
            "safe",
        )

    def test_wrong_signed_device_identity_is_rejected(self):
        self.hp.author(value="identity")
        self.hp.sync.sync_peer("dell", self.transport)
        _url, body, headers, _timeout = self.session.calls[0]
        forged = dict(headers)
        forged[DEVICE_HEADER] = "unknown-node"

        status, _, _ = self.endpoint.handle(headers=forged, body=body)

        self.assertEqual(status, 401)

    def test_disabled_peer_cannot_be_used(self):
        disabled = PeerRegistry({
            "dell": PeerEndpoint(
                "dell", "dell.tailnet.ts.net", enabled=False
            )
        })
        transport = TailscaleHttpTransport(
            local_device_id="hp",
            registry=disabled,
            key_store=self.hp_keys,
            session=self.session,
            clock=self.clock,
        )
        event = self.hp.author(value="blocked")

        report = self.hp.sync.sync_peer("dell", transport)

        self.assertEqual(report.acked, 0)
        self.assertEqual(report.remaining, 1)

    def test_endpoint_headers_are_application_authenticated(self):
        self.hp.author(value="headers")
        self.hp.sync.sync_peer("dell", self.transport)
        _url, _body, headers, _timeout = self.session.calls[0]

        self.assertEqual(headers[DEVICE_HEADER], "hp")
        self.assertEqual(headers[PROTOCOL_HEADER], "1")
        self.assertTrue(headers[TIMESTAMP_HEADER].endswith("Z"))
        self.assertTrue(headers[NONCE_HEADER])
        self.assertEqual(len(headers[SIGNATURE_HEADER]), 64)


if __name__ == "__main__":
    unittest.main()
