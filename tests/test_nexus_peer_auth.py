"""Authenticated two-node sync tests with real Ed25519 signatures."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.merge_applier import MergeApplier
from core.nexus.peer_auth import (
    DeviceSigner,
    PeerAuthenticator,
    PeerRegistry,
    normalize_peer_endpoint,
)
from core.nexus.signed_transport import (
    MEDIA_TYPE,
    SignedHTTPSyncTransport,
    SignedSyncEndpoint,
    SyncTransportError,
)
from core.nexus.sync_daemon import SyncDaemon


NOW = datetime(2026, 9, 30, 12, 30, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, body=b"", status=200, content_type=MEDIA_TYPE):
        self._body = body
        self.status_code = status
        self.headers = {"content-type": content_type}
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for index in range(0, len(self._body), chunk_size):
            yield self._body[index:index + chunk_size]

    def close(self):
        self.closed = True


class EndpointSession:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.calls = []

    def post(self, url, *, data, headers, timeout, allow_redirects, stream):
        self.calls.append({
            "url": url,
            "data": data,
            "headers": headers,
            "timeout": timeout,
            "allow_redirects": allow_redirects,
            "stream": stream,
        })
        return FakeResponse(self.endpoint.handle_batch(data))


class StaticSession:
    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


class Node:
    def __init__(self, root: Path, device_id: str):
        self.store = EventStore(root, device_id)
        self.applier = MergeApplier(self.store)
        self.daemon = SyncDaemon(self.store, self.applier)
        self.registry = PeerRegistry(self.store)
        self.signer = DeviceSigner.generate()
        self.auth = PeerAuthenticator(
            device_id,
            self.signer,
            self.registry,
            clock=lambda: NOW,
        )
        self.endpoint = SignedSyncEndpoint(self.daemon, self.auth)

    def trust(self, other: "Node", endpoint: str):
        return self.registry.trust_peer(
            other.store.device_id,
            other.signer.public_b64,
            endpoint,
        )

    def author(self, value):
        event = self.store.create_event(
            "memory.upsert",
            "memory:test",
            {"value": value},
            entity_type="memory.fact",
        )
        self.applier.apply_local_event(event)
        return event


class PeerAuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hp = Node(root / "hp", "hp-660")
        self.dell = Node(root / "dell", "dell-3420")
        self.hp.trust(self.dell, "http://dell-3420.tailnet.ts.net:8765")
        self.dell.trust(self.hp, "http://hp-660.tailnet.ts.net:8765")

    def tearDown(self):
        self.tmp.cleanup()

    def test_signed_http_transport_converges_two_nodes(self):
        event = self.hp.author("physics")
        session = EndpointSession(self.dell.endpoint)
        transport = SignedHTTPSyncTransport(
            "hp-660",
            self.hp.registry,
            self.hp.auth,
            session=session,
        )

        report = self.hp.daemon.sync_peer("dell-3420", transport)

        self.assertEqual(report.acked, 1)
        self.assertEqual(report.remaining, 0)
        self.assertEqual(
            self.dell.applier.get_snapshot("memory:test").value,
            "physics",
        )
        self.assertEqual(
            self.hp.daemon.delivery_state("dell-3420", event.id)["state"],
            "acked",
        )
        self.assertFalse(session.calls[0]["allow_redirects"])
        self.assertTrue(session.calls[0]["stream"])
        self.assertEqual(
            session.calls[0]["url"],
            "http://dell-3420.tailnet.ts.net:8765/nexus/sync/v1/batch",
        )

    def test_tampered_signed_batch_is_rejected_before_event_store(self):
        event = self.hp.author("safe")
        batch = self.hp.daemon._pending_for_peer("dell-3420")
        from core.nexus.sync_daemon import SyncBatch
        envelope = self.hp.auth.sign(
            "sync.batch",
            "dell-3420",
            {
                "protocol_version": 1,
                "sender_device": "hp-660",
                "batch_id": "tamper-batch",
                "events": [event.to_dict()],
            },
            message_id="tamper-batch",
        )
        tampered = bytearray(envelope)
        tampered[-10] = ord("A") if tampered[-10] != ord("A") else ord("B")

        with self.assertRaises((PermissionError, ValueError)):
            self.dell.endpoint.handle_batch(bytes(tampered))
        self.assertFalse(self.dell.store.seen(event.id))

    def test_revoked_peer_cannot_send_or_receive(self):
        event = self.hp.author("revoked")
        self.assertTrue(self.dell.registry.revoke_peer("hp-660"))

        from core.nexus.signed_transport import batch_to_payload
        from core.nexus.sync_daemon import SyncBatch
        batch = SyncBatch("hp-660", "revoked-batch", (event,))
        encoded = self.hp.auth.sign(
            "sync.batch",
            "dell-3420",
            batch_to_payload(batch),
            message_id=batch.batch_id,
        )

        with self.assertRaises(PermissionError):
            self.dell.endpoint.handle_batch(encoded)

        self.assertTrue(self.hp.registry.revoke_peer("dell-3420"))
        with self.assertRaises(PermissionError):
            self.hp.auth.sign("sync.batch", "dell-3420", {"x": 1})

    def test_wrong_public_key_is_rejected(self):
        rogue = DeviceSigner.generate()
        self.dell.registry.trust_peer(
            "hp-660",
            rogue.public_b64,
            "http://hp-660.tailnet.ts.net:8765",
        )
        encoded = self.hp.auth.sign(
            "sync.batch",
            "dell-3420",
            {
                "protocol_version": 1,
                "sender_device": "hp-660",
                "batch_id": "wrong-key",
                "events": [],
            },
            message_id="wrong-key",
        )
        with self.assertRaises(PermissionError):
            self.dell.auth.verify(encoded, expected_kind="sync.batch")

    def test_stale_envelope_fails_closed(self):
        stale_auth = PeerAuthenticator(
            "hp-660",
            self.hp.signer,
            self.hp.registry,
            clock=lambda: NOW - timedelta(hours=1),
        )
        encoded = stale_auth.sign("sync.batch", "dell-3420", {"x": 1})
        with self.assertRaises(PermissionError):
            self.dell.auth.verify(encoded, expected_kind="sync.batch")

    def test_redirect_and_bad_content_type_are_rejected(self):
        event = self.hp.author("x")
        for response in (
            FakeResponse(status=307),
            FakeResponse(b"{}", status=200, content_type="text/html"),
        ):
            transport = SignedHTTPSyncTransport(
                "hp-660",
                self.hp.registry,
                self.hp.auth,
                session=StaticSession(response),
            )
            report = self.hp.daemon.sync_peer("dell-3420", transport)
            self.assertEqual(report.acked, 0)
            self.assertEqual(report.remaining, 1)
            self.assertEqual(report.error, "SyncTransportError")

    def test_endpoint_validation_rejects_url_confusion(self):
        for value in (
            "ftp://dell.tailnet.ts.net",
            "http://user:pass@dell.tailnet.ts.net",
            "http://dell.tailnet.ts.net/path",
            "http://dell.tailnet.ts.net?next=evil",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_peer_endpoint(value)


if __name__ == "__main__":
    unittest.main()
