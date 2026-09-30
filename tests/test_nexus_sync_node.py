"""Operational tests for NEXUS sync node, scheduler and standalone API."""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.nexus_sync_server import create_sync_app
from core.nexus.pairing import PairingManager
from core.nexus.peer_auth import DeviceSigner, PeerRole
from core.nexus.signed_transport import MEDIA_TYPE, batch_to_payload
from core.nexus.sync_daemon import SyncBatch, SyncReport
from core.nexus.sync_node import NexusSyncNode
from core.nexus.sync_scheduler import SyncScheduler
from core.secret_store import NoopSecretStore


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakePeer:
    def __init__(self, peer_id):
        self.peer_id = peer_id


class FakeRegistry:
    def __init__(self, peers):
        self.peers = list(peers)

    def active_sync_peers(self):
        return tuple(FakePeer(value) for value in self.peers)


class FakeDaemon:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def sync_peer(self, peer_id, transport):
        self.calls.append(peer_id)
        if self.reports:
            return self.reports.pop(0)
        return SyncReport(peer_id, None, 0, 0, 0, 0)


class SyncSchedulerTests(unittest.TestCase):
    def test_exponential_backoff_resets_after_success(self):
        clock = FakeClock()
        registry = FakeRegistry(["dell"])
        daemon = FakeDaemon([
            SyncReport("dell", "a", 1, 0, 1, 1, error="offline"),
            SyncReport("dell", "b", 1, 0, 1, 1, error="offline"),
            SyncReport("dell", "c", 1, 1, 0, 1),
        ])
        scheduler = SyncScheduler(
            daemon,
            registry,
            object(),
            retry_base_seconds=2,
            retry_max_seconds=30,
            idle_seconds=20,
            clock=clock,
        )

        scheduler.tick()
        first = scheduler.state("dell")
        self.assertEqual(first.failures, 1)
        self.assertEqual(first.next_due, 102)

        clock.advance(1)
        self.assertEqual(scheduler.tick(), ())
        self.assertEqual(len(daemon.calls), 1)

        clock.advance(1)
        scheduler.tick()
        second = scheduler.state("dell")
        self.assertEqual(second.failures, 2)
        self.assertEqual(second.next_due, 106)

        clock.advance(4)
        scheduler.tick()
        success = scheduler.state("dell")
        self.assertEqual(success.failures, 0)
        self.assertEqual(success.next_due, 126)

    def test_remaining_work_uses_short_drain_interval(self):
        clock = FakeClock()
        daemon = FakeDaemon([SyncReport("dell", "x", 2, 2, 5, 2)])
        scheduler = SyncScheduler(
            daemon,
            FakeRegistry(["dell"]),
            object(),
            drain_seconds=0.5,
            clock=clock,
        )
        scheduler.tick()
        self.assertEqual(scheduler.state("dell").next_due, 100.5)

    def test_revoked_peer_is_removed_from_schedule(self):
        clock = FakeClock()
        registry = FakeRegistry(["dell"])
        daemon = FakeDaemon([SyncReport("dell", None, 0, 0, 0, 0)])
        scheduler = SyncScheduler(daemon, registry, object(), clock=clock)
        scheduler.tick()
        self.assertIn("dell", scheduler._state)
        registry.peers.clear()
        scheduler.tick()
        self.assertNotIn("dell", scheduler._state)


class SyncNodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.secrets = NoopSecretStore()

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_persists_through_secret_store_and_peer_revokes(self):
        node = NexusSyncNode(self.root / "hp", "hp", secret_store=self.secrets)
        identity = node.identity
        restarted = NexusSyncNode(self.root / "hp", "hp", secret_store=self.secrets)
        self.assertEqual(restarted.identity, identity)

        peer_signer = DeviceSigner.generate()
        peer = node.trust_peer(
            "dell",
            peer_signer.public_b64,
            "http://dell.tailnet.ts.net:8765",
        )
        self.assertEqual(peer.peer_id, "dell")
        self.assertTrue(node.revoke_peer("dell"))
        self.assertIsNone(node.registry.get_peer("dell"))

    def test_recover_materializes_local_pending_event(self):
        node = NexusSyncNode(self.root / "node", "node", secret_store=self.secrets)
        event = node.store.create_event(
            "memory.upsert",
            "memory:recover",
            {"value": "kept"},
            entity_type="memory.fact",
        )
        self.assertEqual(node.store.pending_local_count(), 1)

        result = node.recover()

        self.assertEqual(result["local_applied"], 1)
        self.assertEqual(node.store.pending_local_count(), 0)
        self.assertEqual(node.applier.get_snapshot("memory:recover").value, "kept")
        self.assertTrue(node.store.seen(event.id))

    def test_standalone_api_exposes_public_identity_and_requires_signed_media(self):
        node = NexusSyncNode(self.root / "api", "node-api", secret_store=self.secrets)
        app = create_sync_app(node, run_scheduler=False)
        with TestClient(app) as client:
            health = client.get("/nexus/sync/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["device_id"], "node-api")

            identity = client.get("/nexus/sync/v1/identity")
            self.assertEqual(identity.status_code, 200)
            self.assertEqual(identity.json()["public_key"], node.identity.public_key)

            rejected = client.post(
                "/nexus/sync/v1/batch",
                content=b"{}",
                headers={"content-type": "application/json"},
            )
            self.assertEqual(rejected.status_code, 415)

    def test_pairing_http_intake_requires_local_owner_approval(self):
        desktop = NexusSyncNode(
            self.root / "pair-desktop",
            "desktop",
            secret_store=NoopSecretStore(),
        )
        phone = NexusSyncNode(
            self.root / "pair-phone",
            "phone",
            secret_store=NoopSecretStore(),
        )
        offer = desktop.pairing.create_offer(
            "http://desktop.tailnet.ts.net:8765"
        )
        request = PairingManager.build_request(
            offer,
            "phone",
            phone.signer,
            candidate_role=PeerRole.COMPANION,
        )

        app = create_sync_app(desktop, run_scheduler=False)
        with TestClient(app) as client:
            response = client.post(
                "/nexus/pair/v1/request",
                json=request.to_dict(),
            )
            self.assertEqual(response.status_code, 202)
            self.assertTrue(response.json()["owner_approval_required"])

        self.assertIsNone(desktop.registry.get_peer("phone"))
        self.assertEqual(len(desktop.pairing.pending()), 1)

        peer = desktop.pairing.approve(offer.pairing_id)
        self.assertEqual(peer.peer_id, "phone")
        self.assertEqual(peer.role, PeerRole.COMPANION)
        self.assertIsNone(peer.endpoint)
        self.assertIsNotNone(desktop.registry.get_peer("phone"))
        self.assertEqual(desktop.registry.active_sync_peers(), ())

    def test_pairing_http_rejects_bad_proof(self):
        desktop = NexusSyncNode(
            self.root / "pair-bad-desktop",
            "desktop-bad",
            secret_store=NoopSecretStore(),
        )
        phone = NexusSyncNode(
            self.root / "pair-bad-phone",
            "phone-bad",
            secret_store=NoopSecretStore(),
        )
        offer = desktop.pairing.create_offer(
            "http://desktop-bad.tailnet.ts.net:8765"
        )
        request = PairingManager.build_request(
            offer,
            "phone-bad",
            phone.signer,
            candidate_role=PeerRole.COMPANION,
        )
        value = request.to_dict()
        value["proof"] = "0" * 64

        app = create_sync_app(desktop, run_scheduler=False)
        with TestClient(app) as client:
            response = client.post(
                "/nexus/pair/v1/request",
                json=value,
            )
            self.assertEqual(response.status_code, 403)

        self.assertEqual(desktop.pairing.pending(), ())
        self.assertIsNone(desktop.registry.get_peer("phone-bad"))

    def test_standalone_api_accepts_valid_signed_batch(self):
        hp = NexusSyncNode(self.root / "hp2", "hp2", secret_store=NoopSecretStore())
        dell = NexusSyncNode(self.root / "dell2", "dell2", secret_store=NoopSecretStore())
        hp.trust_peer("dell2", dell.identity.public_key, "http://dell2.tailnet.ts.net:8765")
        dell.trust_peer("hp2", hp.identity.public_key, "http://hp2.tailnet.ts.net:8765")

        event = hp.store.create_event(
            "memory.upsert",
            "memory:http",
            {"value": "synced"},
            entity_type="memory.fact",
        )
        hp.applier.apply_local_event(event)
        batch = SyncBatch("hp2", "http-batch", (event,))
        encoded = hp.authenticator.sign(
            "sync.batch",
            "dell2",
            batch_to_payload(batch),
            message_id=batch.batch_id,
        )

        app = create_sync_app(dell, run_scheduler=False)
        with TestClient(app) as client:
            response = client.post(
                "/nexus/sync/v1/batch",
                content=encoded,
                headers={"content-type": MEDIA_TYPE},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"].split(";")[0], MEDIA_TYPE)

        self.assertEqual(
            dell.applier.get_snapshot("memory:http").value,
            "synced",
        )


if __name__ == "__main__":
    unittest.main()
