"""One-time explicit node/companion pairing tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.pairing import PairingManager, PairingRequest
from core.nexus.peer_auth import DeviceSigner, PeerRegistry, PeerRole


NOW = datetime(2026, 9, 30, 13, 30, tzinfo=timezone.utc)


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name) / "desktop", "hp")
        self.registry = PeerRegistry(self.store)
        self.desktop_signer = DeviceSigner.generate()
        self.manager = PairingManager(
            self.registry,
            "hp",
            self.desktop_signer,
            clock=lambda: NOW,
        )
        self.phone_signer = DeviceSigner.generate()

    def tearDown(self):
        self.tmp.cleanup()

    def offer(self, ttl=300):
        return self.manager.create_offer(
            "http://100.64.0.10:8765",
            ttl_seconds=ttl,
        )

    def companion_request(self, offer):
        return PairingManager.build_request(
            offer,
            "phone",
            self.phone_signer,
            candidate_role=PeerRole.COMPANION,
        )

    def node_request(self, offer):
        return PairingManager.build_request(
            offer,
            "dell",
            self.phone_signer,
            "http://100.64.0.20:8765",
            candidate_role=PeerRole.NODE,
        )

    def test_companion_request_stays_pending_until_owner_approval(self):
        offer = self.offer()
        request = self.companion_request(offer)

        pending = self.manager.receive_request(request)

        self.assertEqual(pending.candidate_device, "phone")
        self.assertEqual(pending.candidate_role, PeerRole.COMPANION)
        self.assertIsNone(pending.candidate_endpoint)
        self.assertIsNone(self.registry.get_peer("phone"))

        peer = self.manager.approve(offer.pairing_id)

        self.assertEqual(peer.peer_id, "phone")
        self.assertEqual(peer.role, PeerRole.COMPANION)
        self.assertIsNone(peer.endpoint)
        self.assertEqual(self.registry.active_sync_peers(), ())

    def test_node_request_retains_sync_endpoint(self):
        offer = self.offer()
        pending = self.manager.receive_request(self.node_request(offer))
        peer = self.manager.approve(offer.pairing_id)

        self.assertEqual(pending.candidate_role, PeerRole.NODE)
        self.assertEqual(peer.role, PeerRole.NODE)
        self.assertEqual(peer.endpoint, "http://100.64.0.20:8765")
        self.assertEqual(self.registry.active_sync_peers()[0].peer_id, "dell")

    def test_offer_survives_manager_restart_without_plaintext_secret_storage(self):
        offer = self.offer()
        request = self.companion_request(offer)

        restarted = PairingManager(
            PeerRegistry(EventStore(Path(self.tmp.name) / "desktop", "hp")),
            "hp",
            self.desktop_signer,
            clock=lambda: NOW,
        )
        pending = restarted.receive_request(request)

        self.assertEqual(pending.candidate_device, "phone")
        self.assertEqual(pending.candidate_role, PeerRole.COMPANION)

    def test_tampered_proof_is_rejected(self):
        offer = self.offer()
        request = self.companion_request(offer)
        bad = PairingRequest(
            version=request.version,
            pairing_id=request.pairing_id,
            candidate_device=request.candidate_device,
            candidate_public_key=request.candidate_public_key,
            candidate_role=request.candidate_role,
            candidate_endpoint=request.candidate_endpoint,
            proof="0" * 64,
        )

        with self.assertRaises(PermissionError):
            self.manager.receive_request(bad)
        self.assertIsNone(self.registry.get_peer("phone"))

    def test_expired_offer_is_rejected(self):
        offer = self.offer(ttl=10)
        late = PairingManager(
            self.registry,
            "hp",
            self.desktop_signer,
            clock=lambda: NOW + timedelta(seconds=11),
        )
        with self.assertRaises(PermissionError):
            late.receive_request(self.companion_request(offer))

    def test_pairing_offer_is_one_time(self):
        offer = self.offer()
        request = self.companion_request(offer)
        self.manager.receive_request(request)
        with self.assertRaises(PermissionError):
            self.manager.receive_request(request)

    def test_cancelled_pairing_cannot_be_used(self):
        offer = self.offer()
        self.assertTrue(self.manager.cancel(offer.pairing_id))
        with self.assertRaises(PermissionError):
            self.manager.receive_request(self.companion_request(offer))

    def test_companion_cannot_claim_inbound_endpoint(self):
        offer = self.offer()
        with self.assertRaises(ValueError):
            PairingManager.build_request(
                offer,
                "phone",
                self.phone_signer,
                "http://100.64.0.99:8765",
                candidate_role=PeerRole.COMPANION,
            )

    def test_approve_requires_pending_request(self):
        offer = self.offer()
        with self.assertRaises(PermissionError):
            self.manager.approve(offer.pairing_id)


if __name__ == "__main__":
    unittest.main()
