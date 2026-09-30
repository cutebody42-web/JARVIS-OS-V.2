"""One-time explicit device/phone pairing tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.pairing import PairingManager, PairingRequest
from core.nexus.peer_auth import DeviceSigner, PeerRegistry


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
            "http://hp.tailnet.ts.net:8765",
            ttl_seconds=ttl,
        )

    def request(self, offer):
        return PairingManager.build_request(
            offer,
            "phone",
            self.phone_signer,
            "http://phone.tailnet.ts.net:8765",
        )

    def test_valid_request_stays_pending_until_owner_approval(self):
        offer = self.offer()
        request = self.request(offer)

        pending = self.manager.receive_request(request)

        self.assertEqual(pending.candidate_device, "phone")
        self.assertIsNone(self.registry.get_peer("phone"))
        self.assertEqual(len(self.manager.pending()), 1)

        peer = self.manager.approve(offer.pairing_id)

        self.assertEqual(peer.peer_id, "phone")
        self.assertIsNotNone(self.registry.get_peer("phone"))
        self.assertEqual(self.manager.pending(), ())

    def test_offer_survives_manager_restart_without_plaintext_secret_storage(self):
        offer = self.offer()
        request = self.request(offer)

        restarted = PairingManager(
            PeerRegistry(EventStore(Path(self.tmp.name) / "desktop", "hp")),
            "hp",
            self.desktop_signer,
            clock=lambda: NOW,
        )
        pending = restarted.receive_request(request)

        self.assertEqual(pending.candidate_device, "phone")

    def test_tampered_proof_is_rejected(self):
        offer = self.offer()
        request = self.request(offer)
        bad = PairingRequest(
            request.version,
            request.pairing_id,
            request.candidate_device,
            request.candidate_public_key,
            request.candidate_endpoint,
            "0" * 64,
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
            late.receive_request(self.request(offer))

    def test_pairing_offer_is_one_time(self):
        offer = self.offer()
        request = self.request(offer)
        self.manager.receive_request(request)

        with self.assertRaises(PermissionError):
            self.manager.receive_request(request)

    def test_cancelled_pairing_cannot_be_used(self):
        offer = self.offer()
        self.assertTrue(self.manager.cancel(offer.pairing_id))

        with self.assertRaises(PermissionError):
            self.manager.receive_request(self.request(offer))

    def test_approve_requires_pending_request(self):
        offer = self.offer()
        with self.assertRaises(PermissionError):
            self.manager.approve(offer.pairing_id)

    def test_endpoint_url_confusion_is_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.create_offer(
                "http://user:pass@hp.tailnet.ts.net:8765"
            )


if __name__ == "__main__":
    unittest.main()
