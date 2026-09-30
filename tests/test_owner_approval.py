"""One-time companion biometric approval queue contracts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import tempfile
import unittest

from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager


NOW = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = EventStore(Path(self.tmp.name), "desktop")
        self.now = NOW
        self.manager = OwnerApprovalManager(self.store, clock=lambda: self.now)
        self.digest = hashlib.sha256(b"major-action").hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def test_biometric_verified_companion_can_approve_once(self):
        item = self.manager.create("Install major JARVIS update", self.digest)
        with self.assertRaises(PermissionError):
            self.manager.decide(
                item.approval_id,
                peer_id="phone",
                approved=True,
                user_verified=False,
            )

        decided = self.manager.decide(
            item.approval_id,
            peer_id="phone",
            approved=True,
            user_verified=True,
        )
        self.assertEqual(decided.state, "approved")
        self.assertEqual(decided.decided_by, "phone")

        consumed = self.manager.consume(item.approval_id, action_digest=self.digest)
        self.assertTrue(consumed.consumed)
        with self.assertRaises(PermissionError):
            self.manager.consume(item.approval_id, action_digest=self.digest)

    def test_approval_is_bound_to_exact_action_digest(self):
        item = self.manager.create("Sensitive operation", self.digest)
        self.manager.decide(
            item.approval_id,
            peer_id="phone",
            approved=True,
            user_verified=True,
        )
        other = hashlib.sha256(b"other").hexdigest()
        with self.assertRaises(PermissionError):
            self.manager.consume(item.approval_id, action_digest=other)

    def test_expired_request_disappears_from_pending(self):
        item = self.manager.create("Short approval", self.digest, ttl_seconds=15)
        self.assertEqual(len(self.manager.pending()), 1)
        self.now = NOW + timedelta(seconds=16)
        self.assertEqual(self.manager.pending(), ())
        self.assertEqual(self.manager.get(item.approval_id).state, "expired")


if __name__ == "__main__":
    unittest.main()
