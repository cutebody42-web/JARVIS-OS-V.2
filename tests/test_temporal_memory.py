import tempfile
import unittest
from pathlib import Path

from core.temporal_memory import TemporalMemoryError, TemporalMemoryStore


class TemporalMemoryTests(unittest.TestCase):
    def test_claims_preserve_provenance_and_do_not_auto_verify(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            claim = store.add_claim(
                subject="project:jarvis", predicate="status", value="alpha", source="owner:chat",
                observed_at="2026-10-07T10:00:00+00:00", confidence=0.8,
                metadata={"kind": "owner_statement"},
            )
            self.assertEqual(claim.value, "alpha")
            self.assertEqual(claim.source, "owner:chat")
            self.assertFalse(claim.verified)
            self.assertTrue(claim.active)
            self.assertEqual(claim.metadata["kind"], "owner_statement")

    def test_supersede_closes_old_validity_without_erasing_history(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            old = store.add_claim(subject="device:laptop", predicate="state", value="offline", source="sensor:local", observed_at="2026-10-07T10:00:00+00:00", verified=True)
            new = store.supersede(old.claim_id, value="online", source="sensor:local", observed_at="2026-10-07T11:00:00+00:00", confidence=0.95, verified=True)
            old_after = store.get(old.claim_id)
            self.assertFalse(old_after.active)
            self.assertEqual(old_after.valid_to, "2026-10-07T11:00:00.000000+00:00")
            self.assertEqual(old_after.superseded_by, new.claim_id)
            self.assertTrue(new.active)
            self.assertEqual(store.active_claims(subject="device:laptop")[0].value, "online")
            self.assertEqual([item.value for item in store.history(subject="device:laptop")], ["online", "offline"])

    def test_claims_at_reconstructs_past_state(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            first = store.add_claim(subject="car", predicate="mode", value="parked", source="owner:input", observed_at="2026-10-07T09:00:00Z")
            store.supersede(first.claim_id, value="driving", source="owner:input", observed_at="2026-10-07T10:00:00Z")
            past = store.claims_at("2026-10-07T09:30:00Z", subject="car", predicate="mode")
            now = store.claims_at("2026-10-07T10:30:00Z", subject="car", predicate="mode")
            self.assertEqual([item.value for item in past], ["parked"])
            self.assertEqual([item.value for item in now], ["driving"])

    def test_search_can_require_verified_active_claims(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            store.add_claim(subject="jarvis", predicate="capability", value="structured UI observation", source="test:verified", verified=True, confidence=0.9)
            store.add_claim(subject="jarvis", predicate="capability", value="unverified UI rumor", source="test:unverified", verified=False, confidence=1.0)
            result = store.search("UI", verified_only=True)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].value, "structured UI observation")

    def test_double_supersede_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            first = store.add_claim(subject="x", predicate="p", value="one", source="test:source")
            store.supersede(first.claim_id, value="two", source="test:source")
            with self.assertRaises(TemporalMemoryError):
                store.supersede(first.claim_id, value="three", source="test:source")


if __name__ == "__main__":
    unittest.main()
