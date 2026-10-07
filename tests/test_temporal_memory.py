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

    def test_semantic_index_is_native_core_with_exact_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            status = store.enable_vector_index(3, prefer_native=False)
            self.assertEqual(status["backend"], "python-exact")
            self.assertTrue(status["in_process"])

            near = store.add_claim(
                subject="jarvis", predicate="memory", value="near", source="test:semantic",
                verified=True,
            )
            far = store.add_claim(
                subject="jarvis", predicate="memory", value="far", source="test:semantic",
                verified=True,
            )
            store.index_claim_embedding(near.claim_id, [0.1, 0.1, 0.1])
            store.index_claim_embedding(far.claim_id, [1.0, 1.0, 1.0])

            matches = store.semantic_search([0.0, 0.0, 0.0], verified_only=True)
            self.assertEqual([item.claim.claim_id for item in matches], [near.claim_id, far.claim_id])
            self.assertLess(matches[0].distance, matches[1].distance)

    def test_semantic_search_filters_superseded_claims_against_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            store.enable_vector_index(2, prefer_native=False)
            old = store.add_claim(
                subject="device", predicate="state", value="old", source="test:semantic",
                verified=True,
            )
            store.index_claim_embedding(old.claim_id, [0.0, 0.0])
            replacement = store.supersede(
                old.claim_id, value="new", source="test:semantic", verified=True,
            )
            store.index_claim_embedding(replacement.claim_id, [0.2, 0.2])

            active = store.semantic_search([0.0, 0.0], active_only=True)
            self.assertEqual([item.claim.claim_id for item in active], [replacement.claim_id])

    def test_double_supersede_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            first = store.add_claim(subject="x", predicate="p", value="one", source="test:source")
            store.supersede(first.claim_id, value="two", source="test:source")
            with self.assertRaises(TemporalMemoryError):
                store.supersede(first.claim_id, value="three", source="test:source")


if __name__ == "__main__":
    unittest.main()
