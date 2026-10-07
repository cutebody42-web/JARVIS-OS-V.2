import tempfile
import unittest
from pathlib import Path

from core.hybrid_retrieval import TemporalHybridRetriever
from core.temporal_memory import TemporalMemoryStore


class HybridRetrievalTests(unittest.TestCase):
    def test_lexical_search_ranks_exact_terms_and_filters_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            exact = store.add_claim(
                subject="jarvis",
                predicate="runtime",
                value="local model storage is owner controlled",
                source="test:verified",
                verified=True,
                confidence=0.9,
            )
            store.add_claim(
                subject="battery",
                predicate="state",
                value="voltage nominal",
                source="test:verified",
                verified=True,
            )
            store.add_claim(
                subject="jarvis",
                predicate="runtime",
                value="local model rumor",
                source="test:unverified",
                verified=False,
                confidence=1.0,
            )

            retriever = TemporalHybridRetriever(store)
            result = retriever.lexical_search("local model storage", verified_only=True)

            self.assertGreaterEqual(len(result), 1)
            self.assertEqual(result[0].claim.claim_id, exact.claim_id)
            self.assertTrue(all(item.claim.verified for item in result))
            self.assertGreater(result[0].score, 0.0)

    def test_lexical_search_is_unicode_aware(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            arabic = store.add_claim(
                subject="المشروع",
                predicate="الحالة",
                value="الذاكرة المحلية سريعة",
                source="test:arabic",
                verified=True,
            )
            store.add_claim(
                subject="device",
                predicate="battery",
                value="charged",
                source="test:other",
                verified=True,
            )

            result = TemporalHybridRetriever(store).lexical_search("الذاكرة المحلية")
            self.assertEqual(result[0].claim.claim_id, arabic.claim_id)

    def test_punctuation_only_claim_does_not_create_zero_length_corpus_failure(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            store.add_claim(
                subject="---",
                predicate="...",
                value="!!!",
                source="test:punctuation",
                verified=True,
            )
            result = TemporalHybridRetriever(store).lexical_search("normal query")
            self.assertEqual(result, ())

    def test_hybrid_search_fuses_vector_and_lexical_rank_without_mixing_raw_scales(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            store.enable_vector_index(2, prefer_native=False)
            lexical = store.add_claim(
                subject="mission",
                predicate="codename",
                value="saturn launch checklist",
                source="test:hybrid",
                verified=True,
            )
            semantic = store.add_claim(
                subject="mission",
                predicate="notes",
                value="deployment preparation",
                source="test:hybrid",
                verified=True,
            )
            store.index_claim_embedding(lexical.claim_id, [1.0, 1.0])
            store.index_claim_embedding(semantic.claim_id, [0.0, 0.0])

            retriever = TemporalHybridRetriever(store)
            result = retriever.hybrid_search(
                "saturn launch",
                [0.0, 0.0],
                lexical_weight=0.7,
                semantic_weight=0.3,
            )

            ids = [item.claim.claim_id for item in result]
            self.assertIn(lexical.claim_id, ids)
            self.assertIn(semantic.claim_id, ids)
            self.assertEqual(result[0].claim.claim_id, lexical.claim_id)
            lexical_match = next(item for item in result if item.claim.claim_id == lexical.claim_id)
            semantic_match = next(item for item in result if item.claim.claim_id == semantic.claim_id)
            self.assertIsNotNone(lexical_match.lexical_score)
            self.assertIsNotNone(lexical_match.semantic_distance)
            self.assertIsNone(semantic_match.lexical_score)
            self.assertIsNotNone(semantic_match.semantic_distance)

    def test_invalid_fusion_controls_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.db")
            store.enable_vector_index(2, prefer_native=False)
            retriever = TemporalHybridRetriever(store)
            with self.assertRaises(ValueError):
                retriever.hybrid_search("query", [0.0, 0.0], lexical_weight=0, semantic_weight=0)
            with self.assertRaises(ValueError):
                retriever.hybrid_search("query", [0.0, 0.0], lexical_weight=float("nan"))
            with self.assertRaises(ValueError):
                retriever.lexical_search("query", limit=0)
            with self.assertRaises(ValueError):
                TemporalHybridRetriever(store, lexical_candidate_limit="100")


if __name__ == "__main__":
    unittest.main()
