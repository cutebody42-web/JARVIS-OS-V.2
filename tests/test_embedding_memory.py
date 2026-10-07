import tempfile
import unittest
from pathlib import Path

from core.embedding_memory import TemporalEmbeddingMemory
from core.local_embedding import EmbeddingInfo
from core.temporal_memory import TemporalMemoryStore


class FakeProvider:
    info = EmbeddingInfo(model_id="test-local", dimension=2, engine="test", normalized=False)

    def embed(self, text):
        lowered = text.casefold()
        return (1.0, 0.0) if "physics" in lowered else (0.0, 1.0)

    def embed_texts(self, texts):
        return tuple(self.embed(text) for text in texts)


class EmbeddingMemoryTests(unittest.TestCase):
    def test_indexes_claims_and_searches_from_text(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.sqlite")
            first = store.add_claim(subject="Physics", predicate="topic", value="circuits", source="notes")
            second = store.add_claim(subject="Chemistry", predicate="topic", value="alkenes", source="notes")
            memory = TemporalEmbeddingMemory(store, FakeProvider(), prefer_native=False)
            self.assertEqual(memory.index_claims([first.claim_id, second.claim_id]), 2)
            matches = memory.semantic_search_text("physics revision")
            self.assertEqual(matches[0].claim.claim_id, first.claim_id)
            self.assertFalse(memory.status()["network_required"])
            self.assertEqual(memory.status()["dimension"], 2)

    def test_hybrid_search_uses_local_embedding(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.sqlite")
            first = store.add_claim(subject="Physics", predicate="contains", value="voltage current", source="notes")
            second = store.add_claim(subject="Chemistry", predicate="contains", value="alkene formula", source="notes")
            memory = TemporalEmbeddingMemory(store, FakeProvider(), prefer_native=False)
            memory.index_active_claims()
            matches = memory.hybrid_search_text("physics voltage", limit=2)
            self.assertEqual(matches[0].claim.claim_id, first.claim_id)
            self.assertEqual({item.claim.claim_id for item in matches}, {first.claim_id, second.claim_id})

    def test_batch_controls_are_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            store = TemporalMemoryStore(Path(td) / "memory.sqlite")
            memory = TemporalEmbeddingMemory(store, FakeProvider(), prefer_native=False)
            with self.assertRaises(ValueError):
                memory.index_claims([], batch_size=0)
            with self.assertRaises(ValueError):
                memory.index_claims(["x"] * 1001)


if __name__ == "__main__":
    unittest.main()
