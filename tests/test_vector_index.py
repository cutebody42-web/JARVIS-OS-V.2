"""Contract tests for the core-owned in-process vector index."""

from pathlib import Path
import tempfile
import unittest

from core.vector_index import VectorIndex, VectorIndexError


class VectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "vectors.db"

    def test_exact_fallback_is_deterministic_and_in_process(self):
        index = VectorIndex(self.path, dimension=3, prefer_native=False)
        self.assertEqual(index.status()["backend"], "python-exact")
        self.assertTrue(index.status()["in_process"])

        index.upsert("far", [1.0, 1.0, 1.0], payload={"kind": "fact"})
        index.upsert("near", [0.1, 0.1, 0.1], payload={"kind": "turn"})
        index.upsert("middle", [0.4, 0.4, 0.4])

        matches = index.search([0.0, 0.0, 0.0], limit=2)
        self.assertEqual([item.entity_id for item in matches], ["near", "middle"])
        self.assertEqual(matches[0].payload, {"kind": "turn"})

    def test_upsert_delete_and_reopen_preserve_data(self):
        index = VectorIndex(self.path, dimension=2, prefer_native=False)
        index.upsert("one", [0.0, 1.0], payload={"v": 1})
        index.upsert("one", [0.0, 0.5], payload={"v": 2})
        self.assertEqual(index.count(), 1)

        reopened = VectorIndex(self.path, dimension=2, prefer_native=False)
        self.assertEqual(reopened.search([0.0, 0.4])[0].payload, {"v": 2})
        self.assertTrue(reopened.delete("one"))
        self.assertFalse(reopened.delete("one"))
        self.assertEqual(reopened.count(), 0)

    def test_dimension_and_vector_validation_fail_closed(self):
        index = VectorIndex(self.path, dimension=2, prefer_native=False)
        invalid = ([1.0], [1.0, float("nan")], [1.0, True], ["1", 2.0])
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    index.upsert("bad", vector)

        with self.assertRaises(VectorIndexError):
            VectorIndex(self.path, dimension=3, prefer_native=False)

    def test_payload_must_be_json_serializable(self):
        index = VectorIndex(self.path, dimension=2, prefer_native=False)
        with self.assertRaises(ValueError):
            index.upsert("bad", [0.0, 0.0], payload={"x": object()})


class SqliteVecProvenanceTests(unittest.TestCase):
    def test_vendored_source_has_pinned_provenance_and_license(self):
        root = Path(__file__).resolve().parents[1] / "core" / "native" / "sqlite_vec"
        upstream = (root / "UPSTREAM.yml").read_text("utf-8")
        self.assertIn("04d28bd21773981e2d266bbf6aa4efbd011eb4f6", upstream)
        self.assertIn("selected_distribution_license: MIT", upstream)
        self.assertTrue((root / "LICENSE-MIT").read_text("utf-8").startswith("MIT License"))
        self.assertTrue((root / "sqlite-vec.c").is_file())
        self.assertTrue((root / "sqlite-vec.h").is_file())


if __name__ == "__main__":
    unittest.main()
