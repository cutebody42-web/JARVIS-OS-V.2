"""Contract tests for the core-owned in-process vector index."""

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

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
        with self.assertRaises(ValueError):
            index.upsert("bad", [0.0, 0.0], payload=[("x", 1)])
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            index.upsert("bad", [0.0, 0.0], payload={"x": "a" * 65_536})

    def test_environment_cannot_select_an_executable_extension(self):
        injected = Path(self.tmp.name) / ("untrusted.dll" if os.name == "nt" else "untrusted.so")
        injected.write_bytes(b"not-an-extension")
        with patch.dict(os.environ, {"JARVIS_SQLITE_VEC_PATH": str(injected)}):
            index = VectorIndex(self.path, dimension=2, prefer_native=False)
        self.assertNotEqual(index._candidate_extension(), injected)

    def test_tampered_metadata_and_payload_fail_closed(self):
        index = VectorIndex(self.path, dimension=2, prefer_native=False)
        index.upsert("one", [0.0, 1.0], payload={"safe": True})
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "UPDATE jarvis_vector_fallback SET payload_json = '[]' WHERE entity_id = 'one'"
            )
            conn.commit()
        with self.assertRaises(VectorIndexError):
            index.search([0.0, 1.0])

        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "UPDATE jarvis_vector_meta SET value = 'not-a-number' WHERE key = 'dimension'"
            )
            conn.commit()
        with self.assertRaises(VectorIndexError):
            VectorIndex(self.path, dimension=2, prefer_native=False)

    def test_concurrent_upserts_cannot_exceed_entry_quota(self):
        first = VectorIndex(self.path, dimension=2, prefer_native=False)
        second = VectorIndex(self.path, dimension=2, prefer_native=False)
        first_checked = threading.Event()
        second_checked = threading.Event()
        release_first = threading.Event()
        original_check = VectorIndex._ensure_insert_budget

        def coordinated_check(index, conn, **kwargs):
            original_check(index, conn, **kwargs)
            if kwargs["entity_id"] == "first":
                first_checked.set()
                if not release_first.wait(5):
                    raise AssertionError("test writer was not released")
            else:
                second_checked.set()

        with patch("core.vector_index._MAX_ENTRIES", 1), patch.object(
            VectorIndex, "_ensure_insert_budget", coordinated_check
        ), ThreadPoolExecutor(max_workers=2) as executor:
            one = executor.submit(first.upsert, "first", [0.0, 1.0])
            try:
                self.assertTrue(first_checked.wait(5))
                two = executor.submit(second.upsert, "second", [1.0, 0.0])
                # An unprotected quota read reaches this event before the first commit.
                second_checked.wait(0.25)
            finally:
                release_first.set()
            one.result(timeout=5)
            with self.assertRaisesRegex(VectorIndexError, "entry limit"):
                two.result(timeout=5)
        self.assertEqual(first.count(), 1)

    def test_sqlite_page_limit_rolls_back_growth_even_if_estimate_is_small(self):
        index = VectorIndex(self.path, dimension=2, prefer_native=False)
        index.upsert("existing", [0.0, 1.0], payload={"v": "original"})
        with closing(sqlite3.connect(self.path)) as conn:
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        maximum_bytes = self.path.stat().st_size + page_size
        with patch("core.vector_index._MAX_STORAGE_BYTES", maximum_bytes), patch.object(
            index, "_ensure_insert_budget"
        ):
            with self.assertRaisesRegex(VectorIndexError, "storage budget"):
                index.upsert("existing", [1.0, 0.0], payload={"v": "x" * 32_000})
        self.assertLessEqual(self.path.stat().st_size, maximum_bytes)
        self.assertEqual(index.search([0.0, 1.0])[0].payload, {"v": "original"})


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
