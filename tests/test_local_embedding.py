import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from core.local_embedding import LocalEmbeddingError, LocalOnnxEmbeddingProvider


class _Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.type_ids = [0] * len(ids)


class _Tokenizer:
    def encode(self, text):
        values = [max(1, min(50, len(piece))) for piece in text.split()]
        return _Encoding(values or [1])


class _Session:
    def __init__(self, *, dimension=3, bad_input=False):
        self.dimension = dimension
        self.bad_input = bad_input
        self.calls = []

    def get_inputs(self):
        names = ["input_ids", "attention_mask", "mystery"] if self.bad_input else ["input_ids", "attention_mask"]
        return [SimpleNamespace(name=name) for name in names]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, requested_outputs, payload):
        self.calls.append((requested_outputs, payload))
        ids = payload["input_ids"].astype(np.float32)
        batch, width = ids.shape
        output = np.zeros((batch, width, self.dimension), dtype=np.float32)
        for column in range(self.dimension):
            output[:, :, column] = ids + float(column + 1)
        return [output]


def _make_model(root: Path, **overrides) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "model.onnx").write_bytes(b"fake-onnx")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_id": "owner/test-embed",
        "dimension": 3,
        "model_file": "model.onnx",
        "tokenizer_file": "tokenizer.json",
        "pooling": "mean",
        "normalize": True,
        "max_length": 16,
        "pad_token_id": 0,
    }
    manifest.update(overrides)
    (model / "jarvis-embedding.json").write_text(json.dumps(manifest), encoding="utf-8")
    return model


class LocalEmbeddingTests(unittest.TestCase):
    def test_runtime_requires_existing_local_model_directory(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LocalEmbeddingError):
                LocalOnnxEmbeddingProvider(Path(td) / "missing")
        with self.assertRaisesRegex(LocalEmbeddingError, "absolute"):
            LocalOnnxEmbeddingProvider(Path("relative-model"))

    def test_manifest_confines_model_files_to_owner_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside.onnx"
            outside.write_bytes(b"x")
            model = _make_model(root, model_file="../outside.onnx")
            with self.assertRaises(LocalEmbeddingError):
                LocalOnnxEmbeddingProvider(model)

    def test_manifest_and_model_files_are_bounded_regular_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = _make_model(root)
            (model / "jarvis-embedding.json").write_text(" " * 65_537, encoding="utf-8")
            with self.assertRaisesRegex(LocalEmbeddingError, "64 KiB"):
                LocalOnnxEmbeddingProvider(model)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = _make_model(root)
            target = root / "outside-tokenizer.json"
            target.write_text("{}", encoding="utf-8")
            tokenizer = model / "tokenizer.json"
            tokenizer.unlink()
            try:
                tokenizer.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with self.assertRaisesRegex(LocalEmbeddingError, "symbolic link|reparse point"):
                LocalOnnxEmbeddingProvider(model)

    def test_runtime_rejects_models_on_mapped_network_drives(self):
        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td))
            with mock.patch("core.local_embedding._windows_drive_type", return_value=4):
                with self.assertRaisesRegex(LocalEmbeddingError, "local fixed drive"):
                    LocalOnnxEmbeddingProvider(model)

    def test_model_asset_change_before_lazy_load_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td))
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: _Session(),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            (model / "model.onnx").write_bytes(b"changed-model")
            with self.assertRaisesRegex(LocalEmbeddingError, "changed"):
                provider.embed("owner memory")

    def test_runtime_is_lazy_local_and_normalizes_embeddings(self):
        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td))
            session = _Session()
            session_paths = []
            tokenizer_paths = []

            def session_factory(path):
                session_paths.append(path)
                return session

            def tokenizer_factory(path):
                tokenizer_paths.append(path)
                return _Tokenizer()

            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=session_factory,
                tokenizer_factory=tokenizer_factory,
            )
            self.assertFalse(provider.loaded)
            vector = provider.embed("local memory embedding")
            self.assertTrue(provider.loaded)
            self.assertEqual(len(vector), 3)
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=6)
            self.assertEqual(session_paths, [(model / "model.onnx").resolve()])
            self.assertEqual(tokenizer_paths, [(model / "tokenizer.json").resolve()])
            self.assertEqual(session.calls[0][0], ["last_hidden_state"])
            self.assertEqual(provider.info.model_id, "owner/test-embed")
            self.assertEqual(provider.info.dimension, 3)
            self.assertTrue(provider.info.local_only)

    def test_runtime_rejects_unknown_model_inputs_and_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = _make_model(root)
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: _Session(bad_input=True),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            with self.assertRaises(LocalEmbeddingError):
                provider.embed("hello")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = _make_model(root)
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: _Session(dimension=4),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            with self.assertRaises(LocalEmbeddingError):
                provider.embed("hello")

    def test_runtime_bounds_input_batch_and_memory(self):
        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td))
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: _Session(),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            with self.assertRaises(ValueError):
                provider.embed(" ")
            with self.assertRaises(ValueError):
                provider.embed_texts([])
            with self.assertRaises(ValueError):
                provider.embed_texts(["x"] * 33)

        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td), dimension=8192, max_length=8192)
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: _Session(dimension=8192),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            with self.assertRaises(ValueError):
                provider.embed("bounded local inference")
            self.assertFalse(provider.loaded)

    def test_runtime_rejects_mismatched_or_oversized_output(self):
        class WrongBatchSession(_Session):
            def run(self, requested_outputs, payload):
                return [np.zeros((2, 1, self.dimension), dtype=np.float32)]

        with tempfile.TemporaryDirectory() as td:
            model = _make_model(Path(td))
            provider = LocalOnnxEmbeddingProvider(
                model,
                session_factory=lambda _: WrongBatchSession(),
                tokenizer_factory=lambda _: _Tokenizer(),
            )
            with self.assertRaisesRegex(LocalEmbeddingError, "token shape"):
                provider.embed("bounded output")


if __name__ == "__main__":
    unittest.main()
