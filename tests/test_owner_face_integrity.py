"""Offline integrity contracts for JARVIS owner-face model provisioning."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.owner_face import FaceIdentityError, OwnerFaceRecognizer


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024 * 1024):
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        yield self.payload[midpoint:]


class OwnerFaceIntegrityTests(unittest.TestCase):
    def test_verified_download_commits_only_after_hash_match(self):
        payload = b"pinned-opencv-model-fixture"
        expected = sha256(payload).hexdigest()
        with TemporaryDirectory() as directory:
            target = Path(directory) / "fixture.onnx"
            recognizer = OwnerFaceRecognizer(
                model_dir=directory,
                http_get=lambda *args, **kwargs: FakeResponse(payload),
            )
            result = recognizer._download_verified(
                "https://example.invalid/fixture.onnx", target, expected
            )
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(target.with_suffix(".onnx.partial").exists())

    def test_hash_mismatch_fails_closed_and_removes_partial(self):
        payload = b"tampered-model"
        with TemporaryDirectory() as directory:
            target = Path(directory) / "fixture.onnx"
            recognizer = OwnerFaceRecognizer(
                model_dir=directory,
                http_get=lambda *args, **kwargs: FakeResponse(payload),
            )
            with self.assertRaises(FaceIdentityError):
                recognizer._download_verified(
                    "https://example.invalid/fixture.onnx", target, "0" * 64
                )
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".onnx.partial").exists())


if __name__ == "__main__":
    unittest.main()
