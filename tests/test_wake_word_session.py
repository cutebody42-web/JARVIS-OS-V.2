import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.wake_word import WakeWordDetection
from core.wake_word_session import WakeWordSession, WakeWordSessionError


class _FakeDetector:
    SAMPLE_RATE = 16000
    DEFAULT_CHUNK_SAMPLES = 1280

    def __init__(self, _files, *, threshold=0.5):
        self.threshold = threshold
        self.reset_count = 0
        self.score = 0.9

    def process(self, _samples):
        return WakeWordDetection(
            detected=self.score >= self.threshold,
            score=self.score,
            model_name="wakeword",
            threshold=self.threshold,
        )

    def reset(self):
        self.reset_count += 1

    def status(self):
        return {
            "enabled": True,
            "engine": "fake",
            "downloads": False,
            "authority": "activation_only",
        }


def _pcm(samples=1280):
    raw = struct.pack("<" + "h" * samples, *([0] * samples))
    return base64.b64encode(raw).decode("ascii")


class WakeWordSessionTests(unittest.TestCase):
    def _model_dir(self, root: Path):
        models = root / "wake"
        models.mkdir()
        for name in ("wakeword.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
            (models / name).write_bytes(b"onnx")
        return models

    @patch("core.wake_word_session.LocalWakeWordDetector", _FakeDetector)
    def test_configure_enable_and_activation_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = WakeWordSession(root / "settings.json")
            models = self._model_dir(root)
            status = session.configure(str(models), threshold=0.6)
            self.assertTrue(status["configured"])
            self.assertFalse(status["enabled"])
            with self.assertRaises(WakeWordSessionError):
                session.process_base64_pcm(_pcm())
            session.set_enabled(True)
            result = session.process_base64_pcm(_pcm())
            self.assertTrue(result["detected"])
            self.assertTrue(result["activated"])
            self.assertEqual(result["authority"], "activation_only")

    @patch("core.wake_word_session.LocalWakeWordDetector", _FakeDetector)
    def test_cooldown_suppresses_duplicate_activation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = WakeWordSession(root / "settings.json")
            session.configure(str(self._model_dir(root)))
            session.set_enabled(True)
            with patch("core.wake_word_session.time.monotonic", side_effect=[10.0, 10.5, 12.0]):
                first = session.process_base64_pcm(_pcm())
                second = session.process_base64_pcm(_pcm())
                third = session.process_base64_pcm(_pcm())
            self.assertTrue(first["activated"])
            self.assertFalse(second["activated"])
            self.assertTrue(second["detected"])
            self.assertTrue(third["activated"])

    @patch("core.wake_word_session.LocalWakeWordDetector", _FakeDetector)
    def test_restart_requires_fresh_owner_enable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "settings.json"
            first = WakeWordSession(settings)
            first.configure(str(self._model_dir(root)))
            first.set_enabled(True)
            persisted = json.loads(settings.read_text("utf-8"))
            self.assertTrue(persisted["enabled"])

            restarted = WakeWordSession(settings)
            status = restarted.status()
            self.assertTrue(status["configured"])
            self.assertFalse(status["enabled"])
            self.assertEqual(status["last_error"], "owner_reenable_required")

    def test_enable_without_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = WakeWordSession(Path(directory) / "settings.json")
            with self.assertRaises(WakeWordSessionError):
                session.set_enabled(True)

    def test_pcm_base64_validation_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            session = WakeWordSession(Path(directory) / "settings.json")
            for invalid in ("not base64!", base64.b64encode(b"x").decode("ascii"), _pcm(10)):
                with self.assertRaises(ValueError):
                    session.process_base64_pcm(invalid)

    @patch("core.wake_word_session.LocalWakeWordDetector", _FakeDetector)
    def test_model_configuration_must_be_absolute_and_local(self):
        with tempfile.TemporaryDirectory() as directory:
            session = WakeWordSession(Path(directory) / "settings.json")
            with self.assertRaises(WakeWordSessionError):
                session.configure("relative/models")
            with self.assertRaises(WakeWordSessionError):
                session.configure(str(Path(directory) / "missing"))

    @patch("core.wake_word_session.LocalWakeWordDetector", _FakeDetector)
    def test_disable_does_not_execute_any_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = WakeWordSession(root / "settings.json")
            session.configure(str(self._model_dir(root)))
            session.set_enabled(True)
            status = session.set_enabled(False)
            self.assertFalse(status["enabled"])
            self.assertEqual(status["authority"], "activation_only")
            self.assertTrue(status["local_only"])
            self.assertFalse(status["microphone_owner"])


if __name__ == "__main__":
    unittest.main()
