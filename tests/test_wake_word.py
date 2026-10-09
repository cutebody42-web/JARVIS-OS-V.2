import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.wake_word import (
    LocalWakeWordDetector,
    WakeWordDetection,
    WakeWordError,
    WakeWordModelFiles,
)


class _FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.frames = []
        self.reset_count = 0

    def predict(self, frame):
        self.frames.append(frame)
        return self.prediction

    def reset(self):
        self.reset_count += 1


class WakeWordTests(unittest.TestCase):
    def _model_dir(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        for name in ("wakeword.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
            (root / name).write_bytes(b"onnx")
        return temp, root

    def test_model_directory_requires_all_local_onnx_files(self):
        temp, root = self._model_dir()
        try:
            files = WakeWordModelFiles.from_directory(root)
            self.assertTrue(files.wakeword_model.is_absolute())
            (root / "embedding_model.onnx").unlink()
            with self.assertRaises(WakeWordError):
                WakeWordModelFiles.from_directory(root)
        finally:
            temp.cleanup()

    def test_detector_is_lazy_and_uses_explicit_local_paths(self):
        temp, root = self._model_dir()
        calls = []
        fake = _FakeModel({"wakeword": 0.8})

        def factory(**kwargs):
            calls.append(kwargs)
            return fake

        try:
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=factory,
                threshold=0.6,
            )
            self.assertFalse(detector.status()["loaded"])
            result = detector.process([0] * 1280)
            self.assertTrue(result.detected)
            self.assertEqual(result.engine, "openwakeword-onnx")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["inference_framework"], "onnx")
            self.assertEqual(calls[0]["wakeword_models"], [str((root / "wakeword.onnx").resolve())])
            self.assertEqual(calls[0]["melspec_model_path"], str((root / "melspectrogram.onnx").resolve()))
            self.assertEqual(calls[0]["embedding_model_path"], str((root / "embedding_model.onnx").resolve()))
            detector.process([0] * 1280)
            self.assertEqual(len(calls), 1)
        finally:
            temp.cleanup()

    def test_threshold_does_not_grant_authority(self):
        temp, root = self._model_dir()
        try:
            fake = _FakeModel({"wakeword": 0.49})
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: fake,
                threshold=0.5,
            )
            result = detector.process([1] * 1280)
            self.assertEqual(
                result,
                WakeWordDetection(False, 0.49, "wakeword", 0.5),
            )
            self.assertEqual(detector.status()["authority"], "activation_only")
            self.assertFalse(detector.status()["downloads"])
        finally:
            temp.cleanup()

    def test_single_unknown_runtime_key_is_accepted(self):
        temp, root = self._model_dir()
        try:
            fake = _FakeModel({"generated_model_name": 0.91})
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: fake,
                model_name="owner-jarvis",
            )
            self.assertTrue(detector.process([0] * 1280).detected)
        finally:
            temp.cleanup()

    def test_nested_multiclass_score_uses_best_numeric_value(self):
        temp, root = self._model_dir()
        try:
            fake = _FakeModel({"wakeword": {"negative": 0.1, "jarvis": 0.73}})
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: fake,
                threshold=0.7,
            )
            result = detector.process([0] * 1280)
            self.assertTrue(result.detected)
            self.assertAlmostEqual(result.score, 0.73)
        finally:
            temp.cleanup()

    def test_input_bounds_and_pcm_contract(self):
        temp, root = self._model_dir()
        try:
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: _FakeModel({"wakeword": 0.5}),
            )
            for invalid in ([0] * 10, [0] * 80001, [40000] * 1280, [1.2] * 1280):
                with self.assertRaises(ValueError):
                    detector.process(invalid)
        finally:
            temp.cleanup()

    def test_invalid_runtime_output_fails_closed(self):
        temp, root = self._model_dir()
        try:
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: _FakeModel({"wakeword": float("nan")}),
            )
            with self.assertRaises(WakeWordError):
                detector.process([0] * 1280)
        finally:
            temp.cleanup()

    def test_reset_delegates_without_reloading(self):
        temp, root = self._model_dir()
        try:
            fake = _FakeModel({"wakeword": 0.8})
            detector = LocalWakeWordDetector(
                WakeWordModelFiles.from_directory(root),
                model_factory=lambda **_: fake,
            )
            detector.process([0] * 1280)
            detector.reset()
            self.assertEqual(fake.reset_count, 1)
        finally:
            temp.cleanup()

    def test_missing_optional_runtime_has_clear_error(self):
        temp, root = self._model_dir()
        try:
            detector = LocalWakeWordDetector(WakeWordModelFiles.from_directory(root))
            real_import = __import__("importlib").import_module

            def import_module(name):
                if name == "openwakeword.model":
                    raise ImportError("missing")
                return real_import(name)

            with patch("core.wake_word.importlib.import_module", side_effect=import_module):
                with self.assertRaisesRegex(WakeWordError, "optional wake-word requirements"):
                    detector.process([0] * 1280)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
