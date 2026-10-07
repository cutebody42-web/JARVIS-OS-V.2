import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.local_stt import FasterWhisperRuntime, LocalSTTError


class _Segment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter([_Segment(" hello "), _Segment("world")]), SimpleNamespace(
            language="en", language_probability=1.2, duration=2.5,
        )


class LocalSTTTests(unittest.TestCase):
    def test_runtime_requires_existing_local_model_directory(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LocalSTTError):
                FasterWhisperRuntime(Path(td) / "missing")

    def test_runtime_is_lazy_and_transcribes_local_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF-test")
            fake = _FakeModel()
            factory_calls = []

            def factory(path, **kwargs):
                factory_calls.append((path, kwargs))
                return fake

            runtime = FasterWhisperRuntime(model_dir, device="cpu", compute_type="int8", model_factory=factory)
            self.assertFalse(runtime.loaded)
            result = runtime.transcribe(audio, language="en", beam_size=3)
            self.assertTrue(runtime.loaded)
            self.assertEqual(result.text, "hello world")
            self.assertEqual(result.language, "en")
            self.assertEqual(result.language_probability, 1.0)
            self.assertEqual(result.duration_seconds, 2.5)
            self.assertEqual(result.engine, "faster-whisper")
            self.assertTrue(factory_calls[0][1]["local_files_only"])
            self.assertEqual(fake.calls[0][1], {"language": "en", "beam_size": 3, "vad_filter": True})

    def test_runtime_rejects_invalid_controls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            audio = root / "sample.wav"
            audio.write_bytes(b"x")
            runtime = FasterWhisperRuntime(model_dir, model_factory=lambda *a, **k: _FakeModel())
            with self.assertRaises(ValueError):
                runtime.transcribe(audio, beam_size=0)
            with self.assertRaises(ValueError):
                runtime.transcribe(audio, language="x" * 17)
            with self.assertRaises(LocalSTTError):
                runtime.transcribe(root / "missing.wav")


if __name__ == "__main__":
    unittest.main()
