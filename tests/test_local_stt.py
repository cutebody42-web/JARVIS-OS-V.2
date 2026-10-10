from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core import local_stt
from core.local_stt import FasterWhisperRuntime, LocalSTTError


class _Segment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, segments=None, info=None):
        self.calls = []
        self.segments = segments or [_Segment(" hello "), _Segment("world")]
        self.info = info or SimpleNamespace(
            language="en", language_probability=1.2, duration=2.5,
        )

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter(self.segments), self.info


def _make_model(root: Path, name: str = "model") -> Path:
    model = root / name
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.bin").write_bytes(b"model")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "vocabulary.txt").write_text("token", encoding="utf-8")
    return model


class LocalSTTTests(unittest.TestCase):
    def test_runtime_requires_existing_local_model_directory(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LocalSTTError):
                FasterWhisperRuntime(Path(td) / "missing")

    def test_runtime_requires_a_complete_nonempty_local_model_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for missing in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
                with self.subTest(missing=missing):
                    model_dir = _make_model(root, f"model-{missing.replace('.', '-')}")
                    (model_dir / missing).unlink()
                    with self.assertRaises(LocalSTTError):
                        FasterWhisperRuntime(model_dir)

            model_dir = _make_model(root, "empty-tokenizer")
            (model_dir / "tokenizer.json").write_bytes(b"")
            with self.assertRaises(LocalSTTError):
                FasterWhisperRuntime(model_dir)

    def test_runtime_is_lazy_and_transcribes_local_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
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
            model_dir = _make_model(root)
            audio = root / "sample.wav"
            audio.write_bytes(b"x")
            runtime = FasterWhisperRuntime(model_dir, model_factory=lambda *a, **k: _FakeModel())
            with self.assertRaises(ValueError):
                runtime.transcribe(audio, beam_size=0)
            with self.assertRaises(ValueError):
                runtime.transcribe(audio, language="x" * 17)
            with self.assertRaises(ValueError):
                runtime.transcribe(audio, vad_filter="yes")
            with self.assertRaises(LocalSTTError):
                runtime.transcribe(root / "missing.wav")

    def test_runtime_rejects_unc_and_reparse_paths(self):
        for path in (
            r"\\server\share\model",
            "//server/share/model",
            r"\\?\UNC\server\share\model",
        ):
            with self.subTest(path=path), self.assertRaises(LocalSTTError):
                FasterWhisperRuntime(path)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            with mock.patch(
                "core.local_stt._is_reparse_point",
                side_effect=lambda path: Path(path).name == model_dir.name,
            ):
                with self.assertRaises(LocalSTTError):
                    FasterWhisperRuntime(model_dir)

            audio = root / "sample.wav"
            audio.write_bytes(b"audio")
            runtime = FasterWhisperRuntime(
                model_dir, model_factory=lambda *a, **k: _FakeModel(),
            )
            with mock.patch(
                "core.local_stt._is_reparse_point",
                side_effect=lambda path: Path(path).name == audio.name,
            ):
                with self.assertRaises(LocalSTTError):
                    runtime.transcribe(audio)

    def test_runtime_rejects_mapped_network_drives_for_models_and_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")

            with mock.patch("core.local_stt._windows_drive_type", return_value=4):
                with self.assertRaisesRegex(LocalSTTError, "local fixed drive"):
                    FasterWhisperRuntime(model_dir)

            runtime = FasterWhisperRuntime(
                model_dir, model_factory=lambda *a, **k: _FakeModel()
            )
            with mock.patch("core.local_stt._windows_drive_type", return_value=4):
                with self.assertRaisesRegex(LocalSTTError, "local fixed drive"):
                    runtime.transcribe(audio)

    def test_runtime_rejects_symlinked_model_and_audio_when_supported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            model_link = root / "linked-model"
            try:
                model_link.symlink_to(model_dir, target_is_directory=True)
            except OSError:
                model_link = None
            if model_link is not None:
                with self.assertRaises(LocalSTTError):
                    FasterWhisperRuntime(model_link)

            outside_config = root / "outside-preprocessor.json"
            outside_config.write_text("{}", encoding="utf-8")
            optional_link = model_dir / "preprocessor_config.json"
            try:
                optional_link.symlink_to(outside_config)
            except OSError:
                optional_link = None
            if optional_link is not None:
                with self.assertRaises(LocalSTTError):
                    FasterWhisperRuntime(model_dir)
                optional_link.unlink()

            audio = root / "sample.wav"
            audio.write_bytes(b"audio")
            audio_link = root / "linked.wav"
            try:
                audio_link.symlink_to(audio)
            except OSError:
                audio_link = None
            if audio_link is not None:
                runtime = FasterWhisperRuntime(model_dir, model_factory=lambda *a, **k: _FakeModel())
                with self.assertRaises(LocalSTTError):
                    runtime.transcribe(audio_link)

    def test_model_load_is_serialized_and_happens_once(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _make_model(Path(td))
            fake = _FakeModel()
            factory_calls = 0
            calls_lock = threading.Lock()

            def factory(*args, **kwargs):
                nonlocal factory_calls
                with calls_lock:
                    factory_calls += 1
                time.sleep(0.02)
                return fake

            runtime = FasterWhisperRuntime(model_dir, model_factory=factory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                models = list(pool.map(lambda _: runtime._load(), range(16)))
            self.assertEqual(factory_calls, 1)
            self.assertTrue(all(model is fake for model in models))

    def test_type_error_from_factory_is_not_retried_without_local_only(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _make_model(Path(td))
            calls = []

            def factory(*args, **kwargs):
                calls.append((args, kwargs))
                raise TypeError("unsupported keyword")

            runtime = FasterWhisperRuntime(model_dir, model_factory=factory)
            with self.assertRaises(LocalSTTError) as caught:
                runtime._load()
            self.assertIn("TypeError", str(caught.exception))
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][1]["local_files_only"], True)

    def test_oversized_audio_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "oversized.wav"
            with audio.open("wb") as stream:
                stream.truncate(local_stt._MAX_AUDIO_BYTES + 1)
            factory_calls = []
            runtime = FasterWhisperRuntime(
                model_dir,
                model_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
            )
            with self.assertRaises(LocalSTTError):
                runtime.transcribe(audio)
            self.assertEqual(factory_calls, [])

    def test_audio_growth_during_model_load_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "growing.wav"
            audio.write_bytes(b"audio")
            fake = _FakeModel()

            def factory(*args, **kwargs):
                with audio.open("r+b") as stream:
                    stream.truncate(local_stt._MAX_AUDIO_BYTES + 1)
                return fake

            runtime = FasterWhisperRuntime(model_dir, model_factory=factory)
            with self.assertRaises(LocalSTTError):
                runtime.transcribe(audio)
            self.assertEqual(fake.calls, [])

    def test_audio_duration_is_bounded_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "long.wav"
            audio.write_bytes(b"audio")
            factory_calls = []
            runtime = FasterWhisperRuntime(
                model_dir,
                model_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs)),
                audio_duration_probe=lambda _: local_stt._MAX_AUDIO_SECONDS + 1,
            )
            with self.assertRaisesRegex(LocalSTTError, "duration"):
                runtime.transcribe(audio)
            self.assertEqual(factory_calls, [])

    def test_segment_iterator_is_bounded_even_for_empty_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")

            class EndlessModel(_FakeModel):
                def transcribe(self, path, **kwargs):
                    def segments():
                        while True:
                            yield _Segment(" ")
                    return segments(), self.info

            runtime = FasterWhisperRuntime(
                model_dir, model_factory=lambda *args, **kwargs: EndlessModel()
            )
            with mock.patch("core.local_stt._MAX_SEGMENTS", 3):
                with self.assertRaisesRegex(LocalSTTError, "segment"):
                    runtime.transcribe(audio)

    def test_transcript_limit_counts_join_separators_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")

            exact = _FakeModel([
                _Segment("a" * 100_000),
                _Segment("b" * 99_999),
            ])
            result = FasterWhisperRuntime(
                model_dir, model_factory=lambda *a, **k: exact,
            ).transcribe(audio)
            self.assertEqual(len(result.text), local_stt._MAX_TRANSCRIPT_CHARS)

            oversized = _FakeModel([
                _Segment("a" * 100_000),
                _Segment("b" * 100_000),
            ])
            runtime = FasterWhisperRuntime(
                model_dir, model_factory=lambda *a, **k: oversized,
            )
            with self.assertRaises(LocalSTTError):
                runtime.transcribe(audio)

    def test_nonfinite_probability_and_duration_are_discarded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = _make_model(root)
            audio = root / "sample.wav"
            audio.write_bytes(b"audio")
            for probability, duration in (
                (float("nan"), float("inf")),
                (float("inf"), float("-inf")),
                (float("-inf"), float("nan")),
            ):
                with self.subTest(probability=probability, duration=duration):
                    fake = _FakeModel(info=SimpleNamespace(
                        language="en",
                        language_probability=probability,
                        duration=duration,
                    ))
                    result = FasterWhisperRuntime(
                        model_dir, model_factory=lambda *a, **k: fake,
                    ).transcribe(audio)
                    self.assertIsNone(result.language_probability)
                    self.assertIsNone(result.duration_seconds)


if __name__ == "__main__":
    unittest.main()
