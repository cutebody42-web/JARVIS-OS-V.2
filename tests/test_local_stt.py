from types import SimpleNamespace

import pytest

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
            language="en",
            language_probability=1.2,
            duration=2.5,
        )


def test_runtime_requires_existing_local_model_directory(tmp_path):
    with pytest.raises(LocalSTTError):
        FasterWhisperRuntime(tmp_path / "missing")


def test_runtime_is_lazy_and_transcribes_local_audio(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test")
    fake = _FakeModel()
    factory_calls = []

    def factory(path, **kwargs):
        factory_calls.append((path, kwargs))
        return fake

    runtime = FasterWhisperRuntime(model_dir, device="cpu", compute_type="int8", model_factory=factory)
    assert runtime.loaded is False

    result = runtime.transcribe(audio, language="en", beam_size=3)

    assert runtime.loaded is True
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.language_probability == 1.0
    assert result.duration_seconds == 2.5
    assert result.engine == "faster-whisper"
    assert factory_calls[0][1]["local_files_only"] is True
    assert fake.calls[0][1] == {"language": "en", "beam_size": 3, "vad_filter": True}


def test_runtime_rejects_invalid_controls(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"x")
    runtime = FasterWhisperRuntime(model_dir, model_factory=lambda *a, **k: _FakeModel())

    with pytest.raises(ValueError):
        runtime.transcribe(audio, beam_size=0)
    with pytest.raises(ValueError):
        runtime.transcribe(audio, language="x" * 17)
    with pytest.raises(LocalSTTError):
        runtime.transcribe(tmp_path / "missing.wav")
