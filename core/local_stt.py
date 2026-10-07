"""Local speech-to-text backend for JARVIS.

This module integrates the MIT-licensed faster-whisper runtime without automatic
model downloads. The owner supplies an existing local model directory; JARVIS
loads it lazily and keeps transcription local to the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


class LocalSTTError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalTranscript:
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    engine: str = "faster-whisper"


class FasterWhisperRuntime:
    """Lazy local-only wrapper around faster-whisper.

    ``model_path`` must already exist on disk. This intentionally prevents an
    assistant request from triggering a model download or selecting a remote
    model identifier behind the owner's back.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        compute_type: str = "default",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists() or not path.is_dir():
            raise LocalSTTError("Local speech model directory does not exist.")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Unsupported faster-whisper device.")
        if not isinstance(compute_type, str) or not compute_type or len(compute_type) > 40:
            raise ValueError("Invalid faster-whisper compute type.")
        self.model_path = path.resolve()
        self.device = device
        self.compute_type = compute_type
        self._model_factory = model_factory
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise LocalSTTError(
                    "faster-whisper is not installed; install the local voice extra first."
                ) from exc
            factory = WhisperModel
        try:
            self._model = factory(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=True,
            )
        except TypeError:
            # Test doubles and older compatible factories may not expose the
            # local_files_only keyword. The path validation above still keeps
            # the selected model local.
            self._model = factory(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise LocalSTTError(f"Local speech model failed to load ({type(exc).__name__}).") from None
        return self._model

    @staticmethod
    def _bounded_probability(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0.0, min(1.0, float(value)))

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> LocalTranscript:
        audio_path = Path(audio).expanduser()
        if not audio_path.exists() or not audio_path.is_file():
            raise LocalSTTError("Audio file does not exist.")
        if language is not None:
            if not isinstance(language, str) or not 1 <= len(language.strip()) <= 16:
                raise ValueError("Invalid language hint.")
            language = language.strip()
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or not 1 <= beam_size <= 10:
            raise ValueError("beam_size must be between 1 and 10.")

        model = self._load()
        try:
            segments, info = model.transcribe(
                str(audio_path.resolve()),
                language=language,
                beam_size=beam_size,
                vad_filter=bool(vad_filter),
            )
            pieces: list[str] = []
            for segment in segments:
                text = getattr(segment, "text", "")
                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())
                if sum(len(part) for part in pieces) > 200000:
                    raise LocalSTTError("Transcript exceeds the safety size limit.")
        except LocalSTTError:
            raise
        except Exception as exc:
            raise LocalSTTError(f"Local transcription failed ({type(exc).__name__}).") from None

        detected = getattr(info, "language", None)
        if not isinstance(detected, str) or not detected:
            detected = language
        duration = getattr(info, "duration", None)
        duration_value = (
            float(duration)
            if not isinstance(duration, bool) and isinstance(duration, (int, float)) and duration >= 0
            else None
        )
        probability = self._bounded_probability(getattr(info, "language_probability", None))
        return LocalTranscript(
            text=" ".join(pieces).strip(),
            language=detected,
            language_probability=probability,
            duration_seconds=duration_value,
        )
