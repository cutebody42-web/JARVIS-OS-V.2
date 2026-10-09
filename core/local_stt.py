"""Local speech-to-text backend for JARVIS.

This module integrates the MIT-licensed faster-whisper runtime without automatic
model downloads. The owner supplies an existing local model directory; JARVIS
loads it lazily and keeps transcription local to the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Callable


_REQUIRED_MODEL_FILES = ("config.json", "model.bin", "tokenizer.json")
_VOCABULARY_FILES = ("vocabulary.json", "vocabulary.txt")
_OPTIONAL_MODEL_FILES = ("preprocessor_config.json",)
_MAX_AUDIO_BYTES = 256 * 1024 * 1024
_MAX_AUDIO_SECONDS = 4 * 60 * 60
_MAX_TRANSCRIPT_CHARS = 200_000
_MAX_SEGMENTS = 100_000
_LANGUAGE = re.compile(r"[A-Za-z0-9-]{1,16}")


class LocalSTTError(RuntimeError):
    pass


def _is_unc_or_device_path(value: str | Path) -> bool:
    """Reject network and device namespaces before pathlib can normalize them."""
    try:
        raw = os.fspath(value)
    except TypeError:
        return False
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    return raw.replace("/", "\\").startswith("\\\\")


def _is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _windows_drive_type(path: Path) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(str(Path(path.anchor))))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise LocalSTTError("Could not verify an owner-local fixed drive.") from exc


def _reject_non_local_windows_drive(path: Path, label: str) -> None:
    drive_type = _windows_drive_type(path)
    if drive_type is not None and drive_type != 3:
        raise LocalSTTError(f"{label} must use an owner-local fixed drive.")


def _reject_link_components(path: Path, label: str) -> None:
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        current /= part
        try:
            unsafe = _is_reparse_point(current)
        except OSError:
            raise LocalSTTError(f"{label} could not be validated as owner-local.") from None
        if unsafe:
            raise LocalSTTError(f"{label} must not traverse a symlink or reparse point.")


def _validated_local_path(
    value: str | Path,
    *,
    label: str,
    directory: bool,
    nonempty: bool = False,
) -> Path:
    if _is_unc_or_device_path(value):
        raise LocalSTTError(f"{label} must be on an owner-local filesystem.")
    try:
        path = Path(value).expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError):
        raise LocalSTTError(f"Invalid {label.lower()} path.") from None
    if _is_unc_or_device_path(path):
        raise LocalSTTError(f"{label} must be on an owner-local filesystem.")
    _reject_non_local_windows_drive(path, label)
    _reject_link_components(path, label)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise LocalSTTError(f"{label} does not exist.") from None
    except OSError:
        raise LocalSTTError(f"{label} could not be validated as owner-local.") from None
    expected_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected_type:
        kind = "directory" if directory else "regular file"
        raise LocalSTTError(f"{label} must be a {kind}.")
    if nonempty and info.st_size <= 0:
        raise LocalSTTError(f"{label} must not be empty.")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise LocalSTTError(f"{label} could not be resolved safely.") from None
    if _is_unc_or_device_path(resolved):
        raise LocalSTTError(f"{label} must be on an owner-local filesystem.")
    _reject_non_local_windows_drive(resolved, label)
    return resolved


def _validated_model_directory(value: str | Path) -> Path:
    root = _validated_local_path(value, label="Local speech model directory", directory=True)
    for name in _REQUIRED_MODEL_FILES:
        _validated_local_path(
            root / name,
            label=f"Local speech model asset {name}",
            directory=False,
            nonempty=True,
        )

    vocabulary_found = False
    for name in _VOCABULARY_FILES:
        candidate = root / name
        try:
            os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError:
            raise LocalSTTError("Local speech vocabulary could not be validated.") from None
        _validated_local_path(
            candidate,
            label=f"Local speech model asset {name}",
            directory=False,
            nonempty=True,
        )
        vocabulary_found = True
    if not vocabulary_found:
        raise LocalSTTError(
            "Local speech model is incomplete; vocabulary.json or vocabulary.txt is required."
        )
    for name in _OPTIONAL_MODEL_FILES:
        candidate = root / name
        try:
            os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError:
            raise LocalSTTError(f"Local speech model asset {name} could not be validated.") from None
        _validated_local_path(
            candidate,
            label=f"Local speech model asset {name}",
            directory=False,
            nonempty=True,
        )
    return root


def _validated_audio_file(value: str | Path) -> Path:
    path = _validated_local_path(
        value,
        label="Audio file",
        directory=False,
        nonempty=True,
    )
    try:
        audio_size = os.lstat(path).st_size
    except OSError:
        raise LocalSTTError("Audio file could not be measured safely.") from None
    if audio_size > _MAX_AUDIO_BYTES:
        raise LocalSTTError("Audio file exceeds the safety size limit.")
    return path


def _probe_audio_duration(path: Path) -> float:
    """Read local container metadata before starting expensive inference."""
    try:
        import av
    except ImportError as exc:
        raise LocalSTTError(
            "PyAV is unavailable; install the local voice extra before transcription."
        ) from exc
    try:
        with av.open(str(path), mode="r") as container:
            candidates: list[float] = []
            if container.duration is not None:
                candidates.append(float(container.duration) / 1_000_000.0)
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    candidates.append(float(stream.duration * stream.time_base))
    except Exception as exc:
        raise LocalSTTError(
            f"Audio duration probe failed ({type(exc).__name__})."
        ) from None
    finite = [value for value in candidates if math.isfinite(value) and value > 0.0]
    if not finite:
        raise LocalSTTError("Audio duration could not be determined safely.")
    return max(finite)


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
        audio_duration_probe: Callable[[Path], float] | None = None,
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Unsupported faster-whisper device.")
        if not isinstance(compute_type, str) or not compute_type or len(compute_type) > 40:
            raise ValueError("Invalid faster-whisper compute type.")
        self.model_path = _validated_model_directory(model_path)
        self.device = device
        self.compute_type = compute_type
        self._model_factory = model_factory
        self._audio_duration_probe = audio_duration_probe
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            # Recheck the complete bundle immediately before loading so a
            # post-construction symlink/junction replacement cannot add egress.
            if _validated_model_directory(self.model_path) != self.model_path:
                raise LocalSTTError("Local speech model path changed before loading.")
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
                model = factory(
                    str(self.model_path),
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                )
            except Exception as exc:
                raise LocalSTTError(
                    f"Local speech model failed to load ({type(exc).__name__})."
                ) from None
            if model is None:
                raise LocalSTTError("Local speech model factory returned no model.")
            self._model = model
            return model

    def _validated_duration(self, audio_path: Path) -> float | None:
        probe = self._audio_duration_probe
        # A custom model factory is a trusted test/embedding seam. Production
        # faster-whisper always probes through its PyAV dependency.
        if probe is None and self._model_factory is not None:
            return None
        probe = probe or _probe_audio_duration
        try:
            duration = probe(audio_path)
        except LocalSTTError:
            raise
        except Exception as exc:
            raise LocalSTTError(
                f"Audio duration probe failed ({type(exc).__name__})."
            ) from None
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise LocalSTTError("Audio duration probe returned an invalid value.")
        value = float(duration)
        if not math.isfinite(value) or value <= 0.0:
            raise LocalSTTError("Audio duration probe returned an invalid value.")
        if value > _MAX_AUDIO_SECONDS:
            raise LocalSTTError("Audio duration exceeds the safety limit.")
        return value

    @staticmethod
    def _bounded_probability(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return max(0.0, min(1.0, number))

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> LocalTranscript:
        audio_path = _validated_audio_file(audio)
        if language is not None:
            if not isinstance(language, str) or not _LANGUAGE.fullmatch(language.strip()):
                raise ValueError("Invalid language hint.")
            language = language.strip()
        if isinstance(beam_size, bool) or not isinstance(beam_size, int) or not 1 <= beam_size <= 10:
            raise ValueError("beam_size must be between 1 and 10.")
        if not isinstance(vad_filter, bool):
            raise ValueError("vad_filter must be boolean.")
        self._validated_duration(audio_path)

        model = self._load()
        if _validated_audio_file(audio_path) != audio_path:
            raise LocalSTTError("Audio file path changed before transcription.")
        self._validated_duration(audio_path)
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            pieces: list[str] = []
            transcript_length = 0
            for segment_count, segment in enumerate(segments, start=1):
                if segment_count > _MAX_SEGMENTS:
                    raise LocalSTTError("Transcript exceeds the segment safety limit.")
                text = getattr(segment, "text", "")
                if isinstance(text, str):
                    piece = "".join(
                        character if ord(character) >= 32 and ord(character) != 127 else " "
                        for character in text
                    ).strip()
                    if not piece:
                        continue
                    added = len(piece) + (1 if pieces else 0)
                    if transcript_length + added > _MAX_TRANSCRIPT_CHARS:
                        raise LocalSTTError("Transcript exceeds the safety size limit.")
                    pieces.append(piece)
                    transcript_length += added
        except LocalSTTError:
            raise
        except Exception as exc:
            raise LocalSTTError(f"Local transcription failed ({type(exc).__name__}).") from None

        detected = getattr(info, "language", None)
        if not isinstance(detected, str) or not _LANGUAGE.fullmatch(detected.strip()):
            detected = language
        elif isinstance(detected, str):
            detected = detected.strip()
        duration = getattr(info, "duration", None)
        duration_value = None
        if not isinstance(duration, bool) and isinstance(duration, (int, float)):
            try:
                finite_duration = float(duration)
            except (OverflowError, ValueError):
                finite_duration = math.nan
            if math.isfinite(finite_duration) and finite_duration >= 0:
                if finite_duration > _MAX_AUDIO_SECONDS:
                    raise LocalSTTError("Transcribed audio duration exceeds the safety limit.")
                duration_value = finite_duration
        probability = self._bounded_probability(getattr(info, "language_probability", None))
        return LocalTranscript(
            text=" ".join(pieces),
            language=detected,
            language_probability=probability,
            duration_seconds=duration_value,
        )
