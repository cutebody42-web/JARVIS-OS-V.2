"""Owner-local wake-word detection for NEXUS voice activation.

The detector is deliberately an input signal only. A positive wake-word score
never grants action authority and never bypasses the Sovereign Owner Kernel or
Action Gateway. Model files are owner-supplied local files; this module has no
model downloader and no network client.

The optional runtime adapter is compatible with openWakeWord 0.6.x using its
ONNX inference path. JARVIS owns path validation, bounds, thresholds, lifecycle,
and the activation contract around that runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence


class WakeWordError(RuntimeError):
    """Raised when the local wake-word runtime cannot operate safely."""


@dataclass(frozen=True)
class WakeWordDetection:
    detected: bool
    score: float
    model_name: str
    threshold: float
    engine: str = "openwakeword-onnx"


@dataclass(frozen=True)
class WakeWordModelFiles:
    wakeword_model: Path
    melspec_model: Path
    embedding_model: Path

    @classmethod
    def from_directory(cls, directory: str | Path, *, wakeword_filename: str = "wakeword.onnx") -> "WakeWordModelFiles":
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise WakeWordError("Wake-word model directory does not exist.")
        return cls(
            wakeword_model=root / wakeword_filename,
            melspec_model=root / "melspectrogram.onnx",
            embedding_model=root / "embedding_model.onnx",
        ).validated(root=root)

    def validated(self, *, root: Path | None = None) -> "WakeWordModelFiles":
        resolved_root: Path | None = None
        if root is not None:
            try:
                resolved_root = Path(root).expanduser().resolve(strict=True)
            except OSError as exc:
                raise WakeWordError("Wake-word model directory does not exist.") from exc
            if not resolved_root.is_dir():
                raise WakeWordError("Wake-word model directory does not exist.")

        validated: list[Path] = []
        for value, label in (
            (self.wakeword_model, "wake-word"),
            (self.melspec_model, "melspectrogram"),
            (self.embedding_model, "audio embedding"),
        ):
            try:
                path = Path(value).expanduser().resolve(strict=True)
            except OSError as exc:
                raise WakeWordError(f"{label} model must be an existing local ONNX file.") from exc
            if resolved_root is not None and resolved_root not in path.parents:
                raise WakeWordError(f"{label} model must stay inside the configured model directory.")
            if not path.is_file() or path.suffix.lower() != ".onnx":
                raise WakeWordError(f"{label} model must be a local ONNX file.")
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise WakeWordError(f"{label} model must be an accessible local ONNX file.") from exc
            if size <= 0:
                raise WakeWordError(f"{label} model file is empty.")
            validated.append(path)
        return WakeWordModelFiles(*validated)


class LocalWakeWordDetector:
    """Bounded, local-only adapter around an owner-provided wake-word model.

    Audio input is mono signed 16-bit PCM sampled at 16 kHz. JARVIS does not
    open the microphone here; callers decide when microphone capture is enabled
    and feed bounded PCM frames to this detector. That keeps activation and
    authority separate and makes deterministic fixture testing possible.
    """

    SAMPLE_RATE = 16000
    DEFAULT_CHUNK_SAMPLES = 1280

    def __init__(
        self,
        model_files: WakeWordModelFiles,
        *,
        threshold: float = 0.5,
        model_name: str | None = None,
        model_factory: Callable[..., object] | None = None,
    ) -> None:
        self.files = model_files.validated()
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a finite number between 0 and 1")
        value = float(threshold)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be a finite number between 0 and 1")
        self.threshold = value
        name = model_name or self.files.wakeword_model.stem
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError("model_name must be short non-empty text")
        self.model_name = name.strip()
        self._factory = model_factory
        self._model: object | None = None

    @staticmethod
    def _default_factory(**kwargs):
        try:
            module = importlib.import_module("openwakeword.model")
            model_type = getattr(module, "Model")
        except (ImportError, AttributeError) as exc:
            raise WakeWordError(
                "Local wake-word runtime is not installed. Install the optional wake-word requirements."
            ) from exc
        return model_type(**kwargs)

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        factory = self._factory or self._default_factory
        try:
            self._model = factory(
                wakeword_models=[str(self.files.wakeword_model)],
                inference_framework="onnx",
                melspec_model_path=str(self.files.melspec_model),
                embedding_model_path=str(self.files.embedding_model),
            )
        except WakeWordError:
            raise
        except Exception as exc:
            raise WakeWordError("Owner-local wake-word model could not be loaded.") from exc
        if self._model is None or not callable(getattr(self._model, "predict", None)):
            self._model = None
            raise WakeWordError("Wake-word runtime does not expose a predict method.")
        return self._model

    @staticmethod
    def _coerce_pcm(samples: Sequence[int]) -> tuple[int, ...]:
        if isinstance(samples, (str, bytes, bytearray)):
            raise ValueError("samples must be signed 16-bit PCM integers")
        length = len(samples)
        if length < 160 or length > LocalWakeWordDetector.SAMPLE_RATE * 5:
            raise ValueError("audio frame must contain between 10 ms and 5 seconds of 16 kHz PCM")
        values: list[int] = []
        for sample in samples:
            if isinstance(sample, bool) or not isinstance(sample, int):
                raise ValueError("samples must be signed 16-bit PCM integers")
            if sample < -32768 or sample > 32767:
                raise ValueError("PCM sample is outside signed 16-bit range")
            values.append(sample)
        return tuple(values)

    @staticmethod
    def _score(prediction: object, model_name: str) -> float:
        if not isinstance(prediction, Mapping) or not prediction:
            raise WakeWordError("Wake-word runtime returned an invalid prediction.")
        candidate = prediction.get(model_name)
        if candidate is None and len(prediction) == 1:
            candidate = next(iter(prediction.values()))
        if isinstance(candidate, Mapping):
            numeric = [value for value in candidate.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
            candidate = max(numeric) if numeric else None
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise WakeWordError("Wake-word runtime returned a non-numeric score.")
        score = float(candidate)
        if not math.isfinite(score):
            raise WakeWordError("Wake-word runtime returned a non-finite score.")
        return max(0.0, min(1.0, score))

    def process(self, samples: Sequence[int]) -> WakeWordDetection:
        pcm = self._coerce_pcm(samples)
        model = self._load()
        try:
            numpy = importlib.import_module("numpy")
            frame = numpy.asarray(pcm, dtype=numpy.int16)
            prediction = model.predict(frame)
        except ImportError as exc:
            raise WakeWordError("NumPy is required by the local wake-word runtime.") from exc
        except WakeWordError:
            raise
        except Exception as exc:
            raise WakeWordError("Wake-word inference failed.") from exc
        score = self._score(prediction, self.model_name)
        return WakeWordDetection(
            detected=score >= self.threshold,
            score=score,
            model_name=self.model_name,
            threshold=self.threshold,
        )

    def reset(self) -> None:
        model = self._model
        if model is not None and callable(getattr(model, "reset", None)):
            try:
                model.reset()
            except Exception as exc:
                raise WakeWordError("Wake-word runtime reset failed.") from exc

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "engine": "openwakeword-onnx",
            "model_name": self.model_name,
            "sample_rate": self.SAMPLE_RATE,
            "threshold": self.threshold,
            "loaded": self._model is not None,
            "local_only": True,
            "downloads": False,
            "authority": "activation_only",
        }
