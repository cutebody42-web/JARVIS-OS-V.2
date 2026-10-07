"""Persistent, owner-controlled wake-word session state.

This controller owns configuration and inference state only. It never opens the
microphone, sends audio over the network, executes a command, or grants action
authority. Product surfaces feed bounded 16 kHz PCM frames after explicit
owner opt-in. A detection is an activation signal only.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import threading
import time
from typing import Any

from core.wake_word import LocalWakeWordDetector, WakeWordError, WakeWordModelFiles


class WakeWordSessionError(RuntimeError):
    pass


class WakeWordSession:
    """Thread-safe local wake-word configuration and activation state."""

    _MAX_BASE64_BYTES = 240_000
    _COOLDOWN_SECONDS = 1.5

    def __init__(self, settings_path: str | Path) -> None:
        self.settings_path = Path(settings_path)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._guard = threading.RLock()
        self._enabled = False
        self._model_dir: Path | None = None
        self._threshold = 0.5
        self._detector: LocalWakeWordDetector | None = None
        self._last_activation_monotonic = 0.0
        self._last_activation_at: str | None = None
        self._last_score: float | None = None
        self._last_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.settings_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        raw_dir = payload.get("model_dir")
        raw_threshold = payload.get("threshold", 0.5)
        raw_enabled = payload.get("enabled", False)
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raw_threshold = 0.5
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            threshold = 0.5
        self._threshold = threshold
        if isinstance(raw_dir, str) and raw_dir.strip():
            path = Path(raw_dir.strip()).expanduser()
            if path.is_absolute():
                try:
                    self._configure_detector(path, threshold)
                except WakeWordSessionError:
                    self._last_error = "configured_model_unavailable"
        # Never silently reopen the microphone after a process restart. The UI
        # must opt in each session even if the persistent preference was true.
        self._enabled = False
        if raw_enabled is True:
            self._last_error = self._last_error or "owner_reenable_required"

    def _write(self) -> None:
        payload = {
            "schema": "jarvis.wake-word-settings.v1",
            "model_dir": str(self._model_dir) if self._model_dir is not None else None,
            "threshold": self._threshold,
            "enabled": self._enabled,
        }
        temp = self.settings_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        temp.replace(self.settings_path)

    def _configure_detector(self, model_dir: Path, threshold: float) -> None:
        if not model_dir.is_absolute():
            raise WakeWordSessionError("Wake-word model directory must be an absolute path.")
        try:
            files = WakeWordModelFiles.from_directory(model_dir)
            detector = LocalWakeWordDetector(files, threshold=threshold)
        except (WakeWordError, OSError, ValueError) as exc:
            raise WakeWordSessionError(str(exc)) from exc
        self._model_dir = model_dir.resolve()
        self._threshold = threshold
        self._detector = detector
        self._last_error = None

    def configure(self, model_dir: str, *, threshold: float = 0.5) -> dict[str, Any]:
        if not isinstance(model_dir, str) or not model_dir.strip() or len(model_dir) > 4096:
            raise ValueError("model_dir must be a non-empty absolute path")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be between 0 and 1")
        score = float(threshold)
        if not 0.0 <= score <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        path = Path(model_dir.strip()).expanduser()
        with self._guard:
            self._configure_detector(path, score)
            self._enabled = False
            self._last_activation_monotonic = 0.0
            self._last_activation_at = None
            self._last_score = None
            self._write()
            return self.status()

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        with self._guard:
            if enabled and self._detector is None:
                raise WakeWordSessionError("Configure owner-local wake-word model files before enabling.")
            self._enabled = enabled
            self._last_error = None
            if not enabled and self._detector is not None:
                try:
                    self._detector.reset()
                except WakeWordError:
                    self._last_error = "detector_reset_failed"
            self._write()
            return self.status()

    @staticmethod
    def _decode_pcm16(payload: str) -> tuple[int, ...]:
        if not isinstance(payload, str) or not payload:
            raise ValueError("pcm16_base64 must be non-empty base64 text")
        if len(payload) > WakeWordSession._MAX_BASE64_BYTES:
            raise ValueError("PCM frame is too large")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError("pcm16_base64 is invalid") from exc
        if len(raw) % 2 != 0:
            raise ValueError("PCM16 byte length must be even")
        sample_count = len(raw) // 2
        if sample_count < 160 or sample_count > LocalWakeWordDetector.SAMPLE_RATE * 5:
            raise ValueError("PCM frame duration is outside the allowed bounds")
        return struct.unpack("<" + "h" * sample_count, raw)

    def process_base64_pcm(self, pcm16_base64: str) -> dict[str, Any]:
        samples = self._decode_pcm16(pcm16_base64)
        with self._guard:
            if not self._enabled:
                raise WakeWordSessionError("Wake-word activation is disabled.")
            detector = self._detector
            if detector is None:
                raise WakeWordSessionError("Wake-word model is not configured.")
            try:
                result = detector.process(samples)
            except (WakeWordError, ValueError) as exc:
                self._last_error = type(exc).__name__
                raise WakeWordSessionError(str(exc)) from exc
            self._last_score = result.score
            now = time.monotonic()
            activated = bool(
                result.detected
                and now - self._last_activation_monotonic >= self._COOLDOWN_SECONDS
            )
            if activated:
                self._last_activation_monotonic = now
                self._last_activation_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            self._last_error = None
            return {
                "detected": result.detected,
                "activated": activated,
                "score": result.score,
                "threshold": result.threshold,
                "model_name": result.model_name,
                "authority": "activation_only",
            }

    def status(self) -> dict[str, Any]:
        with self._guard:
            detector_status = self._detector.status() if self._detector is not None else None
            return {
                "configured": self._detector is not None,
                "enabled": self._enabled,
                "model_dir": str(self._model_dir) if self._model_dir is not None else None,
                "threshold": self._threshold,
                "sample_rate": LocalWakeWordDetector.SAMPLE_RATE,
                "chunk_samples": LocalWakeWordDetector.DEFAULT_CHUNK_SAMPLES,
                "last_score": self._last_score,
                "last_activation_at": self._last_activation_at,
                "last_error": self._last_error,
                "local_only": True,
                "microphone_owner": False,
                "authority": "activation_only",
                "detector": detector_status,
            }
