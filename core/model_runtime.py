"""Lifecycle manager for locally loaded Ollama models.

ModelRuntime owns *execution state*, not model choice. It can load, refresh,
release, inspect, and evict NEXUS-managed models. Provider/persona selection
belongs to model_router.py.

Important safety boundary: eviction only targets models this runtime has marked
as managed. Models loaded by other applications are observable through /api/ps
but are never evicted by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from types import MappingProxyType
from typing import Callable, Mapping

from core.hardware_profile import HardwareSnapshot, _normalize_base_url


_GIB = 1024 ** 3


class ModelRuntimeError(RuntimeError):
    """Safe Ollama lifecycle failure."""


class ModelState(str, Enum):
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class ModelInfo:
    model: str
    state: ModelState
    priority: int
    managed: bool
    reported_size_gb: float | None = None
    vram_gb: float | None = None
    expires_at: datetime | None = None
    last_used: datetime | None = None


@dataclass(frozen=True)
class ModelHandle:
    model: str
    state: ModelState
    priority: int
    keep_alive: str | int | None
    last_used: datetime
    reported_size_gb: float | None = None
    vram_gb: float | None = None


@dataclass(frozen=True)
class RuntimeStatus:
    models: Mapping[str, ModelInfo]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ollama_time(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


class ModelRuntime:
    """Manage Ollama model warm/cold lifecycle without making routing decisions."""

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        default_keep_alive: str | int | None = "5m",
        timeout_seconds: float = 30.0,
        http_get: Callable | None = None,
        http_post: Callable | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = _normalize_base_url(
            ollama_base_url
            or os.getenv("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        self._default_keep_alive = default_keep_alive
        self._timeout = timeout_seconds
        self._http_get = http_get
        self._http_post = http_post
        self._clock = clock or _utc_now

        self._priorities: dict[str, int] = {}
        self._last_used: dict[str, datetime] = {}
        self._managed: set[str] = set()
        self._known: set[str] = set()
        self._warnings: list[str] = []

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("ModelRuntime clock must return timezone-aware datetime")
        return now.astimezone(timezone.utc)

    def _get(self):
        if self._http_get is None:
            import requests
            self._http_get = requests.get
        return self._http_get

    def _post(self):
        if self._http_post is None:
            import requests
            self._http_post = requests.post
        return self._http_post

    @staticmethod
    def _safe_error(exc: Exception, operation: str) -> ModelRuntimeError:
        name = type(exc).__name__
        if "Timeout" in name:
            return ModelRuntimeError(f"Ollama {operation} timed out.")
        if name in {"ConnectionError"}:
            return ModelRuntimeError(f"Ollama is unreachable during {operation}.")
        return ModelRuntimeError(f"Ollama {operation} failed.")

    def _list_loaded_raw(self) -> dict[str, ModelInfo]:
        try:
            response = self._get()(
                f"{self._base_url}/api/ps",
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise self._safe_error(exc, "status query") from None

        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            raise ModelRuntimeError("Ollama returned an invalid model-status response.")

        loaded: dict[str, ModelInfo] = {}
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if not isinstance(name, str) or not name.strip():
                continue

            size = item.get("size")
            size_vram = item.get("size_vram")
            reported_size_gb = (
                float(size) / _GIB
                if isinstance(size, (int, float)) and not isinstance(size, bool) and size >= 0
                else None
            )
            vram_gb = (
                float(size_vram) / _GIB
                if isinstance(size_vram, (int, float))
                and not isinstance(size_vram, bool)
                and size_vram >= 0
                else None
            )
            loaded[name] = ModelInfo(
                model=name,
                state=ModelState.WARM,
                priority=self._priorities.get(name, 0),
                managed=name in self._managed,
                reported_size_gb=reported_size_gb,
                vram_gb=vram_gb,
                expires_at=_parse_ollama_time(item.get("expires_at")),
                last_used=self._last_used.get(name),
            )
        return loaded

    def get_loaded(self) -> dict[str, ModelInfo]:
        """Return currently resident Ollama models, including unmanaged external ones."""
        return self._list_loaded_raw()

    def get_status(self) -> RuntimeStatus:
        loaded = self._list_loaded_raw()
        models = dict(loaded)
        for model in self._known - set(loaded):
            models[model] = ModelInfo(
                model=model,
                state=ModelState.COLD,
                priority=self._priorities.get(model, 0),
                managed=model in self._managed,
                last_used=self._last_used.get(model),
            )
        return RuntimeStatus(models=models, warnings=tuple(self._warnings))

    def _touch_model(
        self,
        model: str,
        *,
        keep_alive: str | int | None,
        operation: str,
    ) -> None:
        payload = {
            "model": model,
            "messages": [],
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            response = self._post()(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise self._safe_error(exc, operation) from None

        if not isinstance(body, dict):
            raise ModelRuntimeError(f"Ollama returned an invalid {operation} response.")
        returned_model = body.get("model")
        if returned_model is not None and (
            not isinstance(returned_model, str) or not returned_model.strip()
        ):
            raise ModelRuntimeError(f"Ollama returned an invalid {operation} response.")

    def ensure(
        self,
        model: str,
        priority: int,
        *,
        keep_alive: str | int | None = None,
    ) -> ModelHandle:
        """Ensure a model is loaded and owned by NEXUS runtime management."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("priority must be an integer")

        ttl = self._default_keep_alive if keep_alive is None else keep_alive
        self._touch_model(model, keep_alive=ttl, operation="model load")
        now = self._now()

        self._managed.add(model)
        self._known.add(model)
        self._priorities[model] = priority
        self._last_used[model] = now

        info = None
        try:
            info = self._list_loaded_raw().get(model)
        except ModelRuntimeError:
            # The successful load is authoritative enough to return a handle.
            # Status refresh can retry later.
            self._warnings.append(f"status_refresh_failed:{model}")

        return ModelHandle(
            model=model,
            state=ModelState.WARM,
            priority=priority,
            keep_alive=ttl,
            last_used=now,
            reported_size_gb=info.reported_size_gb if info else None,
            vram_gb=info.vram_gb if info else None,
        )

    def release(self, model: str, ttl: int = 300) -> ModelHandle:
        """Schedule unload, or unload immediately when ttl=0.

        A cold model is never loaded merely to schedule a future release.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 0:
            raise ValueError("ttl must be a non-negative integer")

        self._known.add(model)
        now = self._now()

        try:
            loaded = self._list_loaded_raw()
        except ModelRuntimeError:
            loaded = {}
            self._warnings.append(f"status_refresh_failed:{model}")

        if model not in loaded:
            return ModelHandle(
                model=model,
                state=ModelState.COLD,
                priority=self._priorities.get(model, 0),
                keep_alive=0,
                last_used=self._last_used.get(model, now),
            )

        self._touch_model(model, keep_alive=ttl, operation="model release")
        if ttl == 0:
            state = ModelState.COLD
        else:
            state = ModelState.WARM
            self._last_used[model] = now

        return ModelHandle(
            model=model,
            state=state,
            priority=self._priorities.get(model, 0),
            keep_alive=ttl,
            last_used=self._last_used.get(model, now),
            reported_size_gb=loaded[model].reported_size_gb,
            vram_gb=loaded[model].vram_gb,
        )

    def evict_lowest_priority(
        self,
        *,
        protected: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> ModelHandle | None:
        """Immediately unload the lowest-priority NEXUS-managed resident model."""
        protected_set = set(protected)
        loaded = self._list_loaded_raw()

        candidates = [
            info
            for name, info in loaded.items()
            if name in self._managed and name not in protected_set
        ]
        if not candidates:
            return None

        epoch = datetime.min.replace(tzinfo=timezone.utc)
        victim = min(
            candidates,
            key=lambda info: (
                info.priority,
                info.last_used or epoch,
                info.model,
            ),
        )
        return self.release(victim.model, ttl=0)

    def relieve_pressure(
        self,
        snapshot: HardwareSnapshot,
        *,
        threshold: float = 0.90,
        protected: set[str] | frozenset[str] | tuple[str, ...] = (),
    ) -> ModelHandle | None:
        """Evict at most one managed model when the current snapshot is critical.

        The caller should capture a fresh HardwareSnapshot after eviction before
        deciding whether more action is required.
        """
        if not isinstance(snapshot, HardwareSnapshot):
            raise TypeError("snapshot must be HardwareSnapshot")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if snapshot.system_pressure <= threshold:
            return None

        victim = self.evict_lowest_priority(protected=protected)
        if victim is not None:
            self._warnings.append(
                f"emergency_eviction:{victim.model}:pressure={snapshot.system_pressure:.3f}"
            )
        return victim
