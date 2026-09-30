"""Pure adaptive model routing for NEXUS.

The router consumes immutable request/hardware/runtime/persona data and returns
an execution plan. It never starts/stops models, reads the OS, performs network
I/O, or dispatches actions.

Persona-specific preferences live in PersonaRoutingProfile. The routing kernel
does not hard-code TABY/FRIDAY/JARVIS names or device IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping

from core.hardware_profile import HardwareSnapshot, PowerSource
from core.model_provider import ModelRequest, ModelTier
from core.model_runtime import ModelInfo, ModelState, RuntimeStatus


class ProviderKind(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"


class TaskKind(str, Enum):
    GENERAL = "general"
    REALTIME = "realtime"
    CODING = "coding"


@dataclass(frozen=True)
class LocalModelCandidate:
    model: str
    min_available_ram_gb: float
    priority: int
    keep_alive: str | int | None
    tiers: FrozenSet[ModelTier]
    tasks: FrozenSet[TaskKind]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("candidate model must be a non-empty string")
        if self.min_available_ram_gb < 0:
            raise ValueError("min_available_ram_gb cannot be negative")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("candidate priority must be an integer")
        if not self.tiers:
            raise ValueError("candidate must support at least one ModelTier")
        if not self.tasks:
            raise ValueError("candidate must support at least one TaskKind")
        object.__setattr__(self, "tiers", frozenset(self.tiers))
        object.__setattr__(self, "tasks", frozenset(self.tasks))


@dataclass(frozen=True)
class PersonaRoutingProfile:
    """Routing-only slice that future PersonaSpec can compose."""

    name: str
    local_candidates: tuple[LocalModelCandidate, ...]
    cloud_fast_model: str
    cloud_standard_model: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("persona routing profile requires a name")
        if not self.local_candidates:
            raise ValueError("persona routing profile requires local candidates")
        if not self.cloud_fast_model or not self.cloud_standard_model:
            raise ValueError("cloud fallback model names must be non-empty")
        object.__setattr__(self, "local_candidates", tuple(self.local_candidates))


@dataclass(frozen=True)
class RouterPolicy:
    low_memory_total_gb: float = 10.0
    high_memory_total_gb: float = 24.0
    max_warm_low_memory: int = 1
    max_warm_balanced: int = 2
    max_warm_high_memory: int = 3
    new_load_pressure_limit: float = 0.85
    battery_pressure_limit: float = 0.80
    battery_min_available_ram_gb: float = 3.0
    warm_hard_stop_pressure: float = 0.97

    def __post_init__(self) -> None:
        if not 0 < self.low_memory_total_gb < self.high_memory_total_gb:
            raise ValueError("memory tier thresholds must be increasing and positive")
        for value in (
            self.max_warm_low_memory,
            self.max_warm_balanced,
            self.max_warm_high_memory,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("warm-model limits must be positive integers")
        for value in (
            self.new_load_pressure_limit,
            self.battery_pressure_limit,
            self.warm_hard_stop_pressure,
        ):
            if not 0 <= value <= 1:
                raise ValueError("pressure thresholds must be between 0 and 1")
        if self.battery_min_available_ram_gb < 0:
            raise ValueError("battery_min_available_ram_gb cannot be negative")


@dataclass(frozen=True)
class ProviderChoice:
    provider: ProviderKind
    model: str
    reason: str
    ensure_priority: int | None = None
    keep_alive: str | int | None = None
    requires_ensure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("ProviderChoice.model must be non-empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("ProviderChoice.reason must be non-empty")
        if self.provider is ProviderKind.OLLAMA:
            if self.ensure_priority is None:
                raise ValueError("Ollama choices require ensure_priority")
        else:
            if self.requires_ensure:
                raise ValueError("cloud choices cannot require local ensure")
            if self.ensure_priority is not None:
                raise ValueError("cloud choices cannot carry ensure_priority")


@dataclass(frozen=True)
class RoutePlan:
    primary: ProviderChoice
    fallbacks: tuple[ProviderChoice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))

    @property
    def choices(self) -> tuple[ProviderChoice, ...]:
        return (self.primary, *self.fallbacks)


class ModelRouter:
    """Pure policy engine for choosing provider/model execution order."""

    def __init__(self, policy: RouterPolicy | None = None):
        self._policy = policy or RouterPolicy()

    def _warm_limit(self, hw: HardwareSnapshot) -> int:
        if hw.total_ram_gb <= self._policy.low_memory_total_gb:
            return self._policy.max_warm_low_memory
        if hw.total_ram_gb >= self._policy.high_memory_total_gb:
            return self._policy.max_warm_high_memory
        return self._policy.max_warm_balanced

    @staticmethod
    def _warm_models(runtime_status: RuntimeStatus) -> Mapping[str, ModelInfo]:
        return MappingProxyType(
            {
                name: info
                for name, info in runtime_status.models.items()
                if info.state is ModelState.WARM
            }
        )

    @staticmethod
    def _cloud_choice(
        request: ModelRequest,
        persona: PersonaRoutingProfile,
        reason: str,
    ) -> ProviderChoice:
        model = (
            persona.cloud_fast_model
            if request.tier is ModelTier.FAST
            else persona.cloud_standard_model
        )
        return ProviderChoice(
            provider=ProviderKind.GEMINI,
            model=model,
            reason=reason,
        )

    @staticmethod
    def _supports(
        candidate: LocalModelCandidate,
        request: ModelRequest,
        task: TaskKind,
    ) -> bool:
        return request.tier in candidate.tiers and task in candidate.tasks

    def _new_load_allowed(
        self,
        candidate: LocalModelCandidate,
        hw: HardwareSnapshot,
        warm_count: int,
    ) -> tuple[bool, str]:
        if hw.system_pressure >= self._policy.new_load_pressure_limit:
            return False, "system_pressure_blocks_cold_load"
        if hw.available_ram_gb < candidate.min_available_ram_gb:
            return False, "insufficient_available_ram"
        if warm_count >= self._warm_limit(hw):
            return False, "warm_capacity_exhausted"
        if hw.power_source is PowerSource.BATTERY and (
            hw.memory_pressure >= self._policy.battery_pressure_limit
            or hw.available_ram_gb < self._policy.battery_min_available_ram_gb
        ):
            return False, "battery_low_resource_policy"
        return True, "cold_local_allowed"

    def route(
        self,
        request: ModelRequest,
        hw: HardwareSnapshot,
        runtime_status: RuntimeStatus,
        persona: PersonaRoutingProfile,
        *,
        task: TaskKind = TaskKind.GENERAL,
    ) -> RoutePlan:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")
        if not isinstance(hw, HardwareSnapshot):
            raise TypeError("hw must be HardwareSnapshot")
        if not isinstance(runtime_status, RuntimeStatus):
            raise TypeError("runtime_status must be RuntimeStatus")
        if not isinstance(persona, PersonaRoutingProfile):
            raise TypeError("persona must be PersonaRoutingProfile")
        if not isinstance(task, TaskKind):
            raise TypeError("task must be TaskKind")

        candidates = tuple(
            candidate
            for candidate in persona.local_candidates
            if self._supports(candidate, request, task)
        )
        cloud = self._cloud_choice(
            request,
            persona,
            reason="cloud_fallback",
        )
        if not candidates:
            return RoutePlan(
                primary=self._cloud_choice(
                    request,
                    persona,
                    reason="no_compatible_local_candidate",
                )
            )

        warm = self._warm_models(runtime_status)

        # Warm compatible models are cheapest to use because they avoid a cold load.
        # Keep persona preference order deterministic.
        warm_choices: list[ProviderChoice] = []
        if hw.system_pressure < self._policy.warm_hard_stop_pressure:
            for candidate in candidates:
                if candidate.model in warm:
                    warm_choices.append(
                        ProviderChoice(
                            provider=ProviderKind.OLLAMA,
                            model=candidate.model,
                            reason="compatible_model_already_warm",
                            ensure_priority=candidate.priority,
                            keep_alive=candidate.keep_alive,
                            requires_ensure=False,
                        )
                    )

        cold_choices: list[ProviderChoice] = []
        warm_count = len(warm)
        for candidate in candidates:
            if candidate.model in warm:
                continue
            allowed, reason = self._new_load_allowed(candidate, hw, warm_count)
            if not allowed:
                continue
            cold_choices.append(
                ProviderChoice(
                    provider=ProviderKind.OLLAMA,
                    model=candidate.model,
                    reason=reason,
                    ensure_priority=candidate.priority,
                    keep_alive=candidate.keep_alive,
                    requires_ensure=True,
                )
            )

        ordered = [*warm_choices, *cold_choices]
        if not ordered:
            reason = (
                "critical_pressure_cloud_fallback"
                if hw.system_pressure >= self._policy.warm_hard_stop_pressure
                else "local_resource_policy_cloud_fallback"
            )
            return RoutePlan(
                primary=self._cloud_choice(request, persona, reason=reason)
            )

        # Always give the caller a deterministic cloud escape hatch if a local
        # ensure/inference fails. Deduplicate choices by provider/model pair.
        deduped: list[ProviderChoice] = []
        seen: set[tuple[ProviderKind, str]] = set()
        for choice in [*ordered, cloud]:
            key = (choice.provider, choice.model)
            if key not in seen:
                seen.add(key)
                deduped.append(choice)

        return RoutePlan(
            primary=deduped[0],
            fallbacks=tuple(deduped[1:]),
        )
