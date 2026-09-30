"""Unified persona-aware NEXUS agent runtime."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from agent.executor import AgentExecutor
from core.hardware_profile import HardwareProfiler
from core.model_router import ModelRouter, TaskKind
from core.model_runtime import ModelRuntime
from core.personas import BUILTIN_PERSONAS, TABY, PersonaSpec
from core.routed_provider import PersonaModelProvider, ProviderFactory


class AgentRuntime:
    """Compose personas, adaptive routing, model lifecycle and action execution.

    The runtime is intentionally small: models propose plans, PersonaSpec narrows
    available tools, and OwnerPolicy/ActionKernel remain authoritative for every
    action. Switching persona never restarts the process or mutates global policy.
    """

    def __init__(
        self,
        awareness=None,
        *,
        personas: Mapping[str, PersonaSpec] | None = None,
        hardware_profiler: HardwareProfiler | None = None,
        model_runtime: ModelRuntime | None = None,
        router: ModelRouter | None = None,
        provider_factory: ProviderFactory | None = None,
        default_persona: str = TABY.name,
    ):
        source = personas or BUILTIN_PERSONAS
        normalized: dict[str, PersonaSpec] = {}
        for name, spec in source.items():
            if not isinstance(spec, PersonaSpec):
                raise TypeError("personas must contain PersonaSpec values")
            if name != spec.name:
                raise ValueError("persona mapping key must match PersonaSpec.name")
            normalized[name] = spec
        if default_persona not in normalized:
            raise ValueError("default_persona must exist in personas")

        self.awareness = awareness
        self.personas = MappingProxyType(normalized)
        self._profiler = hardware_profiler or HardwareProfiler()
        self._model_runtime = model_runtime or ModelRuntime()
        self._router = router or ModelRouter()
        self._provider_factory = provider_factory
        self._active_persona = default_persona

        self.last_executor: AgentExecutor | None = None
        self.last_provider: PersonaModelProvider | None = None

    @property
    def active_persona(self) -> PersonaSpec:
        return self.personas[self._active_persona]

    def switch_persona(self, name: str) -> PersonaSpec:
        if not isinstance(name, str) or name not in self.personas:
            raise ValueError("Unknown persona")
        self._active_persona = name
        return self.active_persona

    def execute(
        self,
        goal: str,
        *,
        persona: str | None = None,
        task: TaskKind | None = None,
        speak=None,
        cancel_flag=None,
    ) -> str:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        spec = self.active_persona if persona is None else self.personas.get(persona)
        if spec is None:
            raise ValueError("Unknown persona")
        resolved_task = spec.resolve_task(task)

        provider = PersonaModelProvider(
            persona=spec,
            task=resolved_task,
            hardware_profiler=self._profiler,
            model_runtime=self._model_runtime,
            router=self._router,
            provider_factory=self._provider_factory,
        )
        executor = AgentExecutor(
            self.awareness,
            provider=provider,
            persona=spec,
        )

        self.last_provider = provider
        self.last_executor = executor
        return executor.execute(
            goal,
            speak=speak,
            cancel_flag=cancel_flag,
        )
