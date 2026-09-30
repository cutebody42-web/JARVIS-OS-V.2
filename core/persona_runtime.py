"""Hot-switchable persona composition for the Phase-1 AgentExecutor.

This runtime shares hardware/model lifecycle state across persona switches.
It is intentionally separate from the Phase-3 OwnerRuntime authorization
boundary; a later integration layer composes both rather than conflating them.
"""

from __future__ import annotations

import threading
from typing import Callable

from agent.executor import AgentExecutor
from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider
from core.model_router import ModelRouter, ProviderChoice, TaskKind
from core.model_runtime import ModelRuntime
from core.personas import PersonaSpec, TABY, get_persona
from core.routed_model_provider import RoutedModelProvider


class PersonaAgentRuntime:
    def __init__(
        self,
        persona: PersonaSpec = TABY,
        *,
        task: TaskKind | None = None,
        awareness=None,
        profiler: HardwareProfiler | None = None,
        model_runtime: ModelRuntime | None = None,
        router: ModelRouter | None = None,
        ollama_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
        gemini_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
        ollama_base_url: str | None = None,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        self._guard = threading.RLock()
        self._awareness = awareness
        self._profiler = profiler or HardwareProfiler(ollama_base_url=ollama_base_url)
        self._model_runtime = model_runtime or ModelRuntime(ollama_base_url=ollama_base_url)
        self._router = router or ModelRouter()
        self._ollama_factory = ollama_factory
        self._gemini_factory = gemini_factory
        self._persona = persona
        self._task = persona.validate_task(task)
        self._provider: RoutedModelProvider
        self._executor: AgentExecutor
        self._rebuild()

    def _rebuild(self) -> None:
        provider = RoutedModelProvider(
            self._persona,
            task=self._task,
            profiler=self._profiler,
            runtime=self._model_runtime,
            router=self._router,
            ollama_factory=self._ollama_factory,
            gemini_factory=self._gemini_factory,
        )
        self._provider = provider
        self._executor = AgentExecutor(
            awareness=self._awareness,
            provider=provider,
            tool_allowlist=self._persona.tool_allowlist,
        )

    @property
    def persona(self) -> PersonaSpec:
        with self._guard:
            return self._persona

    @property
    def task(self) -> TaskKind:
        with self._guard:
            return self._task

    @property
    def provider(self) -> RoutedModelProvider:
        with self._guard:
            return self._provider

    @property
    def context_namespace(self) -> str:
        return self.persona.context_namespace

    def switch_persona(
        self,
        persona: PersonaSpec | str,
        *,
        task: TaskKind | None = None,
    ) -> PersonaSpec:
        resolved = get_persona(persona) if isinstance(persona, str) else persona
        if not isinstance(resolved, PersonaSpec):
            raise TypeError("persona must be PersonaSpec or builtin persona name")
        resolved_task = resolved.validate_task(task)
        with self._guard:
            self._persona = resolved
            self._task = resolved_task
            self._rebuild()
            return self._persona

    def execute(self, goal: str, *, speak=None, cancel_flag=None) -> str:
        with self._guard:
            executor = self._executor
        return executor.execute(goal, speak=speak, cancel_flag=cancel_flag)
