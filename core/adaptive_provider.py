"""Adaptive provider binding a PersonaSpec to the pure ModelRouter.

Routing is re-evaluated for every ModelRequest. That matters because planner
requests may change tier during recovery and hardware pressure/model residency
can change between calls.

This layer does not authorize actions. It only selects cognition providers.
"""

from __future__ import annotations

from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider, ModelRequest, ModelResponse
from core.model_router import (
    ModelRouter,
    ProviderChoice,
    ProviderKind,
    TaskKind,
)
from core.model_runtime import ModelRuntime
from core.personas.persona_spec import PersonaSpec
from core.providers.gemini import GeminiProvider
from core.providers.ollama import OllamaProvider


class AdaptiveProviderError(RuntimeError):
    """All routed cognition choices failed without exposing sensitive details."""


def _default_provider_builder(choice: ProviderChoice) -> ModelProvider:
    if choice.provider is ProviderKind.OLLAMA:
        return OllamaProvider(
            fast_model=choice.model,
            standard_model=choice.model,
            keep_alive=choice.keep_alive,
        )
    if choice.provider is ProviderKind.GEMINI:
        return GeminiProvider(
            fast_model=choice.model,
            standard_model=choice.model,
        )
    raise ValueError(f"Unsupported provider kind: {choice.provider}")


class AdaptivePersonaProvider:
    """ModelProvider implementation that routes every request adaptively."""

    def __init__(
        self,
        *,
        persona: PersonaSpec,
        task: TaskKind,
        router: ModelRouter,
        hardware_profiler: HardwareProfiler,
        model_runtime: ModelRuntime,
        provider_builder: Callable[[ProviderChoice], ModelProvider] | None = None,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        if not isinstance(task, TaskKind):
            raise TypeError("task must be TaskKind")
        if not persona.allows_task(task):
            raise ValueError(
                f"Persona {persona.name!r} does not allow task kind {task.value!r}"
            )
        if not isinstance(router, ModelRouter):
            raise TypeError("router must be ModelRouter")
        if not isinstance(hardware_profiler, HardwareProfiler):
            raise TypeError("hardware_profiler must be HardwareProfiler")
        if not isinstance(model_runtime, ModelRuntime):
            raise TypeError("model_runtime must be ModelRuntime")

        self.persona = persona
        self.task = task
        self.router = router
        self.hardware_profiler = hardware_profiler
        self.model_runtime = model_runtime
        self.provider_builder = provider_builder or _default_provider_builder
        self.last_plan = None
        self.last_choice = None

    def _persona_request(self, request: ModelRequest) -> ModelRequest:
        allowed = ", ".join(sorted(self.persona.tool_allowlist)) or "(none)"
        suffix = (
            f"\n\nActive persona: {self.persona.display_name}. "
            f"{self.persona.system_instruction}\n"
            f"Persona tool ceiling: {allowed}. "
            "This list can only restrict application-owned OwnerPolicy; it does "
            "not grant authority."
        )
        return ModelRequest(
            request.prompt,
            request.system_instruction + suffix,
            request.tier,
            request.json_output,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")

        hw = self.hardware_profiler.capture()
        status = self.model_runtime.get_status()
        plan = self.router.route(
            request,
            hw,
            status,
            self.persona.routing,
            task=self.task,
        )
        self.last_plan = plan
        effective_request = self._persona_request(request)

        for choice in plan.choices:
            try:
                if choice.provider is ProviderKind.OLLAMA and choice.requires_ensure:
                    self.model_runtime.ensure(
                        choice.model,
                        choice.ensure_priority,
                        keep_alive=choice.keep_alive,
                    )
                provider = self.provider_builder(choice)
                response = provider.generate(effective_request)
                if not isinstance(response, ModelResponse):
                    raise TypeError("provider returned invalid ModelResponse")
                self.last_choice = choice
                return response
            except Exception:
                # Choice-specific failures are intentionally not echoed because
                # provider exceptions may contain credentials, endpoints or paths.
                continue

        raise AdaptiveProviderError(
            "No routed cognition provider completed the request."
        )
