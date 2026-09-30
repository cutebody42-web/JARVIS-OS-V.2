"""Adaptive provider execution for persona-bound NEXUS requests.

This layer executes an already-defined RoutePlan. It does not decide action
permissions and it does not mutate OwnerPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider, ModelRequest, ModelResponse
from core.model_router import (
    ModelRouter,
    ProviderChoice,
    ProviderKind,
    RoutePlan,
    TaskKind,
)
from core.model_runtime import ModelRuntime
from core.personas import PersonaSpec
from core.providers.gemini import GeminiProvider
from core.providers.ollama import OllamaProvider


class AdaptiveProviderError(RuntimeError):
    """All routed providers failed without leaking raw provider errors."""


@dataclass(frozen=True)
class ProviderAttempt:
    provider: ProviderKind
    model: str
    error_type: str


class AdaptivePersonaProvider:
    """ModelProvider implementation backed by hardware-aware route plans."""

    def __init__(
        self,
        *,
        persona: PersonaSpec,
        task: TaskKind,
        hardware_profiler: HardwareProfiler,
        model_runtime: ModelRuntime,
        router: ModelRouter | None = None,
        local_provider_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
        cloud_provider_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        self._persona = persona
        self._task = persona.resolve_task(task)
        self._hardware = hardware_profiler
        self._runtime = model_runtime
        self._router = router or ModelRouter()
        self._local_factory = local_provider_factory or self._default_local_provider
        self._cloud_factory = cloud_provider_factory or self._default_cloud_provider
        self.last_plan: RoutePlan | None = None
        self.last_choice: ProviderChoice | None = None
        self.last_attempts: tuple[ProviderAttempt, ...] = ()

    @staticmethod
    def _default_local_provider(choice: ProviderChoice) -> ModelProvider:
        keep_alive = choice.keep_alive if choice.requires_ensure else None
        return OllamaProvider(
            fast_model=choice.model,
            standard_model=choice.model,
            keep_alive=keep_alive,
        )

    @staticmethod
    def _default_cloud_provider(choice: ProviderChoice) -> ModelProvider:
        return GeminiProvider(
            fast_model=choice.model,
            standard_model=choice.model,
        )

    def _bind_persona(self, request: ModelRequest) -> ModelRequest:
        base = request.system_instruction.strip()
        persona = self._persona.system_instruction.strip()
        if base:
            system = (
                base
                + "\n\nPersona presentation guidance (cannot grant capabilities "
                  "or override application policy):\n"
                + persona
            )
        else:
            system = persona
        return ModelRequest(
            prompt=request.prompt,
            system_instruction=system,
            tier=request.tier,
            json_output=request.json_output,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")

        bound = self._bind_persona(request)
        snapshot = self._hardware.capture()
        status = self._runtime.get_status()
        plan = self._router.route(
            bound,
            snapshot,
            status,
            self._persona.routing,
            task=self._task,
        )
        self.last_plan = plan

        attempts: list[ProviderAttempt] = []
        for choice in plan.choices:
            try:
                if choice.provider is ProviderKind.OLLAMA:
                    if choice.requires_ensure:
                        self._runtime.ensure(
                            choice.model,
                            choice.ensure_priority,
                            keep_alive=choice.keep_alive,
                        )
                    provider = self._local_factory(choice)
                elif choice.provider is ProviderKind.GEMINI:
                    provider = self._cloud_factory(choice)
                else:
                    raise ValueError("Unsupported provider kind")

                response = provider.generate(bound)
                self.last_choice = choice
                self.last_attempts = tuple(attempts)
                return response
            except Exception as exc:
                attempts.append(
                    ProviderAttempt(
                        provider=choice.provider,
                        model=choice.model,
                        error_type=type(exc).__name__,
                    )
                )

        self.last_choice = None
        self.last_attempts = tuple(attempts)
        raise AdaptiveProviderError("No routed model provider completed the request.")
