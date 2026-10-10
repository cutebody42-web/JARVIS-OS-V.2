"""Adaptive provider composition for a trusted PersonaSpec.

Implements the existing ModelProvider protocol so Planner/Executor can adopt
hardware-aware routing without importing provider SDK details.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import threading
from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import (
    CloudDisclosureError,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelTier,
)
from core.model_router import ModelRouter, ProviderChoice, ProviderKind, RoutePlan, TaskKind
from core.model_runtime import ModelRuntime, ModelRuntimeError, RuntimeStatus
from core.personas.persona_spec import PersonaSpec


class RoutedProviderError(RuntimeError):
    pass


_CONVERSATIONAL_EVIDENCE_CONTRACT = (
    "For conversational responses, personal or situational claims must be grounded in the "
    "current owner request, the current-session conversation, or explicitly supplied durable "
    "synchronized continuity. Missing context stays UNKNOWN or conditional. Do not invent "
    "unstated meetings, events, calendar entries, task lists, existing tasks, projects, goals, "
    "priorities, people, locations, files, sessions, materials, documents, notes, workspaces, "
    "resources, device state, habits or preferences. If the owner did not name the object to "
    "work on, keep it abstract (for example, a chosen focus) instead of supplying a plausible "
    "object. Respect explicit numeric and time constraints and verify arithmetic before "
    "presenting a plan."
)


@dataclass(frozen=True)
class RouteAttempt:
    provider: ProviderKind
    model: str
    reason: str
    outcome: str


ProviderFactory = Callable[[ProviderChoice], ModelProvider]


class RoutedModelProvider:
    def __init__(
        self,
        persona: PersonaSpec,
        *,
        task: TaskKind | None = None,
        profiler: HardwareProfiler | None = None,
        runtime: ModelRuntime | None = None,
        router: ModelRouter | None = None,
        ollama_factory: ProviderFactory | None = None,
        gemini_factory: ProviderFactory | None = None,
        ollama_base_url: str | None = None,
        allow_cloud: bool = True,
    ):
        if not isinstance(persona, PersonaSpec):
            raise TypeError("persona must be PersonaSpec")
        self.persona = persona
        self.task = persona.validate_task(task)
        self._ollama_base_url = (
            ollama_base_url
            or os.environ.get("NEXUS_OLLAMA_URL")
            or os.environ.get("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        self.profiler = profiler or HardwareProfiler(ollama_base_url=self._ollama_base_url)
        self.runtime = runtime or ModelRuntime(ollama_base_url=self._ollama_base_url)
        self.router = router or ModelRouter()
        self._ollama_factory = ollama_factory or self._default_ollama_factory
        self._gemini_factory = gemini_factory or self._default_gemini_factory
        self.allow_cloud = bool(allow_cloud)
        self._lock = threading.Lock()
        self._last_attempts: tuple[RouteAttempt, ...] = ()

    @property
    def last_attempts(self) -> tuple[RouteAttempt, ...]:
        with self._lock:
            return self._last_attempts

    def _record_attempts(self, attempts: list[RouteAttempt]) -> None:
        with self._lock:
            self._last_attempts = tuple(attempts)

    def _default_ollama_factory(self, choice: ProviderChoice) -> ModelProvider:
        from core.providers.ollama import OllamaProvider
        return OllamaProvider(
            choice.model,
            base_url=self._ollama_base_url,
            keep_alive=choice.keep_alive,
        )

    @staticmethod
    def _default_gemini_factory(choice: ProviderChoice) -> ModelProvider:
        from core.providers.gemini import GeminiProvider
        return GeminiProvider(fast_model=choice.model, standard_model=choice.model)

    def _with_persona_instruction(self, request: ModelRequest) -> ModelRequest:
        application_parts = []
        application = request.system_instruction.strip()
        if application:
            application_parts.append(application)
        if not request.json_output:
            application_parts.append(_CONVERSATIONAL_EVIDENCE_CONTRACT)
        application_contract = "\n\n".join(application_parts)
        persona = self.persona.system_instruction.strip()
        combined = (
            "APPLICATION CONTRACT (must be preserved):\n"
            + application_contract
            + "\n\nTRUSTED PERSONA PROFILE:\n"
            + persona
            if application_contract
            else persona
        )
        # Preserve classified context until a concrete provider has been
        # selected.  Reconstructing only the legacy fields here would silently
        # discard local memory as personas are applied.
        return replace(request, system_instruction=combined)

    @staticmethod
    def _request_for_provider(
        request: ModelRequest,
        provider: ProviderKind,
    ) -> ModelRequest:
        """Apply the privacy boundary before a provider object sees a request."""
        if provider is ProviderKind.OLLAMA:
            return request.for_local_provider()
        if provider is ProviderKind.GEMINI:
            return request.for_cloud_provider()
        raise RoutedProviderError("Unsupported routed provider")

    def _cloud_only_plan(self, request: ModelRequest) -> RoutePlan:
        if not self.allow_cloud:
            raise RoutedProviderError(
                "Local hardware state is unavailable and cloud routing is disabled."
            )
        model = (
            self.persona.routing.cloud_fast_model
            if request.tier is ModelTier.FAST
            else self.persona.routing.cloud_standard_model
        )
        return RoutePlan(
            ProviderChoice(
                provider=ProviderKind.GEMINI,
                model=model,
                reason="hardware_snapshot_unavailable",
            )
        )

    def _plan(self, request: ModelRequest) -> RoutePlan:
        try:
            hw = self.profiler.capture()
        except Exception:
            return self._cloud_only_plan(request)

        try:
            runtime_status = self.runtime.get_status()
        except ModelRuntimeError:
            runtime_status = RuntimeStatus(
                models={},
                warnings=("runtime_status_unavailable",),
            )

        return self.router.route(
            request,
            hw,
            runtime_status,
            self.persona.routing,
            task=self.task,
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")

        routed_request = self._with_persona_instruction(request)
        plan = self._plan(routed_request)
        attempts: list[RouteAttempt] = []

        for choice in plan.choices:
            if choice.provider is ProviderKind.GEMINI and not self.allow_cloud:
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        choice.reason,
                        "skipped:cloud_disabled",
                    )
                )
                continue
            try:
                if choice.provider is ProviderKind.OLLAMA:
                    if choice.requires_ensure:
                        self.runtime.ensure(
                            choice.model,
                            choice.ensure_priority,
                            keep_alive=choice.keep_alive,
                        )
                    provider = self._ollama_factory(choice)
                elif choice.provider is ProviderKind.GEMINI:
                    provider = self._gemini_factory(choice)
                else:
                    raise RoutedProviderError("Unsupported routed provider")

                # The selected provider receives a fresh, fully materialized
                # request.  A cloud provider never receives local-only context
                # in either ``prompt`` or the request's ``context`` field.
                provider_request = self._request_for_provider(
                    routed_request,
                    choice.provider,
                )
                response = provider.generate(provider_request)
                if not isinstance(response, ModelResponse):
                    raise RoutedProviderError("Provider returned an invalid response")
                attempts.append(RouteAttempt(choice.provider, choice.model, choice.reason, "succeeded"))
                self._record_attempts(attempts)
                return response
            except CloudDisclosureError:
                # This is a policy decision, not a transient provider failure.
                # Keep the audit value content-free and continue so a later
                # local route can still handle the unmodified request.
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        choice.reason,
                        "blocked:sensitive_content",
                    )
                )
            except Exception as exc:
                attempts.append(
                    RouteAttempt(
                        choice.provider,
                        choice.model,
                        choice.reason,
                        f"failed:{type(exc).__name__}",
                    )
                )

        self._record_attempts(attempts)
        if not self.allow_cloud:
            raise RoutedProviderError(
                "No local JARVIS Brain route completed the request; cloud routing is disabled."
            )
        raise RoutedProviderError("No routed model provider completed the request.")
