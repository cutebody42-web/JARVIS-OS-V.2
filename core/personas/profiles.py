"""Built-in trusted persona profiles.

All profiles are internal cognitive lanes of one user-visible identity: JARVIS.
They share one memory namespace and one authority boundary.
"""

from __future__ import annotations

from core.model_provider import ModelTier
from core.model_router import LocalModelCandidate, PersonaRoutingProfile, TaskKind
from core.personas.persona_spec import PersonaSpec


_SAFE_READ_TOOLS = frozenset({"system_time", "web_search", "weather_report"})


def _candidate(model, ram, priority, keep_alive, tiers, tasks):
    return LocalModelCandidate(
        model=model,
        min_available_ram_gb=ram,
        priority=priority,
        keep_alive=keep_alive,
        tiers=frozenset(tiers),
        tasks=frozenset(tasks),
    )


TABY = PersonaSpec(
    name="taby",
    system_instruction=(
        "You are JARVIS. This is the general-presence cognitive lane. Be clear, practical "
        "and continuous across tasks. Never present internal model/provider/persona names "
        "to the owner. Preserve one identity and one memory across devices. Never claim "
        "authority you do not have and never bypass application policy."
    ),
    routing=PersonaRoutingProfile(
        name="taby",
        local_candidates=(
            _candidate(
                "qwen3.5:4b", 4.0, 10, 300,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.GENERAL, TaskKind.REALTIME},
            ),
            _candidate(
                "qwen3:1.7b", 2.0, 7, 60,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.GENERAL, TaskKind.REALTIME},
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=_SAFE_READ_TOOLS,
    default_task=TaskKind.GENERAL,
    allowed_tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
    context_namespace="jarvis",
)


FRIDAY = PersonaSpec(
    name="friday",
    system_instruction=(
        "You are JARVIS. This is the realtime cognitive lane. Optimize for low latency, "
        "concise realtime assistance, interruption recovery and operational clarity. "
        "Never present this lane or the underlying model as a separate assistant. "
        "Never expand tool authority or infer owner approval."
    ),
    routing=PersonaRoutingProfile(
        name="friday",
        local_candidates=(
            _candidate(
                "qwen3.5:4b", 4.0, 10, 300,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.REALTIME, TaskKind.GENERAL},
            ),
            _candidate(
                "qwen3:1.7b", 2.0, 8, 60,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.REALTIME, TaskKind.GENERAL},
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=_SAFE_READ_TOOLS,
    default_task=TaskKind.REALTIME,
    allowed_tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
    context_namespace="jarvis",
)


JARVIS = PersonaSpec(
    name="jarvis",
    system_instruction=(
        "You are JARVIS. This is the engineering cognitive lane. Prioritize structured "
        "reasoning, engineering accuracy, planning and explicit uncertainty. Never present "
        "the underlying model/provider as a separate assistant. Tool execution remains "
        "bounded by application policy and owner consent."
    ),
    routing=PersonaRoutingProfile(
        name="jarvis",
        local_candidates=(
            _candidate(
                "qwen2.5-coder:7b", 6.0, 6, 300,
                {ModelTier.STANDARD},
                {TaskKind.CODING},
            ),
            _candidate(
                "qwen3.5:4b", 4.0, 7, 180,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.GENERAL, TaskKind.CODING},
            ),
            _candidate(
                "qwen3:1.7b", 2.0, 4, 60,
                {ModelTier.FAST, ModelTier.STANDARD},
                {TaskKind.GENERAL, TaskKind.CODING},
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=_SAFE_READ_TOOLS,
    default_task=TaskKind.CODING,
    allowed_tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
    context_namespace="jarvis",
)


BUILTIN_PERSONAS = {persona.name: persona for persona in (TABY, FRIDAY, JARVIS)}


def get_persona(name: str) -> PersonaSpec:
    try:
        return BUILTIN_PERSONAS[name.strip().casefold()]
    except (AttributeError, KeyError):
        raise KeyError(f"Unknown persona: {name!r}") from None
