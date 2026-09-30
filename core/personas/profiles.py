"""Built-in TABY, FRIDAY and JARVIS routing/persona profiles."""

from __future__ import annotations

from core.model_provider import ModelTier
from core.model_router import LocalModelCandidate, PersonaRoutingProfile, TaskKind

from .persona_spec import PersonaRegistry, PersonaSpec


_FAST_LOCAL = LocalModelCandidate(
    model="qwen3.5:4b",
    min_available_ram_gb=3.0,
    priority=10,
    keep_alive=300,
    tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
    tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
)

_SMALL_LOCAL = LocalModelCandidate(
    model="qwen3:1.7b",
    min_available_ram_gb=1.5,
    priority=8,
    keep_alive=60,
    tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
    tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME, TaskKind.CODING}),
)

_CODER_LOCAL = LocalModelCandidate(
    model="qwen2.5-coder:7b",
    min_available_ram_gb=6.0,
    priority=5,
    keep_alive=300,
    tiers=frozenset({ModelTier.STANDARD}),
    tasks=frozenset({TaskKind.CODING}),
)


TABY = PersonaSpec(
    name="taby",
    system_instruction=(
        "You are TABY, the single visible assistant identity. Be clear, calm and "
        "helpful. Delegate internally when useful, but never claim capabilities "
        "that application policy has not granted."
    ),
    routing=PersonaRoutingProfile(
        name="taby",
        local_candidates=(_FAST_LOCAL, _SMALL_LOCAL, _CODER_LOCAL),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search", "weather_report"}),
    default_task=TaskKind.GENERAL,
    allowed_tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME, TaskKind.CODING}),
    context_namespace="persona:taby",
)

FRIDAY = PersonaSpec(
    name="friday",
    system_instruction=(
        "You are the FRIDAY operational mode inside TABY: concise, responsive and "
        "focused on real-time assistance. Application policy and receipts remain "
        "authoritative over all actions."
    ),
    routing=PersonaRoutingProfile(
        name="friday",
        local_candidates=(_FAST_LOCAL, _SMALL_LOCAL),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search", "weather_report"}),
    default_task=TaskKind.REALTIME,
    allowed_tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
    context_namespace="persona:friday",
)

JARVIS = PersonaSpec(
    name="jarvis",
    system_instruction=(
        "You are the JARVIS strategic mode inside TABY: precise, analytical and "
        "engineering-oriented. Propose only bounded work; OwnerPolicy decides "
        "whether any action is authorized."
    ),
    routing=PersonaRoutingProfile(
        name="jarvis",
        local_candidates=(_CODER_LOCAL, _SMALL_LOCAL),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search"}),
    default_task=TaskKind.CODING,
    allowed_tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
    context_namespace="persona:jarvis",
)

BUILTIN_PERSONAS = PersonaRegistry({
    TABY.name: TABY,
    FRIDAY.name: FRIDAY,
    JARVIS.name: JARVIS,
})
