"""FRIDAY: low-latency realtime operations persona."""

from core.model_provider import ModelTier
from core.model_router import (
    LocalModelCandidate,
    PersonaRoutingProfile,
    TaskKind,
)
from core.personas.persona_spec import PersonaSpec


FRIDAY = PersonaSpec(
    name="friday",
    display_name="F.R.I.D.A.Y.",
    system_instruction=(
        "You are the F.R.I.D.A.Y. realtime operating mode under TABY/NEXUS. "
        "Prioritize low latency, concise responses, interruption safety, and quick "
        "recovery. Do not expand permissions or claim unverified actions."
    ),
    routing=PersonaRoutingProfile(
        name="friday",
        local_candidates=(
            LocalModelCandidate(
                model="qwen3.5:4b",
                min_available_ram_gb=4.5,
                priority=10,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
            ),
            LocalModelCandidate(
                model="qwen3.5:2b",
                min_available_ram_gb=3.0,
                priority=9,
                keep_alive=120,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search", "weather_report"}),
    default_task=TaskKind.REALTIME,
    allowed_tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
    context_namespace="persona:friday",
)
