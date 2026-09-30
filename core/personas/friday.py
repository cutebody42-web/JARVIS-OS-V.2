"""Realtime FRIDAY specialist persona."""

from core.model_provider import ModelTier
from core.model_router import LocalModelCandidate, PersonaRoutingProfile, TaskKind
from core.personas.persona_spec import PersonaSpec


FRIDAY = PersonaSpec(
    name="friday",
    display_name="F.R.I.D.A.Y.",
    system_instruction=(
        "You are the realtime operations mode behind TABY. Prefer short, "
        "latency-sensitive answers and bounded operational planning. You may "
        "propose only persona-allowed tools; owner policy remains authoritative. "
        "Do not claim completion without trusted action evidence."
    ),
    routing=PersonaRoutingProfile(
        name="friday",
        local_candidates=(
            LocalModelCandidate(
                model="qwen3.5:4b",
                min_available_ram_gb=4.0,
                priority=10,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
            ),
            LocalModelCandidate(
                model="qwen3:1.7b",
                min_available_ram_gb=2.0,
                priority=9,
                keep_alive=60,
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
