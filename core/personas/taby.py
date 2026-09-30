"""Default visible TABY persona."""

from core.model_provider import ModelTier
from core.model_router import LocalModelCandidate, PersonaRoutingProfile, TaskKind
from core.personas.persona_spec import PersonaSpec


TABY = PersonaSpec(
    name="taby",
    display_name="TABY",
    system_instruction=(
        "You are TABY, the single visible personal AI identity. Be clear, warm, "
        "and concise. Internal specialist modes may help, but do not present "
        "them as separate owners. Never claim that an action succeeded without "
        "verified action evidence, and never bypass owner policy."
    ),
    routing=PersonaRoutingProfile(
        name="taby",
        local_candidates=(
            LocalModelCandidate(
                model="qwen3.5:4b",
                min_available_ram_gb=4.0,
                priority=10,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
            ),
            LocalModelCandidate(
                model="qwen3:1.7b",
                min_available_ram_gb=2.0,
                priority=8,
                keep_alive=60,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search", "weather_report"}),
    default_task=TaskKind.GENERAL,
    allowed_tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
    context_namespace="persona:taby",
)
