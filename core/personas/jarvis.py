"""Engineering JARVIS specialist persona."""

from core.model_provider import ModelTier
from core.model_router import LocalModelCandidate, PersonaRoutingProfile, TaskKind
from core.personas.persona_spec import PersonaSpec


JARVIS = PersonaSpec(
    name="jarvis",
    display_name="JARVIS",
    system_instruction=(
        "You are the engineering and strategic reasoning mode behind TABY. "
        "Prefer explicit assumptions, structured plans, verification, and "
        "reversible changes. Model output is advisory; owner policy and action "
        "verification remain authoritative. Never invent tool success."
    ),
    routing=PersonaRoutingProfile(
        name="jarvis",
        local_candidates=(
            LocalModelCandidate(
                model="qwen2.5-coder:7b",
                min_available_ram_gb=6.0,
                priority=5,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
            ),
            LocalModelCandidate(
                model="qwen3:1.7b",
                min_available_ram_gb=2.0,
                priority=4,
                keep_alive=60,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    tool_allowlist=frozenset({"system_time", "web_search"}),
    default_task=TaskKind.CODING,
    allowed_tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
    context_namespace="persona:jarvis",
)
