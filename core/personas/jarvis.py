"""JARVIS: engineering, planning and coding persona."""

from core.model_provider import ModelTier
from core.model_router import (
    LocalModelCandidate,
    PersonaRoutingProfile,
    TaskKind,
)
from core.personas.persona_spec import PersonaSpec


JARVIS = PersonaSpec(
    name="jarvis",
    display_name="JARVIS",
    system_instruction=(
        "You are the JARVIS strategic engineering mode under TABY/NEXUS. "
        "Prioritize careful planning, coding, systems reasoning and explicit "
        "uncertainty. Proposed actions remain subject to persona restrictions, "
        "OwnerPolicy and evidence-bearing execution."
    ),
    routing=PersonaRoutingProfile(
        name="jarvis",
        local_candidates=(
            LocalModelCandidate(
                model="qwen2.5-coder:7b",
                min_available_ram_gb=6.5,
                priority=6,
                keep_alive=300,
                tiers=frozenset({ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.CODING}),
            ),
            LocalModelCandidate(
                model="qwen2.5-coder:3b",
                min_available_ram_gb=3.0,
                priority=5,
                keep_alive=120,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
            ),
            LocalModelCandidate(
                model="qwen3.5:2b",
                min_available_ram_gb=3.0,
                priority=4,
                keep_alive=60,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.GENERAL}),
            ),
        ),
        cloud_fast_model="gemini-2.5-flash-lite",
        cloud_standard_model="gemini-2.5-flash",
    ),
    # Phase-1 capabilities only. Future broker capabilities can be added here
    # after OwnerPolicy admits them; this list never grants authority itself.
    tool_allowlist=frozenset({"system_time", "web_search", "weather_report"}),
    default_task=TaskKind.CODING,
    allowed_tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
    context_namespace="persona:jarvis",
)
