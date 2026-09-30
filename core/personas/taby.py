"""TABY: visible owner-facing identity."""

from core.model_provider import ModelTier
from core.model_router import (
    LocalModelCandidate,
    PersonaRoutingProfile,
    TaskKind,
)
from core.personas.persona_spec import PersonaSpec


TABY = PersonaSpec(
    name="taby",
    display_name="TABY",
    system_instruction=(
        "You are TABY, the single visible owner-facing identity of NEXUS. "
        "Be natural, concise and context-aware. Coordinate tasks without claiming "
        "actions succeeded unless verified by the action layer. Never bypass owner "
        "policy, approvals, or capability boundaries."
    ),
    routing=PersonaRoutingProfile(
        name="taby",
        local_candidates=(
            LocalModelCandidate(
                model="qwen3.5:4b",
                min_available_ram_gb=4.5,
                priority=10,
                keep_alive=300,
                tiers=frozenset({ModelTier.FAST, ModelTier.STANDARD}),
                tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
            ),
            LocalModelCandidate(
                model="qwen3.5:2b",
                min_available_ram_gb=3.0,
                priority=8,
                keep_alive=120,
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
