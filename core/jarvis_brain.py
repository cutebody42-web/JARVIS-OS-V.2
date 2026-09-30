"""One user-visible JARVIS identity over multiple internal inference lanes.

The lane/model/provider are implementation details. Conversation identity,
memory namespace, authority and tool policy remain singular: JARVIS.

Underlying providers are still recorded in RouteAttempt for auditability. This
module does not falsify provider provenance; it only prevents provider/model
names from becoming separate assistant identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import threading
from typing import Callable

from core.hardware_profile import HardwareProfiler
from core.model_provider import ModelProvider, ModelTier
from core.model_router import (
    LocalModelCandidate,
    ModelRouter,
    PersonaRoutingProfile,
    ProviderChoice,
    TaskKind,
)
from core.model_runtime import ModelRuntime
from core.persona_runtime import PersonaAgentRuntime
from core.personas.persona_spec import PersonaSpec


JARVIS_IDENTITY = "JARVIS"
JARVIS_CONTEXT_NAMESPACE = "jarvis"


class BrainLane(str, Enum):
    GENERAL = "general"
    REALTIME = "realtime"
    ENGINEERING = "engineering"


@dataclass(frozen=True)
class BrainPolicy:
    """Owner-controlled cognition policy.

    Cloud is disabled by default: JARVIS remains local unless the owner opts in
    to cloud acceleration. This is a product/privacy choice, not an attempt to
    change any provider's own policies.
    """

    allow_cloud: bool = False


_COMMON_IDENTITY = (
    "You are JARVIS, one continuous personal AI identity. "
    "Never introduce yourself as an underlying model, vendor, route, lane or provider. "
    "Internal model changes do not create a new assistant. Preserve the same voice, "
    "memory continuity and owner relationship across every task. "
    "Prioritize the owner's explicit goals and preferences while remaining inside "
    "the application's authorization, consent and safety boundaries. "
    "If the owner explicitly asks which engine handled a request, answer truthfully."
)

_COMMON_TOOLS = frozenset({
    "system_time",
    "web_search",
    "weather_report",
})


def _candidate(model, ram, priority, keep_alive, tiers, tasks):
    return LocalModelCandidate(
        model=model,
        min_available_ram_gb=ram,
        priority=priority,
        keep_alive=keep_alive,
        tiers=frozenset(tiers),
        tasks=frozenset(tasks),
    )


def _profile(
    *,
    name: str,
    lane_instruction: str,
    candidates: tuple[LocalModelCandidate, ...],
    default_task: TaskKind,
    allowed_tasks: frozenset[TaskKind],
) -> PersonaSpec:
    return PersonaSpec(
        name=name,
        system_instruction=_COMMON_IDENTITY + " " + lane_instruction,
        routing=PersonaRoutingProfile(
            name=name,
            local_candidates=candidates,
            cloud_fast_model="gemini-2.5-flash-lite",
            cloud_standard_model="gemini-2.5-flash",
        ),
        tool_allowlist=_COMMON_TOOLS,
        default_task=default_task,
        allowed_tasks=allowed_tasks,
        context_namespace=JARVIS_CONTEXT_NAMESPACE,
    )


JARVIS_GENERAL = _profile(
    name="jarvis_general",
    lane_instruction=(
        "Handle everyday conversation, planning and mixed tasks with balanced depth."
    ),
    candidates=(
        _candidate(
            "jarvis-brain-fast", 4.0, 10, 300,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.GENERAL, TaskKind.REALTIME},
        ),
        _candidate(
            "jarvis-brain-lite", 2.0, 8, 60,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.GENERAL, TaskKind.REALTIME},
        ),
    ),
    default_task=TaskKind.GENERAL,
    allowed_tasks=frozenset({TaskKind.GENERAL, TaskKind.REALTIME}),
)

JARVIS_REALTIME = _profile(
    name="jarvis_realtime",
    lane_instruction=(
        "Optimize for low latency, concise realtime help, interruption recovery "
        "and clear operational feedback."
    ),
    candidates=(
        _candidate(
            "jarvis-brain-fast", 4.0, 10, 300,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.REALTIME, TaskKind.GENERAL},
        ),
        _candidate(
            "jarvis-brain-lite", 2.0, 9, 60,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.REALTIME, TaskKind.GENERAL},
        ),
    ),
    default_task=TaskKind.REALTIME,
    allowed_tasks=frozenset({TaskKind.REALTIME, TaskKind.GENERAL}),
)

JARVIS_ENGINEERING = _profile(
    name="jarvis_engineering",
    lane_instruction=(
        "Optimize for engineering, coding, debugging, architecture and rigorous planning. "
        "Prefer explicit assumptions and verifiable technical steps."
    ),
    candidates=(
        _candidate(
            "jarvis-brain-engineering", 6.0, 7, 300,
            {ModelTier.STANDARD},
            {TaskKind.CODING},
        ),
        _candidate(
            "jarvis-brain-fast", 4.0, 8, 180,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.CODING, TaskKind.GENERAL},
        ),
        _candidate(
            "jarvis-brain-lite", 2.0, 5, 60,
            {ModelTier.FAST, ModelTier.STANDARD},
            {TaskKind.CODING, TaskKind.GENERAL},
        ),
    ),
    default_task=TaskKind.CODING,
    allowed_tasks=frozenset({TaskKind.CODING, TaskKind.GENERAL}),
)


LANE_PROFILES = {
    BrainLane.GENERAL: JARVIS_GENERAL,
    BrainLane.REALTIME: JARVIS_REALTIME,
    BrainLane.ENGINEERING: JARVIS_ENGINEERING,
}


_CODE_RE = re.compile(
    r"\b(code|coding|python|javascript|typescript|rust|java|c\+\+|"
    r"bug|debug|traceback|exception|compile|build|test|pytest|github|git|"
    r"commit|branch|pull request|pr|repository|repo|api|class|function|"
    r"database|sql|architecture|refactor)\b",
    re.IGNORECASE,
)
_REALTIME_RE = re.compile(
    r"\b(open|close|mute|unmute|volume|brightness|launch|start|stop|"
    r"pause|resume|status|battery|time|weather|notify|remind|timer)\b",
    re.IGNORECASE,
)


def classify_lane(goal: str) -> BrainLane:
    """Deterministic, conservative lane choice.

    It never changes identity or authority; it only selects an inference profile.
    """
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be non-empty text")
    text = goal.strip()
    if _CODE_RE.search(text):
        return BrainLane.ENGINEERING
    if len(text.split()) <= 24 and _REALTIME_RE.search(text):
        return BrainLane.REALTIME
    return BrainLane.GENERAL


def lane_task(lane: BrainLane) -> TaskKind:
    if lane is BrainLane.ENGINEERING:
        return TaskKind.CODING
    if lane is BrainLane.REALTIME:
        return TaskKind.REALTIME
    return TaskKind.GENERAL


class JarvisBrain:
    """Single JARVIS runtime with hidden hot-switched cognition lanes."""

    def __init__(
        self,
        *,
        policy: BrainPolicy | None = None,
        awareness=None,
        owner_runtime=None,
        profiler: HardwareProfiler | None = None,
        model_runtime: ModelRuntime | None = None,
        router: ModelRouter | None = None,
        ollama_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
        gemini_factory: Callable[[ProviderChoice], ModelProvider] | None = None,
        ollama_base_url: str | None = None,
    ):
        self.policy = policy or BrainPolicy()
        self._lock = threading.RLock()
        self._lane = BrainLane.GENERAL
        self._runtime = PersonaAgentRuntime(
            JARVIS_GENERAL,
            task=TaskKind.GENERAL,
            awareness=awareness,
            owner_runtime=owner_runtime,
            profiler=profiler,
            model_runtime=model_runtime,
            router=router,
            ollama_factory=ollama_factory,
            gemini_factory=gemini_factory,
            ollama_base_url=ollama_base_url,
            allow_cloud=self.policy.allow_cloud,
        )

    @property
    def identity(self) -> str:
        return JARVIS_IDENTITY

    @property
    def context_namespace(self) -> str:
        return JARVIS_CONTEXT_NAMESPACE

    @property
    def lane(self) -> BrainLane:
        with self._lock:
            return self._lane

    @property
    def last_attempts(self):
        with self._lock:
            return self._runtime.provider.last_attempts

    def _switch_lane(self, lane: BrainLane) -> None:
        if not isinstance(lane, BrainLane):
            raise TypeError("lane must be BrainLane")
        with self._lock:
            if lane is self._lane:
                return
            profile = LANE_PROFILES[lane]
            self._runtime.switch_persona(profile, task=lane_task(lane))
            self._lane = lane

    def execute(
        self,
        goal: str,
        *,
        task: TaskKind | None = None,
        speak=None,
        cancel_flag=None,
    ) -> str:
        if task is None:
            lane = classify_lane(goal)
        elif task is TaskKind.CODING:
            lane = BrainLane.ENGINEERING
        elif task is TaskKind.REALTIME:
            lane = BrainLane.REALTIME
        elif task is TaskKind.GENERAL:
            lane = BrainLane.GENERAL
        else:
            raise ValueError("unsupported task kind")
        self._switch_lane(lane)
        with self._lock:
            runtime = self._runtime
        return runtime.execute(goal, speak=speak, cancel_flag=cancel_flag)

    ask = execute
