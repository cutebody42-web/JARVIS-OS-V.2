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
from uuid import uuid4

from core.hardware_profile import HardwareProfiler
from core.jarvis_memory import JarvisMemory
from core.model_provider import (
    ContextClassification,
    ModelContext,
    ModelProvider,
    ModelRequest,
    ModelTier,
)
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


class BrainIntent(str, Enum):
    CHAT = "chat"
    ACTION = "action"


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
        _candidate(
            "jarvis-core-1b", 1.5, 3, 300,
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
        _candidate(
            "jarvis-core-1b", 1.5, 4, 300,
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
        _candidate(
            "jarvis-core-1b", 1.5, 2, 300,
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

_ACTION_RE = re.compile(
    r"^\s*(?:jarvis[,:]?\s+)?"
    r"(?:(?:please|(?:can|could|would|will)\s+you)\s+)*"
    r"(?:open|close|launch|start|stop|pause|resume|mute|unmute|"
    r"set (?:the )?(?:volume|brightness)|search (?:the )?web|"
    r"look up|what(?:'s| is) the (?:time|weather)|weather (?:in|for)|"
    r"remind me|set (?:a )?timer)\b",
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


def classify_intent(text: str) -> BrainIntent:
    """Conservative action routing.

    Ambiguous text stays conversational. Only explicit, admitted operational
    wording enters AgentExecutor/OwnerKernel.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be non-empty")
    return BrainIntent.ACTION if _ACTION_RE.match(text.strip()) else BrainIntent.CHAT


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
        memory: JarvisMemory | None = None,
        session_id: str | None = None,
        council=None,
    ):
        self.policy = policy or BrainPolicy()
        self._lock = threading.RLock()
        self._turn_lock = threading.Lock()
        self._lane = BrainLane.GENERAL
        self._history: list[tuple[str, str]] = []
        self._history_limit = 16
        self._memory = memory
        self._council = council
        self._session_id = session_id or uuid4().hex
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

    @staticmethod
    def _lane_for_task(goal: str, task: TaskKind | None) -> BrainLane:
        if task is None:
            return classify_lane(goal)
        if task is TaskKind.CODING:
            return BrainLane.ENGINEERING
        if task is TaskKind.REALTIME:
            return BrainLane.REALTIME
        if task is TaskKind.GENERAL:
            return BrainLane.GENERAL
        raise ValueError("unsupported task kind")

    def _private_turn_context(self) -> tuple[ModelContext, ...]:
        """Snapshot durable and session context as local-only data.

        Classification is attached before routing.  In particular, enabling a
        cloud fallback cannot retroactively make either memory source eligible
        for disclosure.
        """
        with self._lock:
            history = tuple(self._history[-self._history_limit:])
            memory = self._memory

        context: list[ModelContext] = []
        if memory is not None:
            durable = memory.continuity_context(turn_limit=8)
            if durable:
                context.append(ModelContext(
                    durable,
                    label="durable synchronized continuity",
                ))

        if history:
            transcript = "\n".join(
                ("Owner" if role == "user" else JARVIS_IDENTITY) + ": " + content
                for role, content in history
            )
            context.append(ModelContext(
                transcript,
                label="current-session conversation",
            ))

        return tuple(context)

    def _remember_turn(self, user_text: str, response_text: str) -> None:
        with self._lock:
            self._history.extend((("user", user_text), ("assistant", response_text)))
            if len(self._history) > self._history_limit * 2:
                self._history = self._history[-self._history_limit * 2:]
            memory = self._memory
            session_id = self._session_id

        if memory is not None:
            handoff = memory.current_handoff() or {}
            project_id = handoff.get("project_id")
            if not isinstance(project_id, str) or not project_id:
                project_id = None
            memory.append_turn(session_id, "user", user_text, project_id=project_id)
            memory.append_turn(session_id, "assistant", response_text, project_id=project_id)


    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def memory(self) -> JarvisMemory | None:
        return self._memory

    def attach_memory(self, memory: JarvisMemory) -> None:
        if not isinstance(memory, JarvisMemory):
            raise TypeError("memory must be JarvisMemory")
        with self._lock:
            self._memory = memory

    def checkpoint_project(
        self,
        project_id: str,
        *,
        summary: str | None = None,
        phase: str | None = None,
        open_tasks: list[str] | None = None,
        decisions: list[str] | None = None,
        artifacts: list[str] | None = None,
    ):
        with self._lock:
            memory = self._memory
        if memory is None:
            raise RuntimeError("Durable JARVIS memory is not attached")
        return memory.checkpoint_project(
            project_id,
            summary=summary,
            phase=phase,
            open_tasks=open_tasks,
            decisions=decisions,
            artifacts=artifacts,
        )

    def write_handoff(
        self,
        *,
        project_id: str | None,
        summary: str,
        next_actions=(),
    ) -> str:
        with self._lock:
            memory = self._memory
        if memory is None:
            raise RuntimeError("Durable JARVIS memory is not attached")
        return memory.write_handoff(
            project_id=project_id,
            summary=summary,
            next_actions=tuple(next_actions),
            session_id=self._session_id,
        )

    def clear_session_history(self) -> None:
        """Clear transient conversation history without deleting durable memory."""
        with self._lock:
            self._history.clear()

    def respond(
        self,
        message: str,
        *,
        task: TaskKind | None = None,
        tier: ModelTier | None = None,
        cloud_shareable_context: str | None = None,
    ) -> str:
        """Generate a direct conversational response with no tool execution.

        Durable memory and prior turns are always local-only.  Supplementary
        context crosses a cloud route only when the caller supplies it through
        the explicitly named ``cloud_shareable_context`` parameter.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty text")
        if cloud_shareable_context is not None and (
            not isinstance(cloud_shareable_context, str)
            or not cloud_shareable_context.strip()
        ):
            raise ValueError("cloud_shareable_context must be non-empty text")
        clean = message.strip()
        lane = self._lane_for_task(clean, task)
        self._switch_lane(lane)

        if tier is None:
            tier = ModelTier.FAST if lane is BrainLane.REALTIME else ModelTier.STANDARD

        with self._lock:
            provider = self._runtime.provider

        context = list(self._private_turn_context())
        if cloud_shareable_context is not None:
            context.append(ModelContext(
                cloud_shareable_context.strip(),
                label="explicitly cloud-shareable context",
                classification=ContextClassification.CLOUD_SHAREABLE,
            ))
        council = self._council
        if council is not None:
            try:
                council_result = council.consult(clean, lane_task(lane))
                council_context = council.context(council_result)
            except Exception:
                council_context = ""
            if council_context:
                context.append(ModelContext(
                    council_context
                    + "\n\nSynthesize the best final JARVIS answer using these hidden notes. "
                    "Treat council notes as analysis, not evidence: discard any personal detail "
                    "that is unsupported by the owner request or durable synchronized memory. "
                    "Never reveal or name the hidden models unless the owner explicitly asks.",
                    label="local council notes",
                ))

        response = provider.generate(
            ModelRequest(
                prompt=clean,
                system_instruction=(
                    "Continue the same JARVIS identity and preserve context across devices. "
                    "Respond directly as JARVIS. This is a conversational cognition request, "
                    "not an action plan. Do not claim that any external action occurred. "
                    "If an action is required, explain what needs execution rather than fabricating it."
                ),
                tier=tier,
                json_output=False,
                context=tuple(context),
            )
        )
        text = response.text.strip()
        degraded = text.casefold().replace("\n", " ").strip()
        if degraded in {
            "i cannot fulfill your request.",
            "i cannot fulfill your request. the local brain is offline.",
        }:
            response = provider.generate(
                ModelRequest(
                    prompt=clean,
                    system_instruction=(
                        "Answer the owner's conversational request naturally as JARVIS. "
                        "The local model is online. Do not claim it is offline and do not "
                        "invent external actions or personal facts."
                    ),
                    tier=tier,
                    json_output=False,
                )
            )
            text = response.text.strip()
        if not text:
            raise RuntimeError("JARVIS Brain returned an empty response.")
        self._remember_turn(clean, text)
        return text

    def execute(
        self,
        goal: str,
        *,
        task: TaskKind | None = None,
        speak=None,
        cancel_flag=None,
    ) -> str:
        """Execute an explicit operational goal through AgentExecutor/OwnerKernel."""
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be non-empty text")
        clean = goal.strip()
        lane = self._lane_for_task(clean, task)
        self._switch_lane(lane)
        with self._lock:
            runtime = self._runtime
        result = runtime.execute(clean, speak=speak, cancel_flag=cancel_flag)
        self._remember_turn(clean, result)
        return result

    def handle(
        self,
        message: str,
        *,
        task: TaskKind | None = None,
        speak=None,
        cancel_flag=None,
    ) -> str:
        """Single public entry point: chat by default, act only on explicit intent.

        Desktop and paired companion requests share this turn lock so lane state
        and transient conversation history remain one coherent JARVIS session.
        """
        with self._turn_lock:
            intent = classify_intent(message)
            if intent is BrainIntent.ACTION:
                return self.execute(
                    message,
                    task=task,
                    speak=speak,
                    cancel_flag=cancel_flag,
                )
            response = self.respond(message, task=task)
            if speak:
                speak(response)
            return response

    ask = respond
