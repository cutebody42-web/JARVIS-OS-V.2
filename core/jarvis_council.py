"""Hidden multi-model local council coordinated by JARVIS Core 1B.

JARVIS Core is the coordinator. Specialist Ollama models are implementation
details and never become separate user-visible identities. The council is
adaptive: it runs parallel specialists only when local memory headroom allows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import threading

from core.hardware_profile import HardwareProfiler, HardwareSnapshot
from core.model_provider import ModelRequest, ModelTier
from core.model_router import TaskKind
from core.model_runtime import ModelRuntime
from core.providers.ollama import OllamaProvider


CORE_MODEL = "jarvis-core-1b"
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")

_COUNCIL_GROUNDING_RULES = (
    "Ground every personal or situational claim in evidence. In a council call, the current "
    "owner request is the only current-turn evidence. Mark missing personal context as UNKNOWN "
    "and never fill gaps by assumption; never invent it or imply meetings, presentations, "
    "appointments, classes, exams, work shifts, deadlines, travel, people, locations, device "
    "state, files, calendar events, task lists, existing tasks, projects, goals, priorities, "
    "habits or preferences that were not stated. Respect every explicit numeric or time "
    "constraint and verify arithmetic before proposing a plan. When a useful plan depends on "
    "missing context, use a conditional or neutral placeholder instead."
)

_SYNTHESIS_GROUNDING_CONTRACT = (
    "Grounding contract for synthesis: hidden council notes are analysis, not evidence. "
    "A personal or situational fact may appear in the final answer only when it is supported "
    "by the current owner request or by separately supplied durable synchronized memory or "
    "recent owner messages. Otherwise keep it UNKNOWN or phrase the advice conditionally. "
    "Never convert a plausible scenario into a claimed meeting, presentation, appointment, "
    "class, exam, shift, deadline, trip, person, location, device state, file, task list, "
    "existing task, project, goal or priority. Respect explicit numeric and time budgets; "
    "verify arithmetic and never allocate more time than the owner made available."
)


@dataclass(frozen=True)
class CouncilNote:
    model: str
    text: str


@dataclass(frozen=True)
class CouncilResult:
    core_brief: str
    notes: tuple[CouncilNote, ...]
    models: tuple[str, ...]


class JarvisCouncil:
    def __init__(
        self,
        *,
        ollama_base_url: str,
        profiler: HardwareProfiler,
        runtime: ModelRuntime,
        manual_model: str | None = None,
        max_parallel_experts: int = 2,
    ):
        if isinstance(max_parallel_experts, bool) or not isinstance(max_parallel_experts, int) or not 1 <= max_parallel_experts <= 4:
            raise ValueError("max_parallel_experts must be between 1 and 4")
        self.base_url = ollama_base_url
        self.profiler = profiler
        self.runtime = runtime
        self.max_parallel_experts = max_parallel_experts
        self._lock = threading.RLock()
        self._manual_model = None
        self.set_manual_model(manual_model)

    @property
    def manual_model(self) -> str | None:
        with self._lock:
            return self._manual_model

    def set_manual_model(self, model: str | None) -> None:
        if model is not None:
            if not isinstance(model, str):
                raise ValueError("manual model name is invalid")
            model = model.strip()
            if not _MODEL_NAME.fullmatch(model):
                raise ValueError("manual model name is invalid")
        with self._lock:
            self._manual_model = model or None

    @staticmethod
    def _expert_candidates(task: TaskKind, snapshot: HardwareSnapshot) -> list[str]:
        # The 1B coordinator stays loaded; specialists are added only when the
        # current machine has enough free system memory to keep them useful.
        if snapshot.system_pressure >= 0.88 or snapshot.available_ram_gb < 3.5:
            return []

        if task is TaskKind.CODING:
            if snapshot.available_ram_gb >= 11.0 and snapshot.total_ram_gb >= 16:
                return ["jarvis-brain-engineering", "jarvis-brain-fast"]
            if snapshot.available_ram_gb >= 7.0 and snapshot.total_ram_gb >= 12:
                return ["jarvis-brain-engineering"]
            if snapshot.available_ram_gb >= 5.0:
                return ["jarvis-brain-fast"]
            return ["jarvis-brain-lite"]

        if task is TaskKind.REALTIME:
            if snapshot.available_ram_gb >= 8.0:
                return ["jarvis-brain-fast", "jarvis-brain-lite"]
            if snapshot.available_ram_gb >= 5.0:
                return ["jarvis-brain-fast"]
            return ["jarvis-brain-lite"]

        if snapshot.available_ram_gb >= 8.0:
            return ["jarvis-brain-fast", "jarvis-brain-lite"]
        if snapshot.available_ram_gb >= 5.0:
            return ["jarvis-brain-fast"]
        return ["jarvis-brain-lite"]

    def _provider(self, model: str, *, keep_alive: str | int | None = "5m") -> OllamaProvider:
        return OllamaProvider(
            model,
            base_url=self.base_url,
            keep_alive=keep_alive,
            timeout=60,
        )

    def _core_brief(self, message: str, task: TaskKind) -> str:
        self.runtime.ensure(CORE_MODEL, 100, keep_alive="10m")
        response = self._provider(CORE_MODEL, keep_alive="10m").generate(
            ModelRequest(
                prompt=message,
                system_instruction=(
                    "You are the hidden JARVIS Core coordinator. Produce a compact internal "
                    "routing brief for specialist models. Identify the owner's goal, key "
                    "constraints, uncertainty, and what the final answer must verify. "
                    + _COUNCIL_GROUNDING_RULES + " "
                    "Do not answer the owner directly. Do not mention model names."
                ),
                tier=ModelTier.FAST,
            )
        )
        return response.text.strip()[:5000]

    def _consult_one(
        self,
        model: str,
        message: str,
        task: TaskKind,
        core_brief: str,
    ) -> CouncilNote:
        self.runtime.ensure(model, 50, keep_alive="5m")
        response = self._provider(model).generate(
            ModelRequest(
                prompt=(
                    "Owner request:\n" + message +
                    "\n\nJARVIS Core routing brief:\n" + core_brief
                ),
                system_instruction=(
                    "You are a hidden local JARVIS specialist. Give concise, factual internal "
                    "analysis that another JARVIS layer will synthesize. "
                    + _COUNCIL_GROUNDING_RULES + " "
                    "Do not introduce yourself, do not address the owner, and do not claim "
                    "actions occurred."
                ),
                tier=ModelTier.STANDARD,
            )
        )
        return CouncilNote(model, response.text.strip()[:7000])

    def consult(self, message: str, task: TaskKind) -> CouncilResult | None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty")
        if not isinstance(task, TaskKind):
            raise ValueError("task must be a supported TaskKind")
        try:
            snapshot = self.profiler.capture()
            core_brief = self._core_brief(message.strip(), task)
        except Exception:
            # Council enrichment must never make the single-model route unusable.
            return None

        candidates = self._expert_candidates(task, snapshot)
        manual = self.manual_model
        # Manual selection adds an expert but must obey the same memory and
        # pressure gate as automatic selection. It never displaces the core.
        experts_allowed = snapshot.system_pressure < 0.88 and snapshot.available_ram_gb >= 3.5
        if experts_allowed and manual and manual not in {CORE_MODEL, *candidates}:
            candidates.insert(0, manual)

        # Preserve order while deduplicating, then cap concurrent specialists.
        selected = tuple(dict.fromkeys(candidates))[: self.max_parallel_experts]
        if not selected:
            return CouncilResult(core_brief, (), (CORE_MODEL,))

        notes: list[CouncilNote] = []
        with ThreadPoolExecutor(
            max_workers=len(selected),
            thread_name_prefix="jarvis-expert",
        ) as pool:
            futures = {
                pool.submit(self._consult_one, model, message.strip(), task, core_brief): model
                for model in selected
            }
            for future in as_completed(futures):
                try:
                    note = future.result()
                except Exception:
                    continue
                if note.text:
                    notes.append(note)

        notes.sort(key=lambda item: selected.index(item.model))
        return CouncilResult(
            core_brief,
            tuple(notes),
            (CORE_MODEL, *tuple(note.model for note in notes)),
        )

    @staticmethod
    def context(result: CouncilResult | None) -> str:
        if result is None:
            return ""
        parts = [
            _SYNTHESIS_GROUNDING_CONTRACT,
            "Hidden JARVIS Core routing brief:\n" + result.core_brief,
        ]
        if result.notes:
            parts.append(
                "Hidden local specialist notes:\n"
                + "\n\n".join(note.text for note in result.notes)
            )
        return "\n\n".join(parts)
