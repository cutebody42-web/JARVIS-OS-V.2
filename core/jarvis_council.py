"""Hidden multi-model local council coordinated by JARVIS Core 1B.

JARVIS Core is the coordinator. Specialist Ollama models are implementation
details and never become separate user-visible identities. The council is
adaptive: it runs parallel specialists only when local memory headroom allows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
import threading

from core.hardware_profile import HardwareProfiler, HardwareSnapshot
from core.model_provider import ModelRequest, ModelTier
from core.model_router import TaskKind
from core.model_runtime import ModelRuntime
from core.providers.ollama import OllamaProvider


CORE_MODEL = "jarvis-core-1b"
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_CORE_FIELDS = ("goal", "stated_constraints", "unknowns", "verify")
_CORE_LABEL = re.compile(
    r"(?im)^\s*(?:\d+[.)]\s*)?(GOAL|STATED_CONSTRAINTS|UNKNOWNS|VERIFY):\s*(.*)$"
)
_NEGATIVE_EVIDENCE = re.compile(
    r"\b(?:do\s+not|don't|never|without|no|none|unknown)\b",
    re.IGNORECASE,
)

# A class has two patterns: what constitutes a generated *claim*, and what
# counts as positive owner evidence for that class. Meta words such as
# "owner goal" or "specialist note" therefore do not trigger filtering.
_EVIDENCE_SENSITIVE_CLAIMS = (
    (
        "calendar_or_schedule",
        re.compile(
            r"\b(?:(?:review|check|open|inspect|consult|look\s+at)\s+(?:your\s+|the\s+)?"
            r"(?:calendar|schedule)|(?:your|tomorrow(?:'s)?)\s+(?:calendar|schedule)|"
            r"calendar\s+events?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:calendar|schedule)\b", re.IGNORECASE),
    ),
    (
        "task_inventory",
        re.compile(
            r"\b(?:(?:review|plan|rank|prioritize|organize|draft|finish|complete|work\s+on)\s+"
            r"(?:your\s+)?(?:(?:core|current|existing|planned|relevant)\s+){0,2}"
            r"(?:tasks?|projects?|goals?|priorities)|(?:your|tomorrow(?:'s)?)\s+"
            r"(?:(?:core|current|existing|planned|relevant)\s+){0,2}"
            r"(?:tasks?|projects?|goals?|priorities|task\s+list|to-?do\s+list))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:task\s+list|to-?do\s+list|tasks?|projects?|goals?|priorities)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reminders",
        re.compile(
            r"\b(?:set|create|check|review|schedule)\s+(?:your\s+|the\s+)?reminders?\b",
            re.IGNORECASE,
        ),
        re.compile(r"\breminders?\b", re.IGNORECASE),
    ),
    (
        "personal_event",
        re.compile(
            r"\b(?:(?:your|tomorrow(?:'s)?)\s+(?:meeting|presentation|appointment|class|exam|"
            r"shift|interview|flight|deadline|trip)|(?:before|after)\s+(?:the|your)\s+"
            r"(?:meeting|presentation|appointment|class|exam|shift|interview|flight|deadline))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:events?|meeting|presentation|appointment|class|exam|shift|interview|"
            r"flight|deadline|trip|travel)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "artifact_or_resource",
        re.compile(
            r"\b(?:review|prepare|organize|draft|finish|check|rehearse|open|use|gather|ensure)\s+"
            r"(?:your\s+|the\s+|relevant\s+|available\s+){0,2}"
            r"(?:files?|materials?|documents?|notes?|slides?|workspace|resources?|session)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:files?|materials?|documents?|notes?|slides?|workspace|resources?|session)\b",
            re.IGNORECASE,
        ),
    ),
)

_COUNCIL_GROUNDING_RULES = (
    "Ground every personal or situational claim in evidence. In a council call, the current "
    "owner request is the only current-turn evidence. Mark missing personal context as UNKNOWN "
    "and never fill gaps by assumption; never invent it or imply meetings, presentations, "
    "appointments, classes, exams, work shifts, deadlines, travel, people, locations, device "
    "state, files, calendar events, task lists, existing tasks, projects, goals, priorities, "
    "sessions, materials, documents, notes, workspaces, resources, reminders, habits or "
    "preferences that were not stated. Respect every explicit numeric or time constraint and "
    "verify arithmetic before proposing a plan. Unknown context is not a reason to refuse: "
    "preserve the stated constraints and keep missing objects abstract or conditional."
)

_SYNTHESIS_GROUNDING_CONTRACT = (
    "Grounding contract for synthesis: hidden council notes are analysis, not evidence. "
    "A personal or situational fact may appear in the final answer only when it is supported "
    "by the current owner request or by separately supplied durable synchronized memory or "
    "recent owner messages. Otherwise keep it UNKNOWN or phrase the advice conditionally. "
    "Never convert a plausible scenario into a claimed meeting, presentation, appointment, "
    "class, exam, shift, deadline, trip, person, location, device state, file, task list, "
    "existing task, project, goal, priority, session, material, document, note, workspace, "
    "resource, calendar item or reminder. If the owner supplied no object to work on, refer "
    "abstractly to a chosen focus instead of inventing one. Respect explicit numeric and time "
    "budgets; verify arithmetic and never allocate more time than the owner made available."
)


def _brief_value(value) -> str:
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    if isinstance(value, (list, tuple)):
        items = [" ".join(str(item).split()).strip() for item in value]
        return "; ".join(item for item in items if item)
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _parse_core_brief(raw_text: str) -> dict[str, str]:
    """Best-effort parser for tiny-model output; policy is enforced after parsing."""
    text = raw_text.strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None

    fields: dict[str, str] = {}
    if isinstance(value, dict):
        for key in _CORE_FIELDS:
            normalized = _brief_value(value.get(key))
            if normalized:
                fields[key] = normalized
        return fields

    label_map = {
        "GOAL": "goal",
        "STATED_CONSTRAINTS": "stated_constraints",
        "UNKNOWNS": "unknowns",
        "VERIFY": "verify",
    }
    for label, body in _CORE_LABEL.findall(text):
        normalized = _brief_value(body)
        if normalized:
            fields[label_map[label.upper()]] = normalized

    if not fields and text:
        fields["goal"] = _brief_value(text)
    return fields


def _sentence_has_positive_evidence(text: str, pattern: re.Pattern) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        if not pattern.search(sentence):
            continue
        # Conservative by design: explicit negation/prohibition does not establish
        # that the object exists in the owner's real context.
        if _NEGATIVE_EVIDENCE.search(sentence):
            continue
        return True
    return False


def unsupported_evidence_objects(owner_evidence: str, candidate: str) -> tuple[str, ...]:
    """Find generated evidence-sensitive claims unsupported by positive owner evidence."""
    unsupported = []
    for name, claim_pattern, evidence_pattern in _EVIDENCE_SENSITIVE_CLAIMS:
        if not claim_pattern.search(candidate):
            continue
        if _sentence_has_positive_evidence(owner_evidence, evidence_pattern):
            continue
        unsupported.append(name)
    return tuple(unsupported)


def _safe_reasoning_field(value: str, owner: str, fallback: str) -> str:
    if not value or unsupported_evidence_objects(owner, value):
        return fallback
    return value


def _normalize_core_brief(raw_text: str, owner_request: str) -> str:
    """Build a deterministic coordinator envelope around probabilistic Core reasoning.

    The exact owner request is the authoritative constraint ledger. Model-extracted
    constraints are intentionally not trusted as constraints, and model reasoning that
    introduces unsupported evidence-sensitive claims is replaced with a safe fallback.
    """
    fields = _parse_core_brief(raw_text)
    owner = " ".join(owner_request.split()).strip()

    goal = _safe_reasoning_field(
        fields.get("goal", ""),
        owner,
        "Interpret the authoritative owner request without adding facts.",
    )
    unknowns = fields.get("unknowns") or (
        "Anything not stated in the authoritative owner request is UNKNOWN."
    )
    verify = _safe_reasoning_field(
        fields.get("verify", ""),
        owner,
        "Verify every explicit owner constraint and add no unsupported personal context.",
    )
    constraints = "AUTHORITATIVE_OWNER_REQUEST: " + owner

    normalized = (
        f"GOAL: {goal}\n"
        f"STATED_CONSTRAINTS: {constraints}\n"
        f"UNKNOWNS: {unknowns}\n"
        f"VERIFY: {verify}"
    )
    return normalized[:5000]


@dataclass(frozen=True)
class CouncilNote:
    model: str
    text: str


@dataclass(frozen=True)
class CouncilResult:
    core_brief: str
    notes: tuple[CouncilNote, ...]
    models: tuple[str, ...]
    owner_request: str = ""
    rejected_notes: tuple[CouncilNote, ...] = ()


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
                prompt=(
                    "INTERNAL COORDINATOR TASK — do not answer the owner.\n"
                    "Read the OWNER REQUEST and return one JSON object with exactly four string "
                    "keys: goal, stated_constraints, unknowns, verify. Do not wrap it in markdown. "
                    "Copy explicit counts, time budgets, output-shape requirements and prohibitions "
                    "into stated_constraints. Put missing personal/situational facts in unknowns. "
                    "Never greet, apologize, refuse, ask a question, offer more help, or answer "
                    "the owner directly. Missing context is never a reason to refuse.\n\n"
                    "OWNER REQUEST:\n" + message
                ),
                system_instruction=(
                    "You are the hidden JARVIS Core coordinator. Produce compact internal routing "
                    "reasoning for specialist models. Identify the owner's goal, key constraints, "
                    "uncertainty, and what the final answer must verify. "
                    + _COUNCIL_GROUNDING_RULES + " "
                    "Explicit constraints from the owner belong in the JSON and must not be "
                    "refused or omitted. Output only the requested JSON object. Do not mention "
                    "model names."
                ),
                tier=ModelTier.FAST,
                json_output=True,
            )
        )
        return _normalize_core_brief(response.text, message)

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
                    "If the request does not name the thing being prepared or worked on, keep "
                    "that thing abstract as a chosen focus. Do not replace missing context with "
                    "a calendar, tasks, reminders, a session, materials, documents, notes, a "
                    "workspace or resources. Do not introduce yourself, do not address the owner, "
                    "and do not claim actions occurred."
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
        clean = message.strip()
        try:
            snapshot = self.profiler.capture()
            core_brief = self._core_brief(clean, task)
        except Exception:
            return None

        candidates = self._expert_candidates(task, snapshot)
        manual = self.manual_model
        experts_allowed = snapshot.system_pressure < 0.88 and snapshot.available_ram_gb >= 3.5
        if experts_allowed and manual and manual not in {CORE_MODEL, *candidates}:
            candidates.insert(0, manual)

        selected = tuple(dict.fromkeys(candidates))[: self.max_parallel_experts]
        if not selected:
            return CouncilResult(core_brief, (), (CORE_MODEL,), owner_request=clean)

        raw_notes: list[CouncilNote] = []
        with ThreadPoolExecutor(
            max_workers=len(selected),
            thread_name_prefix="jarvis-expert",
        ) as pool:
            futures = {
                pool.submit(self._consult_one, model, clean, task, core_brief): model
                for model in selected
            }
            for future in as_completed(futures):
                try:
                    note = future.result()
                except Exception:
                    continue
                if note.text:
                    raw_notes.append(note)

        raw_notes.sort(key=lambda item: selected.index(item.model))
        rejected = tuple(
            note for note in raw_notes
            if unsupported_evidence_objects(clean, note.text)
        )
        return CouncilResult(
            core_brief,
            tuple(raw_notes),
            (CORE_MODEL, *tuple(note.model for note in raw_notes)),
            owner_request=clean,
            rejected_notes=rejected,
        )

    @staticmethod
    def context(result: CouncilResult | None) -> str:
        if result is None:
            return ""
        parts = [
            _SYNTHESIS_GROUNDING_CONTRACT,
            "Hidden JARVIS Core routing brief:\n" + result.core_brief,
        ]
        rejected = set(result.rejected_notes)
        safe_notes = tuple(note for note in result.notes if note not in rejected)
        if safe_notes:
            parts.append(
                "Hidden local specialist notes (grounding-filtered):\n"
                + "\n\n".join(note.text for note in safe_notes)
            )
        if result.rejected_notes:
            parts.append(
                "Grounding filter: one or more hidden specialist notes were omitted because "
                "they introduced evidence-sensitive claims absent from positive owner evidence."
            )
        return "\n\n".join(parts)
