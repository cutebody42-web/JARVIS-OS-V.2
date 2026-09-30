"""Hidden multi-model local council coordinated by JARVIS Core 1B.

JARVIS Core is the coordinator. Specialist Ollama models are implementation
details and never become separate user-visible identities. The council is
adaptive: it runs parallel specialists only when local memory headroom allows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from typing import Iterable

from core.hardware_profile import HardwareProfiler, HardwareSnapshot
from core.model_provider import ModelRequest, ModelTier
from core.model_router import TaskKind
from core.model_runtime import ModelRuntime
from core.providers.ollama import OllamaProvider


CORE_MODEL = "jarvis-core-1b"


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
        if not isinstance(max_parallel_experts, int) or not 1 <= max_parallel_experts <= 4:
            raise ValueError("max_parallel_experts must be between 1 and 4")
        self.base_url = ollama_base_url
        self.profiler = profiler
        self.runtime = runtime
        self.max_parallel_experts = max_parallel_experts
        self._manual_model = manual_model
        self._lock = threading.RLock()

    @property
    def manual_model(self) -> str | None:
        with self._lock:
            return self._manual_model

    def set_manual_model(self, model: str | None) -> None:
        if model is not None:
            model = model.strip()
            if not model or len(model) > 128:
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
            if snapshot.available_ram_gb >= 7.0 and snapshot.total_ram_gb >= 12:
                return ["jarvis-brain-engineering", "jarvis-brain-fast"]
            return ["jarvis-brain-lite"]

        if task is TaskKind.REALTIME:
            if snapshot.available_ram_gb >= 5.0:
                return ["jarvis-brain-fast", "jarvis-brain-lite"]
            return ["jarvis-brain-lite"]

        if snapshot.available_ram_gb >= 5.0:
            return ["jarvis-brain-fast", "jarvis-brain-lite"]
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
                    "analysis that another JARVIS layer will synthesize. Do not introduce "
                    "yourself, do not address the owner, and do not claim actions occurred."
                ),
                tier=ModelTier.STANDARD,
            )
        )
        return CouncilNote(model, response.text.strip()[:7000])

    def consult(self, message: str, task: TaskKind) -> CouncilResult | None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be non-empty")
        try:
            snapshot = self.profiler.capture()
            core_brief = self._core_brief(message.strip(), task)
        except Exception:
            # Council enrichment must never make the single-model route unusable.
            return None

        candidates = self._expert_candidates(task, snapshot)
        manual = self.manual_model
        if manual and manual not in {CORE_MODEL, *candidates}:
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
            "Hidden JARVIS Core routing brief:\n" + result.core_brief,
        ]
        if result.notes:
            parts.append(
                "Hidden local specialist notes:\n"
                + "\n\n".join(note.text for note in result.notes)
            )
        return "\n\n".join(parts)
