#!/usr/bin/env python3
"""Verify one real conversational turn through the production local AI stack.

Requires an already-running Ollama with JARVIS Core and the fast expert. This
script does not download weights, mock providers, attach owner memory, execute
tools, or certify physical hardware. It does enforce one narrow, deterministic
grounding-quality contract for the production planning probe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.jarvis_council import CORE_MODEL

EXPERT_MODEL = "jarvis-brain-fast"
TIME_BUDGET_MINUTES = 30
PROMPT = (
    "I have thirty minutes to prepare for tomorrow. Return exactly three short numbered "
    "planning steps (1., 2., 3.) using only that information. Do not assume my calendar, "
    "events or preferences. If you assign times, they must total no more than thirty minutes. "
    "Keep the final answer under eighty words."
)

_UNSUPPORTED_ASSUMPTION_PATTERNS = (
    (
        "claimed_personal_event",
        re.compile(
            r"\b(?:your|tomorrow(?:'s)?)\s+(?:meeting|presentation|appointment|class|exam|"
            r"shift|interview|flight|deadline)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claimed_known_event",
        re.compile(
            r"\b(?:before|after)\s+(?:the|your)\s+(?:meeting|presentation|appointment|class|"
            r"exam|shift|interview|flight|deadline)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "invented_presentation_work",
        re.compile(
            r"\b(?:prepare|review|rehearse|finish)\s+(?:your\s+)?(?:slides|presentation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claimed_task_inventory",
        re.compile(
            r"\b(?:your|tomorrow(?:'s)?)\s+"
            r"(?:(?:core|current|existing|planned|relevant|highest[- ]priority)\s+){0,2}"
            r"(?:task\s+list|to-?do\s+list|tasks?|projects?|goals?|priorities)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claimed_work_object",
        re.compile(
            r"\b(?:review|prepare|organize|rank|prioritize|draft|finish|check|rehearse|work\s+on)\s+"
            r"(?:your\s+)?(?:(?:core|current|existing|planned|relevant|available)\s+){0,2}"
            r"(?:tasks?|projects?|goals?|priorities|materials?|documents?|notes?|workspace|"
            r"resources?|session)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "claimed_preparation_context",
        re.compile(
            r"\b(?:tomorrow(?:'s)?\s+session|prepare\s+(?:your\s+)?materials?|"
            r"review\s+(?:your\s+|relevant\s+)?documents?|organize\s+(?:your\s+)?notes?|"
            r"ensure\s+(?:your\s+|the\s+)?workspace|available\s+resources?)\b",
            re.IGNORECASE,
        ),
    ),
)

_CORE_REQUIRED_LABELS = ("GOAL", "STATED_CONSTRAINTS", "UNKNOWNS", "VERIFY")
_CORE_REFUSAL_PATTERNS = (
    re.compile(r"\bI\s+(?:cannot|can't|am unable to)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+provide\b", re.IGNORECASE),
    re.compile(r"\bcan I help you with something else\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:provide|help|comply)\b", re.IGNORECASE),
)

_MINUTE_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty-five": 25,
    "thirty": 30,
    "forty": 40,
    "forty-five": 45,
    "fifty": 50,
    "sixty": 60,
}
_MINUTE_TOKEN = r"(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|twenty-five|thirty|forty|forty-five|fifty|sixty)"
_PER_STEP_TIME_PATTERNS = (
    re.compile(
        rf"\b(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\s+(?:per|for each)\s+(?:action|step)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\s+(?:on\s+)?each\s+(?:action|step)\b",
        re.IGNORECASE,
    ),
)
_NUMBERED_STEP_BLOCK = re.compile(
    r"(?ms)(?<!\d)([123])[.)]\s+(.*?)(?=(?<!\d)[123][.)]\s+|\Z)"
)
_DIRECT_STEP_TIME_PATTERNS = (
    re.compile(
        rf"\b(?:spend|allocate|use|reserve|take)\s+(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bset\s+aside\s+(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\s+(?:to|for|on)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:first|next|final)\s+(?P<minutes>{_MINUTE_TOKEN})\s*(?:minutes?|mins?)\b",
        re.IGNORECASE,
    ),
)


def _minute_value(token: str) -> int | None:
    normalized = token.strip().lower()
    if normalized.isdigit():
        return int(normalized)
    return _MINUTE_WORDS.get(normalized)


def _time_budget_evidence(text: str, step_count: int) -> list[dict]:
    evidence = []
    if step_count <= 0:
        return evidence
    for pattern in _PER_STEP_TIME_PATTERNS:
        for match in pattern.finditer(text):
            per_step = _minute_value(match.group("minutes"))
            if per_step is None:
                continue
            evidence.append({
                "phrase": match.group(0),
                "minutes_per_step": per_step,
                "step_count": step_count,
                "implied_total_minutes": per_step * step_count,
            })
    return evidence


def _step_time_allocations(text: str) -> list[dict]:
    allocations = []
    for step, body in _NUMBERED_STEP_BLOCK.findall(text):
        accepted_spans = []
        for pattern in _DIRECT_STEP_TIME_PATTERNS:
            for match in pattern.finditer(body):
                start, end = match.span()
                if any(start < other_end and end > other_start for other_start, other_end in accepted_spans):
                    continue
                minutes = _minute_value(match.group("minutes"))
                if minutes is None:
                    continue
                accepted_spans.append((start, end))
                allocations.append({
                    "step": int(step),
                    "phrase": match.group(0),
                    "minutes": minutes,
                })
    return allocations


def observed_routes(route_attempts) -> list[dict]:
    return [{
        "provider": getattr(attempt.provider, "value", attempt.provider),
        "model": attempt.model,
        "reason": attempt.reason,
        "outcome": attempt.outcome,
    } for attempt in route_attempts]


def observed_councils(council_results) -> list[dict]:
    return [{
        "completed": result is not None,
        "core_brief": result.core_brief if result is not None else None,
        "models": list(result.models) if result is not None else [],
        "notes": [{"model": note.model, "text": note.text} for note in result.notes]
        if result is not None else [],
    } for result in council_results]


def validate_inference(response_text, council_results, route_attempts) -> dict:
    """Reject fallback-only evidence even if the final response is nonempty."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("The production JARVIS response was empty.")
    if len(council_results) != 1:
        raise RuntimeError("Expected exactly one production council consultation.")
    result = council_results[0]
    if result is None or not isinstance(result.core_brief, str) or not result.core_brief.strip():
        raise RuntimeError("JARVIS Core did not produce a council routing brief.")
    if CORE_MODEL not in result.models:
        raise RuntimeError("The council result did not identify JARVIS Core.")
    notes = [
        {"model": note.model, "text": note.text.strip()}
        for note in result.notes
        if isinstance(note.text, str) and note.text.strip()
    ]
    if EXPERT_MODEL not in result.models or not any(note["model"] == EXPERT_MODEL for note in notes):
        raise RuntimeError("The real fast expert did not contribute a council note.")
    attempts = observed_routes(route_attempts)
    successful = [attempt for attempt in attempts if attempt["outcome"] == "succeeded"]
    if not successful or any(attempt["provider"] != "ollama" for attempt in successful):
        raise RuntimeError("The final response did not complete through a local Ollama route.")
    return {
        "response": response_text.strip(),
        "core_brief": result.core_brief.strip(),
        "expert_notes": notes,
        "council_models": list(result.models),
        "route_attempts": attempts,
        "core_and_expert_completed": True,
    }


def validate_core_brief_quality(core_brief: str) -> dict:
    """Certify that deterministic Core normalization preserved the owner contract."""
    if not isinstance(core_brief, str) or not core_brief.strip():
        raise RuntimeError("Cannot evaluate an empty JARVIS Core brief.")

    text = core_brief.strip()
    matches = re.findall(
        r"(?im)^(GOAL|STATED_CONSTRAINTS|UNKNOWNS|VERIFY):\s*(.*)$",
        text,
    )
    labels_seen = [label for label, _ in matches]
    fields = {label: value.strip() for label, value in matches}
    refusal_markers = [
        pattern.pattern for pattern in _CORE_REFUSAL_PATTERNS
        if pattern.search(text)
    ]
    constraints = fields.get("STATED_CONSTRAINTS", "")
    lower_constraints = constraints.lower()
    has_authoritative_request = "authoritative_owner_request:" in lower_constraints
    has_time_budget = bool(re.search(r"\b(?:30|thirty)\b", constraints, re.IGNORECASE))
    has_step_shape = bool(
        re.search(r"\b(?:3|three)\b", constraints, re.IGNORECASE)
        and re.search(r"\bnumbered\b", constraints, re.IGNORECASE)
        and re.search(r"\bsteps?\b", constraints, re.IGNORECASE)
    )
    has_word_budget = bool(
        re.search(r"\b(?:80|eighty)\b", constraints, re.IGNORECASE)
        and re.search(r"\bwords?\b", constraints, re.IGNORECASE)
    )
    has_grounding_prohibition = (
        "calendar" in lower_constraints
        and "events" in lower_constraints
        and "preferences" in lower_constraints
        and bool(re.search(r"\b(?:do\s+not|don't)\s+assume\b", constraints, re.IGNORECASE))
    )

    failures = []
    if labels_seen != list(_CORE_REQUIRED_LABELS):
        failures.append("core_brief_protocol_mismatch")
    if refusal_markers:
        failures.append("core_brief_refusal")
    if not has_authoritative_request:
        failures.append("core_brief_missing_authoritative_request")
    if not has_time_budget:
        failures.append("core_brief_omitted_time_budget")
    if not has_step_shape:
        failures.append("core_brief_omitted_output_shape")
    if not has_word_budget:
        failures.append("core_brief_omitted_word_budget")
    if not has_grounding_prohibition:
        failures.append("core_brief_omitted_grounding_prohibition")

    return {
        "certified": not failures,
        "scope": "core-coordinator-routing-brief-v2",
        "required_labels": list(_CORE_REQUIRED_LABELS),
        "labels_seen": labels_seen,
        "refusal_markers": refusal_markers,
        "authoritative_request_preserved": has_authoritative_request,
        "time_budget_preserved": has_time_budget,
        "output_shape_preserved": has_step_shape,
        "word_budget_preserved": has_word_budget,
        "grounding_prohibition_preserved": has_grounding_prohibition,
        "failures": failures,
    }


def validate_grounding_quality(response_text: str) -> dict:
    """Evaluate the narrow planning probe without asking another model to grade it.

    The gate intentionally certifies only properties that can be observed
    deterministically: requested numbered shape, word budget, absence of known
    unsupported personal assumptions, and consistency with the explicit thirty-
    minute time budget. It is not a general intelligence benchmark.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Cannot evaluate grounding quality for an empty response.")

    text = response_text.strip()
    word_count = len(re.findall(r"\S+", text))
    step_markers = re.findall(r"(?<!\d)([123])[.)](?=\s)", text)
    assumption_markers = [
        name for name, pattern in _UNSUPPORTED_ASSUMPTION_PATTERNS
        if pattern.search(text)
    ]
    time_budget_evidence = _time_budget_evidence(text, len(step_markers))
    step_time_allocations = _step_time_allocations(text)
    explicit_step_total_minutes = sum(item["minutes"] for item in step_time_allocations)
    time_budget_violations = [
        item for item in time_budget_evidence
        if item["implied_total_minutes"] > TIME_BUDGET_MINUTES
    ]
    if explicit_step_total_minutes > TIME_BUDGET_MINUTES:
        time_budget_violations.append({
            "kind": "explicit_step_allocations",
            "total_minutes": explicit_step_total_minutes,
            "budget_minutes": TIME_BUDGET_MINUTES,
            "allocations": step_time_allocations,
        })

    failures = []
    if word_count > 80:
        failures.append("word_limit_exceeded")
    if step_markers != ["1", "2", "3"]:
        failures.append("missing_exact_three_numbered_steps")
    if assumption_markers:
        failures.append("unsupported_personal_assumption")
    if time_budget_violations:
        failures.append("time_budget_inconsistent")

    return {
        "certified": not failures,
        "scope": "thirty-minute-planning-grounding-probe-v7",
        "word_count": word_count,
        "word_limit": 80,
        "numbered_steps_seen": step_markers,
        "unsupported_assumption_markers": assumption_markers,
        "time_budget_minutes": TIME_BUDGET_MINUTES,
        "time_budget_evidence": time_budget_evidence,
        "explicit_step_time_allocations": step_time_allocations,
        "explicit_step_total_minutes": explicit_step_total_minutes,
        "time_budget_violations": time_budget_violations,
        "failures": failures,
    }


def run_inference(base_url: str, report: dict) -> None:
    from core.hardware_profile import HardwareProfiler
    from core.jarvis_brain import BrainPolicy, JarvisBrain
    from core.jarvis_council import JarvisCouncil
    from core.model_router import TaskKind
    from core.model_runtime import ModelRuntime

    class ObservedCouncil(JarvisCouncil):
        """Record real consultations without replacing any model call."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = []
            self.calls = []

        def consult(self, message, task):
            started = time.monotonic()
            try:
                result = super().consult(message, task)
            except Exception as exc:
                self.calls.append({"completed": False, "error_type": type(exc).__name__})
                raise
            self.results.append(result)
            self.calls.append({
                "completed": result is not None,
                "models": list(result.models) if result is not None else [],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
            return result

    profiler = HardwareProfiler(ollama_base_url=base_url)
    snapshot = profiler.capture()
    report["runner_hardware"] = {
        "ram_total_gb": round(snapshot.total_ram_gb, 3),
        "ram_available_gb": round(snapshot.available_ram_gb, 3),
        "system_pressure": round(snapshot.system_pressure, 3),
        "cpu_count": snapshot.cpu_count,
        "gpu_vram_gb": snapshot.gpu_vram_gb,
        "warnings": list(snapshot.warnings),
    }
    runtime = ModelRuntime(ollama_base_url=base_url, timeout_seconds=60)
    council = ObservedCouncil(ollama_base_url=base_url, profiler=profiler, runtime=runtime)
    brain = JarvisBrain(
        policy=BrainPolicy(allow_cloud=False),
        profiler=profiler,
        model_runtime=runtime,
        ollama_base_url=base_url,
        council=council,
    )
    started = time.monotonic()
    try:
        # respond() is the real chat path and never invokes AgentExecutor.execute.
        response = brain.respond(PROMPT, task=TaskKind.GENERAL)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["council_calls"] = council.calls
        # Preserve partial evidence when synthesis/routing fails after a real
        # council completed. Success validation must not hide that failure path.
        report["council_results"] = observed_councils(council.results)
        report["route_attempts"] = observed_routes(brain.last_attempts)
        report["identity"] = brain.identity
        report["lane"] = brain.lane.value

    report.update(validate_inference(response, council.results, brain.last_attempts))
    core_quality = validate_core_brief_quality(report["core_brief"])
    final_quality = validate_grounding_quality(response)
    report["core_brief_quality"] = core_quality
    report["core_brief_quality_certified"] = core_quality["certified"]
    report["intelligence_quality"] = final_quality
    report["intelligence_quality_certified"] = (
        core_quality["certified"] and final_quality["certified"]
    )
    combined_failures = [
        *core_quality["failures"],
        *final_quality["failures"],
    ]
    if combined_failures:
        failures = ", ".join(combined_failures)
        raise RuntimeError(f"JARVIS intelligence quality gate failed: {failures}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {
        "successful": False,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "execution_path": "JarvisBrain.respond -> JarvisCouncil -> local Ollama",
        "prompt": PROMPT,
        "cloud_allowed": False,
        "tools_executed": False,
        "physical_device_certified": False,
        "core_brief_quality_certified": False,
        "intelligence_quality_certified": False,
    }
    try:
        report["source_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5,
        ).strip()
        report["application_source_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal", "--",
             "core", "agent", "models", "packaging"],
            cwd=ROOT, text=True, timeout=5,
        ).strip())
        report["smoke_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["manifest_sha256"] = hashlib.sha256(
            (ROOT / "models" / "manifest.json").read_bytes()
        ).hexdigest()
        run_inference(args.ollama_url, report)
        report["successful"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
