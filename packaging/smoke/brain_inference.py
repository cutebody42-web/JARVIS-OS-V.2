#!/usr/bin/env python3
"""Verify one real conversational turn through the production local AI stack.

Requires an already-running Ollama with JARVIS Core and the fast expert. This
script does not download weights, mock providers, attach owner memory, execute
tools, or certify physical hardware or intelligence quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.jarvis_council import CORE_MODEL

EXPERT_MODEL = "jarvis-brain-fast"
PROMPT = (
    "I have thirty minutes to prepare for tomorrow. Give me three short planning "
    "steps using only that information. Do not assume my calendar or preferences. "
    "Keep the final answer under eighty words."
)


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
    attempts = [
        {
            "provider": getattr(attempt.provider, "value", attempt.provider),
            "model": attempt.model,
            "reason": attempt.reason,
            "outcome": attempt.outcome,
        }
        for attempt in route_attempts
    ]
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
    report.update(validate_inference(response, council.results, brain.last_attempts))
    report["identity"] = brain.identity
    report["lane"] = brain.lane.value


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
