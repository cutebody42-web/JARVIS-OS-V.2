#!/usr/bin/env python3
"""Run the reviewed JARVIS conversational intelligence benchmark.

The benchmark is intentionally separate from release gating. It can either replay
recorded responses deterministically or exercise the real local JarvisBrain path
against an already-running Ollama. It never invokes AgentExecutor or action tools.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = ROOT / "benchmarks" / "intelligence_v1.json"


def load_suite(path: Path) -> dict:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("unsupported intelligence benchmark version")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark must contain at least one case")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("benchmark case must be an object")
        case_id = case.get("id")
        prompt = case.get("prompt")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("benchmark case ids must be unique non-empty strings")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"benchmark case {case_id} has no prompt")
        seen.add(case_id)
        for key in ("required_regex", "forbidden_regex"):
            patterns = case.get(key, [])
            if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
                raise ValueError(f"benchmark case {case_id} has invalid {key}")
            for pattern in patterns:
                re.compile(pattern)
        steps = case.get("numbered_steps")
        if steps is not None and (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(item, str) and item.isdigit() for item in steps)
        ):
            raise ValueError(f"benchmark case {case_id} has invalid numbered_steps")
        max_words = case.get("max_words")
        if max_words is not None and (
            isinstance(max_words, bool) or not isinstance(max_words, int) or not 1 <= max_words <= 1000
        ):
            raise ValueError(f"benchmark case {case_id} has invalid max_words")
    return value


def evaluate_case(case: dict, response: str) -> dict:
    if not isinstance(response, str) or not response.strip():
        return {
            "certified": False,
            "failures": ["empty_response"],
            "word_count": 0,
            "numbered_steps_seen": [],
            "required_regex_missing": [],
            "forbidden_regex_seen": [],
        }

    text = response.strip()
    failures = []
    word_count = len(re.findall(r"\S+", text))
    expected_steps = case.get("numbered_steps")
    steps_seen = re.findall(r"(?<!\d)(\d+)[.)](?=\s)", text)
    if expected_steps is not None and steps_seen != expected_steps:
        failures.append("numbered_step_protocol_mismatch")

    max_words = case.get("max_words")
    if max_words is not None and word_count > max_words:
        failures.append("word_limit_exceeded")

    required_missing = [
        pattern for pattern in case.get("required_regex", [])
        if re.search(pattern, text) is None
    ]
    forbidden_seen = [
        pattern for pattern in case.get("forbidden_regex", [])
        if re.search(pattern, text) is not None
    ]
    if required_missing:
        failures.append("required_evidence_missing")
    if forbidden_seen:
        failures.append("unsupported_or_forbidden_claim")

    return {
        "certified": not failures,
        "failures": failures,
        "word_count": word_count,
        "word_limit": max_words,
        "numbered_steps_seen": steps_seen,
        "numbered_steps_expected": expected_steps,
        "required_regex_missing": required_missing,
        "forbidden_regex_seen": forbidden_seen,
    }


def _load_recorded_responses(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("responses file must be a JSON object mapping case id to response text")
    return value


def _run_live(cases: list[dict], ollama_url: str) -> dict[str, dict]:
    sys.path.insert(0, str(ROOT))
    from core.hardware_profile import HardwareProfiler
    from core.jarvis_brain import BrainPolicy, JarvisBrain
    from core.model_router import TaskKind
    from core.model_runtime import ModelRuntime

    profiler = HardwareProfiler(ollama_base_url=ollama_url)
    runtime = ModelRuntime(ollama_base_url=ollama_url, timeout_seconds=60)
    brain = JarvisBrain(
        policy=BrainPolicy(allow_cloud=False),
        profiler=profiler,
        model_runtime=runtime,
        ollama_base_url=ollama_url,
    )
    observed = {}
    for case in cases:
        started = time.monotonic()
        response = brain.respond(case["prompt"], task=TaskKind.GENERAL)
        observed[case["id"]] = {
            "response": response,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "route_attempts": [
                {
                    "provider": getattr(item.provider, "value", item.provider),
                    "model": item.model,
                    "reason": item.reason,
                    "outcome": item.outcome,
                }
                for item in brain.last_attempts
            ],
        }
    return observed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--responses", type=Path, help="Replay a JSON case-id -> response mapping")
    source.add_argument("--ollama-url", help="Run the real local JarvisBrain against this Ollama URL")
    parser.add_argument("--case", action="append", dest="case_ids", help="Run only this case id; repeatable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    cases = suite["cases"]
    if args.case_ids:
        selected = set(args.case_ids)
        known = {case["id"] for case in cases}
        unknown = sorted(selected - known)
        if unknown:
            parser.error("unknown case id(s): " + ", ".join(unknown))
        cases = [case for case in cases if case["id"] in selected]

    if args.responses:
        recorded = _load_recorded_responses(args.responses)
        observations = {
            case["id"]: {"response": recorded.get(case["id"], ""), "elapsed_seconds": None, "route_attempts": []}
            for case in cases
        }
        mode = "recorded_replay"
    else:
        observations = _run_live(cases, args.ollama_url)
        mode = "real_local_jarvis_brain"

    results = []
    for case in cases:
        observation = observations[case["id"]]
        evaluation = evaluate_case(case, observation["response"])
        results.append({
            "id": case["id"],
            "language": case.get("language"),
            "prompt": case["prompt"],
            **observation,
            "evaluation": evaluation,
        })

    source_sha = None
    try:
        source_sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        pass

    certified_count = sum(1 for item in results if item["evaluation"]["certified"])
    report = {
        "kind": "jarvis_intelligence_benchmark_v1",
        "mode": mode,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "case_count": len(results),
        "certified_count": certified_count,
        "all_certified": certified_count == len(results),
        "release_blocking": False,
        "limits": (
            "Deterministic surface checks over a reviewed prompt set. This does not certify general "
            "intelligence, factual correctness, physical-device behavior or action execution."
        ),
        "results": results,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", "utf-8")
    print(text)
    return 0 if report["all_certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
