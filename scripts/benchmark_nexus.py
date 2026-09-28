#!/usr/bin/env python3
"""Offline control-flow benchmark; never contacts a provider or acts on the desktop.

The baseline runs its real planner/executor with a fake Gemini module and a
stubbed clock action. New paths use the native clock and real receipts. Network,
model quality, Windows/iGPU behavior and end-to-end voice latency are NOT measured.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


def worker(path, root, iterations):
    sys.path.insert(0, str(root))
    calls = [0]
    plan = json.dumps({"goal": "What time is it?", "steps": [{
        "step": 1, "tool": "system_time", "parameters": {}, "description": "Read clock",
    }]})
    if path == "legacy":
        google = ModuleType("google")
        sdk = ModuleType("google.generativeai")
        class Model:
            def __init__(self, *args, **kwargs):
                self.planning = bool(kwargs.get("system_instruction"))
            def generate_content(self, *args, **kwargs):
                calls[0] += 1
                return SimpleNamespace(text=plan if self.planning else "Local clock read (stub summary).")
        sdk.configure = lambda **kwargs: None
        sdk.GenerativeModel = Model
        google.generativeai = sdk
        sys.modules.update({"google": google, "google.generativeai": sdk})
        planner = importlib.import_module("agent.planner")
        executor_module = importlib.import_module("agent.executor")
        planner._get_api_key = lambda: "offline-placeholder"
        executor_module._get_api_key = lambda: "offline-placeholder"
        executor_module._call_tool = lambda *a, **k: datetime.now().astimezone().isoformat()
        executor = executor_module.AgentExecutor()
        command = "What time is it?"
    else:
        from core.model_provider import ModelResponse
        from agent.executor import AgentExecutor
        class OfflineProvider:
            def generate(self, request):
                calls[0] += 1
                return ModelResponse(plan, "offline", "stub")
        executor = AgentExecutor(provider=OfflineProvider())
        command = "What time is it?" if path == "reflex" else "Please report the machine clock reading"
    samples = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(20):
            executor.execute(command)
        calls[0] = 0
        for _ in range(iterations):
            started = time.perf_counter_ns()
            executor.execute(command)
            samples.append((time.perf_counter_ns() - started) / 1e6)
    samples.sort()
    return {"iterations": iterations, "median_ms": statistics.median(samples),
            "p95_ms": samples[max(0, (95 * iterations + 99) // 100 - 1)],
            "provider_calls_total": calls[0], "provider_calls_per_command": calls[0] / iterations,
            "evidence_receipts_per_command": len(getattr(executor, "last_action_receipts", []))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("legacy", "model", "reflex"))
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100000:
        parser.error("iterations must be between 1 and 100000")
    if args.worker:
        root = args.baseline_dir if args.worker == "legacy" else ROOT
        print(json.dumps(worker(args.worker, root.resolve(), args.iterations)))
        return
    result = {"kind": "offline_control_flow_only", "python": platform.python_version(),
              "platform": platform.platform(), "measured_at": datetime.now(timezone.utc).isoformat(),
              "limits": "Stubbed provider with zero network/model latency; baseline action is stubbed. No live Gemini, Windows, audio, RAM or iGPU benchmark."}
    result["baseline_commit"] = subprocess.check_output(
        ["git", "-C", str(args.baseline_dir), "rev-parse", "HEAD"], text=True).strip()
    for path in ("legacy", "model", "reflex"):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--baseline-dir", str(args.baseline_dir.resolve()), "--iterations", str(args.iterations),
            "--worker", path], check=True, capture_output=True, text=True)
        result[path] = json.loads(completed.stdout)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
