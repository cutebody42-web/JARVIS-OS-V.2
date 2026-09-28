#!/usr/bin/env python3
"""Compare real Phase 1/2 clock paths with an offline provider; no device claims."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def worker(root, route, iterations):
    sys.path.insert(0, str(root))
    from agent.executor import AgentExecutor
    from core.model_provider import ModelResponse
    class OfflineProvider:
        calls = 0
        def generate(self, request):
            self.calls += 1
            return ModelResponse(json.dumps({"goal": "clock", "steps": [{
                "step": 1, "tool": "system_time", "parameters": {}, "description": "Read clock"
            }]}), "offline", "stub")
    provider = OfflineProvider()
    executor = AgentExecutor(provider=provider)
    command = "What time is it?" if route == "reflex" else "Report the machine clock reading"
    for _ in range(20):
        executor.execute(command)
    provider.calls = 0
    samples = []
    for _ in range(iterations):
        tick = time.perf_counter_ns()
        executor.execute(command)
        samples.append((time.perf_counter_ns() - tick) / 1e6)
        assert executor.last_action_receipts[-1].result.status.value == "succeeded"
    return {"iterations": iterations, "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(iterations * .95)], "provider_calls_per_command": provider.calls / iterations,
            "receipts_per_command": len(executor.last_action_receipts),
            "receipt_schema": executor.last_action_receipts[-1].schema_version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("reflex", "model"))
    args = parser.parse_args()
    if not 100 <= args.iterations <= 100000:
        parser.error("Use 100..100000 samples")
    if args.worker:
        print(json.dumps(worker(args.baseline_dir.resolve(), args.worker, args.iterations)))
        return
    result = {"kind": "cloud_offline_control_flow", "measured_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "baseline_commit": subprocess.check_output(["git", "-C", str(args.baseline_dir), "rev-parse", "HEAD"], text=True).strip(),
              "limits": "Real native clock/receipt/policy paths. Stubbed provider; zero model/network latency. No laptop, Windows, microphone, camera, iGPU or end-to-end benchmark."}
    for phase, root in (("phase1", args.baseline_dir), ("phase2", Path(__file__).resolve().parents[1])):
        result[phase] = {}
        for route in ("reflex", "model"):
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--baseline-dir", str(root.resolve()),
                                      "--iterations", str(args.iterations), "--worker", route],
                                     env={**os.environ, "JARVIS_QA_MODE": "0"}, text=True, capture_output=True, check=True)
            result[phase][route] = json.loads(process.stdout)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
