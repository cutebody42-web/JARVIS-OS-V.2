#!/usr/bin/env python3
"""Native clock control flow and real SQLite commits; no real model/device claims."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def summary(samples):
    return {"samples": len(samples), "median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[int(len(samples) * .95)], "provider_calls": 0}


def worker(root, durable, count):
    sys.path.insert(0, str(root))
    class NoModel:
        def generate(self, request):
            raise AssertionError("Reflex attempted inference")
    samples = []
    with tempfile.TemporaryDirectory() as directory:
        if durable:
            from agent.missions import MissionService
            host = MissionService(directory, provider=NoModel())
            def run():
                m = host.create_goal("benchmark-owner", "what time is it")
                result = host.run("benchmark-owner", m["id"], m["session_id"], m["version"])
                assert result["state"] == "succeeded"
        else:
            from agent.executor import AgentExecutor
            host = AgentExecutor(provider=NoModel())
            def run():
                host.execute("what time is it")
                assert host.last_status.value == "succeeded"
        for _ in range(10):
            run()
        for _ in range(count):
            start = time.perf_counter_ns()
            run()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        if durable:
            host.close()
            # Observe persistence with a new connection/host, no replay.
            reopened = MissionService(directory)
            with reopened.store._guard:
                stored = reopened.store._db.execute("SELECT COUNT(*) FROM missions WHERE state='succeeded'").fetchone()[0]
            assert stored == count + 10
            reopened.close()
    return summary(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--worker", choices=("bare", "durable"))
    args = parser.parse_args()
    if not 100 <= args.samples <= 900:
        parser.error("Use 100–900 samples within the production storage quota")
    if args.worker:
        print(json.dumps(worker(args.baseline_dir.resolve(), args.worker == "durable", args.samples)))
        return
    root = Path(__file__).resolve().parents[1]
    result = {"kind": "cloud_native_sqlite_overhead", "measured_at": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "baseline_commit": subprocess.check_output(["git", "-C", str(args.baseline_dir), "rev-parse", "HEAD"], text=True).strip(),
              "limits": "Native clock, real SQLite WAL/FULL commits, 10 warmups. Model-free; no laptop/Windows/iGPU/audio or real inference benchmark. Durable case includes creation, execution and journal reads.",
              "hardware_certified": False}
    for name, checkout, mode in (("phase2_bare_reflex", args.baseline_dir, "bare"),
                                ("phase3_bare_reflex", root, "bare"), ("phase3_durable_reflex", root, "durable")):
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--baseline-dir", str(checkout.resolve()),
                            "--samples", str(args.samples), "--worker", mode], check=True, capture_output=True,
                            text=True, env={**os.environ, "JARVIS_QA_MODE": "0"})
        result[name] = json.loads(p.stdout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
