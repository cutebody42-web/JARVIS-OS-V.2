"""Report compute needs before allocating a random 1.5B model or downloading."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from training.common import architecture_lock, private_output, private_root


def memory_gib() -> float:
    """Use the cgroup limit when it is smaller than host RAM."""
    limits = []
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        pass
    for candidate in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(candidate).read_text().strip()
            if value != "max":
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return round(min(limits) / 2**30, 2) if limits else 0.0


def gpu_info() -> list[dict]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    result = subprocess.run(
        [executable, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode:
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3:
            try:
                devices.append({"name": fields[0], "memory_gib": round(float(fields[1]) / 1024, 2), "free_gib": round(float(fields[2]) / 1024, 2)})
            except ValueError:
                continue
    return devices


def inspect(output: Path, *, development: bool = False) -> dict:
    lock = architecture_lock()
    existing = output
    while not existing.exists():
        existing = existing.parent
    devices = gpu_info()
    free_disk = round(shutil.disk_usage(existing).free / 2**30, 2)
    available_ram = memory_gib()
    minimum_disk = 80 if development else 200
    missing = [name for name in ("torch", "transformers", "accelerate", "safetensors") if importlib.util.find_spec(name) is None]
    reasons = []
    if not devices:
        reasons.append("No NVIDIA CUDA GPU was detected; CPU training of this model is rejected.")
    elif not any(device["memory_gib"] >= lock["minimum_gpu_memory_gib"] and device["free_gib"] >= 36 for device in devices):
        reasons.append("This full-weight AdamW recipe needs a GPU with at least 48 GiB VRAM and 36 GiB currently free.")
    if available_ram < 16:
        reasons.append("This recipe requires at least 16 GiB host/cgroup RAM.")
    if free_disk < minimum_disk:
        reasons.append(f"This recipe requires at least {minimum_disk} GiB free private storage.")
    if missing:
        reasons.append("Missing isolated training dependencies: " + ", ".join(missing) + ".")
    parameters = lock["expected_parameters"]
    token_target = parameters * lock["planning_tokens_per_parameter"]
    flops = 6 * parameters * token_target
    return {
        "eligible_for_this_recipe": not reasons,
        "reasons": reasons,
        "parameters": parameters,
        "all_weights_trainable": True,
        "cpu_count": os.cpu_count(),
        "ram_limit_gib": available_ram,
        "free_disk_gib": free_disk,
        "gpus": devices,
        "planning_token_target": token_target,
        "approximate_training_flops": flops,
        "illustrative_days_at_100_sustained_tflops": round(flops / 1e14 / 86400, 1),
        "estimate_note": "The 20 tokens/parameter and 6*N*T figures are planning heuristics, not measured throughput or a guarantee of intelligence. Hardware must be benchmarked before scheduling.",
        "state": "No model allocation, checkpoint, corpus download or external training job was performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=private_root() / "runs")
    parser.add_argument("--development", action="store_true", help="Inspect smaller development storage needs; this never marks a model production-ready.")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = inspect(private_output(args.output), development=args.development)
    print(json.dumps(report, indent=2))
    if args.require_ready and not report["eligible_for_this_recipe"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
