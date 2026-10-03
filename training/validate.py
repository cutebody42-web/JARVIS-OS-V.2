"""Validate dataset rights, explicit splits and basic leakage without ML deps."""

import argparse
import json
from pathlib import Path

from training.common import PipelineError, ROOT, architecture_lock, dataset_summary, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "starter.jsonl")
    args = parser.parse_args()
    try:
        lock = architecture_lock()
        records = load_dataset(args.data)
    except PipelineError as error:
        parser.exit(1, f"Dataset validation failed: {error}\n")
    print(json.dumps({"model_name": lock["model_name"], "scratch_parameters": lock["expected_parameters"], **dataset_summary(records)}, indent=2))


if __name__ == "__main__":
    main()
