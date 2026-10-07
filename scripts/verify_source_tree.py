#!/usr/bin/env python3
"""Fail CI when the checked-out repository is only a partial/skeleton source tree.

GitHub's Code -> Download ZIP contains the tracked source tree for the selected
ref. This guard makes an accidentally tiny/default branch visible in CI instead
of letting a README-only or otherwise incomplete tree look like a valid JARVIS
checkout. Built installers, dependencies and local model weights intentionally
remain release/runtime artifacts and are not expected in the source archive.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MIN_TRACKED_FILES = 150
MIN_TRACKED_BYTES = 1_000_000

REQUIRED_PATHS = (
    "README.md",
    "Install-JARVIS.cmd",
    ".github/workflows/ci.yml",
    ".github/workflows/nexus-contracts.yml",
    "core",
    "actions",
    "agent",
    "models",
    "packaging",
    "scripts",
    "tests",
)


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return [ROOT / item.decode("utf-8") for item in raw.split(b"\0") if item]


def main() -> int:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    files = tracked_files()
    regular_files = [path for path in files if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in regular_files)

    report = {
        "tracked_files": len(regular_files),
        "tracked_bytes": total_bytes,
        "minimum_tracked_files": MIN_TRACKED_FILES,
        "minimum_tracked_bytes": MIN_TRACKED_BYTES,
        "required_paths": list(REQUIRED_PATHS),
        "missing_required_paths": missing,
        "complete": (
            not missing
            and len(regular_files) >= MIN_TRACKED_FILES
            and total_bytes >= MIN_TRACKED_BYTES
        ),
        "note": (
            "Source completeness only; installers, dependencies and model weights "
            "are intentionally not committed to Git."
        ),
    }
    print(json.dumps(report, indent=2))

    if missing:
        print("Incomplete source tree: required paths are missing.", file=sys.stderr)
    if len(regular_files) < MIN_TRACKED_FILES:
        print(
            f"Incomplete source tree: only {len(regular_files)} tracked files.",
            file=sys.stderr,
        )
    if total_bytes < MIN_TRACKED_BYTES:
        print(
            f"Incomplete source tree: tracked source is only {total_bytes} bytes.",
            file=sys.stderr,
        )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
