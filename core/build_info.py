"""Commit provenance embedded in each packaged desktop sidecar."""
from __future__ import annotations

import json
from pathlib import Path
import re

from core.app_paths import resource_path


def read_build_info(path: Path | None = None) -> dict[str, str]:
    candidate = path or resource_path("build-info.json")
    if candidate.stat().st_size > 16384:
        raise ValueError("Build provenance is too large")
    value = json.loads(candidate.read_text("utf-8"))
    if (not isinstance(value, dict)
            or not isinstance(value.get("version"), str)
            or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", value["version"])
            or not isinstance(value.get("commit_sha"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", value["commit_sha"])):
        raise ValueError("Build provenance must identify its version and exact source commit")
    return {"version": value["version"], "commit_sha": value["commit_sha"]}
