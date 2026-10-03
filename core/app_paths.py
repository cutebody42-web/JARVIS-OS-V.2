"""Runtime paths that work from source and frozen desktop builds."""

from __future__ import annotations

from pathlib import Path
import os
import sys


def resource_root() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        path = base / "JARVIS"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "JARVIS"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "jarvis"
    path.mkdir(parents=True, exist_ok=True)
    return path
