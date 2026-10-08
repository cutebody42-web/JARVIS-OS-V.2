"""Console compatibility helpers for the desktop runtime and test harness."""

from __future__ import annotations

import sys


def configure_utf8_console() -> None:
    """Keep Unicode status output from crashing Windows CP1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # GUI/frozen runtimes may expose a stream that cannot be reconfigured.
            # Status output must never prevent JARVIS from starting.
            continue
