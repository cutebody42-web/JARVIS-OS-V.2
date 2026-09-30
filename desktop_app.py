"""Installed desktop entry point for local-first JARVIS.

This is the product entrypoint packaged into JARVIS.exe. Python is embedded by
the desktop build; the owner never needs to install or invoke Python manually.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback

from core.action_gateway import create_runtime
from core.hardware_profile import HardwareProfiler
from core.jarvis_brain import BrainPolicy, JarvisBrain
from core.model_runtime import ModelRuntime
from core.ollama_bootstrap import (
    OllamaBootstrapError,
    bootstrap_local_brain,
    find_ollama_executable,
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cloud-boost", action="store_true")
    parser.add_argument("--install-brain", action="store_true")
    parser.add_argument("--skip-brain-setup", action="store_true")
    return parser


def _confirm_ollama_install(parent) -> bool:
    from PyQt6.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        parent,
        "Set up the local JARVIS Brain",
        (
            "JARVIS needs the local Ollama runtime for private on-device AI.\n\n"
            "Install it now using the official Windows Package Manager package?\n"
            "Python is not required and will not be installed separately."
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    return answer is QMessageBox.StandardButton.Yes


def _safe_ui_log(ui, message: str) -> None:
    try:
        ui.write_log(message)
    except Exception:
        pass


def _setup_brain(ui, *, allow_install: bool, cloud_boost: bool) -> JarvisBrain:
    ui.set_state("THINKING")
    _safe_ui_log(ui, "SYS: CHECKING LOCAL JARVIS BRAIN")

    profiler = HardwareProfiler()
    snapshot = None
    try:
        snapshot = profiler.capture()
        _safe_ui_log(
            ui,
            (
                f"SYS: HARDWARE {snapshot.total_ram_gb:.1f} GB RAM · "
                f"{snapshot.available_ram_gb:.1f} GB AVAILABLE"
            ),
        )
    except Exception:
        _safe_ui_log(ui, "SYS: HARDWARE TELEMETRY PARTIAL — SAFE PROFILE ACTIVE")

    def progress(phase: str, percent: float, message: str) -> None:
        _safe_ui_log(ui, f"SYS: [{phase.upper()} {percent:05.1f}%] {message}")

    result = bootstrap_local_brain(
        snapshot=snapshot,
        allow_install=allow_install,
        progress=progress,
    )
    _safe_ui_log(
        ui,
        "SYS: JARVIS BRAIN READY · " + ", ".join(result.provisioned or ("existing models",)),
    )
    if result.skipped:
        _safe_ui_log(
            ui,
            "SYS: DEFERRED LANES · " + ", ".join(result.skipped),
        )

    owner_runtime = create_runtime(
        owner_id="local-owner",
        environment="desktop",
    )
    model_runtime = ModelRuntime()
    brain = JarvisBrain(
        policy=BrainPolicy(allow_cloud=cloud_boost),
        owner_runtime=owner_runtime,
        profiler=profiler,
        model_runtime=model_runtime,
    )
    ui.set_state("LISTENING")
    _safe_ui_log(
        ui,
        "SYS: CLOUD BOOST ON" if cloud_boost else "SYS: CLOUD BOOST OFF · LOCAL-FIRST",
    )
    return brain


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    os.environ["JARVIS_LOCAL_BRAIN"] = "1"
    os.environ.setdefault("JARVIS_SKIP_CLAP_GATE", "1")

    from ui import JarvisUI

    ui = JarvisUI("")
    ui.set_state("THINKING")

    state = {
        "brain": None,
        "setup_error": None,
        "closing": False,
    }

    def initialize() -> None:
        try:
            allow_install = bool(args.install_brain)
            if find_ollama_executable() is None and not allow_install:
                # The prompt itself must run on the Qt thread.
                decision = threading.Event()
                approved = {"value": False}

                def ask():
                    approved["value"] = _confirm_ollama_install(ui._win)
                    decision.set()

                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, ask)
                decision.wait()
                allow_install = approved["value"]

            if args.skip_brain_setup:
                if find_ollama_executable() is None:
                    raise OllamaBootstrapError("Local JARVIS Brain runtime is not installed.")
            brain = _setup_brain(
                ui,
                allow_install=allow_install,
                cloud_boost=bool(args.cloud_boost or _truthy(os.environ.get("JARVIS_CLOUD_BOOST"))),
            )
            state["brain"] = brain
            _safe_ui_log(ui, "JARVIS: Local brain online.")
        except Exception as exc:
            state["setup_error"] = exc
            ui.set_state("OFFLINE")
            _safe_ui_log(ui, f"ERR: LOCAL BRAIN SETUP FAILED · {str(exc)[:180]}")

    def command(text: str) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        brain = state.get("brain")
        if brain is None:
            _safe_ui_log(
                ui,
                "JARVIS: My local brain is still initializing." if state.get("setup_error") is None
                else "JARVIS: The local brain setup needs attention before I can respond.",
            )
            return

        ui.set_state("THINKING")
        try:
            reply = brain.handle(clean)
            _safe_ui_log(ui, f"JARVIS: {reply}")
            try:
                ui.show_subtitle(reply)
                ui.start_subtitle_hold()
            except Exception:
                pass
        except Exception as exc:
            _safe_ui_log(ui, f"ERR: {str(exc)[:220]}")
        finally:
            if not state["closing"]:
                ui.set_state("LISTENING")

    def quit_requested():
        state["closing"] = True
        try:
            ui._app.quit()
        except Exception:
            pass

    ui.on_text_command = command
    ui.on_quit_requested = quit_requested
    threading.Thread(target=initialize, name="jarvis-brain-bootstrap", daemon=True).start()

    try:
        ui.root.mainloop()
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        state["closing"] = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
