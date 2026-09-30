"""First-run provisioning for the local JARVIS Brain.

The desktop installer bundles this module, so the owner never needs to install
Python or run pip. Ollama is installed separately through its official Windows
package when the owner explicitly approves local-AI setup.

All subprocess calls use argument arrays; no shell or generated command text is
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable

import requests

from core.app_paths import resource_path
from core.hardware_profile import HardwareSnapshot


Progress = Callable[[str, float, str], None]


@dataclass(frozen=True)
class BrainModelSpec:
    alias: str
    base: str
    role: str
    min_ram_gb: float
    modelfile: Path


@dataclass(frozen=True)
class BootstrapResult:
    ollama_path: str
    installed_ollama: bool
    provisioned: tuple[str, ...]
    skipped: tuple[str, ...]


class OllamaBootstrapError(RuntimeError):
    pass


def _emit(callback: Progress | None, phase: str, percent: float, message: str) -> None:
    if callback is not None:
        callback(phase, max(0.0, min(100.0, percent)), message)


def load_brain_manifest() -> tuple[BrainModelSpec, ...]:
    manifest_path = resource_path("models", "manifest.json")
    try:
        value = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaBootstrapError("JARVIS Brain model manifest is unavailable.") from exc

    if value.get("schema_version") != 1 or not isinstance(value.get("models"), list):
        raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")

    specs = []
    for item in value["models"]:
        if not isinstance(item, dict):
            raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")
        alias = item.get("alias")
        base = item.get("base")
        role = item.get("role")
        min_ram = item.get("min_ram_gb")
        if (
            not isinstance(alias, str)
            or not isinstance(base, str)
            or not isinstance(role, str)
            or isinstance(min_ram, bool)
            or not isinstance(min_ram, (int, float))
            or min_ram < 0
        ):
            raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")
        modelfile = resource_path("models", f"{alias}.Modelfile")
        if not modelfile.is_file():
            raise OllamaBootstrapError(f"Missing Modelfile for {alias}.")
        specs.append(
            BrainModelSpec(alias, base, role, float(min_ram), modelfile)
        )
    return tuple(specs)


def find_ollama_executable() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found

    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if local.is_file():
            return str(local)
        program_files = Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe"
        if program_files.is_file():
            return str(program_files)
    return None


def install_ollama_windows(*, runner=subprocess.run) -> str:
    if sys.platform != "win32":
        raise OllamaBootstrapError("Automatic Ollama installation is currently Windows-only.")

    winget = shutil.which("winget")
    if not winget:
        raise OllamaBootstrapError(
            "Windows Package Manager (winget) is required for automatic Ollama installation."
        )

    command = [
        winget,
        "install",
        "--id",
        "Ollama.Ollama",
        "--exact",
        "--silent",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]
    try:
        completed = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OllamaBootstrapError("Ollama installation could not be started.") from exc

    if completed.returncode not in {0, 3010}:
        raise OllamaBootstrapError("The official Ollama installer did not complete successfully.")

    found = find_ollama_executable()
    if not found:
        raise OllamaBootstrapError("Ollama installed but its executable could not be located.")
    return found


def wait_for_ollama(
    base_url: str = "http://127.0.0.1:11434",
    *,
    timeout_seconds: float = 45.0,
    poll_seconds: float = 0.5,
    request_get=requests.get,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = request_get(base_url + "/", timeout=1.5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(poll_seconds)
    return False


def ensure_ollama_service(
    executable: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    popen=subprocess.Popen,
) -> None:
    if wait_for_ollama(base_url, timeout_seconds=2.0):
        return

    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        popen([executable, "serve"], **kwargs)
    except OSError as exc:
        raise OllamaBootstrapError("Ollama service could not be started.") from exc

    if not wait_for_ollama(base_url):
        raise OllamaBootstrapError("Ollama service did not become ready.")


def _run_ollama(executable: str, args: list[str], *, runner=subprocess.run) -> None:
    try:
        completed = runner(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OllamaBootstrapError("Ollama model provisioning failed.") from exc
    if completed.returncode != 0:
        raise OllamaBootstrapError("Ollama model provisioning failed.")


def installed_model_names(
    executable: str,
    *,
    runner=subprocess.run,
) -> set[str]:
    try:
        completed = runner(
            [executable, "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if completed.returncode != 0:
        return set()

    names = set()
    for index, line in enumerate(completed.stdout.splitlines()):
        if index == 0 or not line.strip():
            continue
        names.add(line.split()[0].split(":latest")[0])
    return names


def select_brain_models(
    specs: Iterable[BrainModelSpec],
    snapshot: HardwareSnapshot | None,
) -> tuple[BrainModelSpec, ...]:
    specs = tuple(specs)
    if snapshot is None:
        # Conservative bootstrap: smallest lane only. Other lanes are pulled
        # lazily when hardware telemetry becomes available.
        lite = [spec for spec in specs if spec.role == "low_memory_fallback"]
        return tuple(lite[:1])

    capacity = max(snapshot.available_ram_gb, snapshot.total_ram_gb * 0.45)
    eligible = [spec for spec in specs if spec.min_ram_gb <= capacity]

    # Always keep a low-memory fallback if the manifest contains one.
    lite = next((spec for spec in specs if spec.role == "low_memory_fallback"), None)
    if lite is not None and lite not in eligible:
        eligible.append(lite)

    # On <=10 GB systems avoid downloading the 7B engineering lane by default.
    if snapshot.total_ram_gb <= 10:
        eligible = [spec for spec in eligible if spec.role != "engineering"]

    order = {"low_memory_fallback": 0, "general_realtime": 1, "engineering": 2}
    return tuple(sorted(eligible, key=lambda spec: order.get(spec.role, 99)))


def provision_brain_models(
    executable: str,
    *,
    snapshot: HardwareSnapshot | None = None,
    progress: Progress | None = None,
    runner=subprocess.run,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    specs = load_brain_manifest()
    selected = select_brain_models(specs, snapshot)
    selected_aliases = {spec.alias for spec in selected}
    skipped = tuple(spec.alias for spec in specs if spec.alias not in selected_aliases)
    existing = installed_model_names(executable, runner=runner)

    completed_aliases: list[str] = []
    total = max(1, len(selected))

    for index, spec in enumerate(selected):
        base_percent = index / total * 100
        if spec.alias in existing:
            _emit(progress, "models", base_percent, f"{spec.alias} already ready")
            completed_aliases.append(spec.alias)
            continue

        _emit(progress, "models", base_percent, f"Preparing {spec.alias}")
        _run_ollama(executable, ["pull", spec.base], runner=runner)
        _run_ollama(
            executable,
            ["create", spec.alias, "-f", str(spec.modelfile)],
            runner=runner,
        )
        completed_aliases.append(spec.alias)
        _emit(
            progress,
            "models",
            (index + 1) / total * 100,
            f"{spec.alias} ready",
        )

    return tuple(completed_aliases), skipped


def bootstrap_local_brain(
    *,
    snapshot: HardwareSnapshot | None = None,
    allow_install: bool = False,
    progress: Progress | None = None,
    runner=subprocess.run,
) -> BootstrapResult:
    executable = find_ollama_executable()
    installed = False

    if executable is None:
        if not allow_install:
            raise OllamaBootstrapError(
                "Ollama is not installed. Owner approval is required before installation."
            )
        _emit(progress, "ollama", 5, "Installing the local JARVIS Brain runtime")
        executable = install_ollama_windows(runner=runner)
        installed = True

    _emit(progress, "ollama", 20, "Starting the local JARVIS Brain runtime")
    ensure_ollama_service(executable)

    _emit(progress, "models", 25, "Preparing JARVIS Brain models")
    provisioned, skipped = provision_brain_models(
        executable,
        snapshot=snapshot,
        progress=progress,
        runner=runner,
    )
    _emit(progress, "complete", 100, "JARVIS Brain is ready")

    return BootstrapResult(executable, installed, provisioned, skipped)
