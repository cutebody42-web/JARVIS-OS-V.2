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
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Iterable
from urllib.parse import urlparse

import requests

from core.app_paths import resource_path
from core.hardware_profile import HardwareSnapshot


Progress = Callable[[str, float, str], None]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# A model directory is a server setting, not a per-request/CLI setting. Keep
# ownership of servers we start so a later directory change can restart only
# our process, without terminating the owner's unrelated Ollama application.
_SERVICE_LOCK = threading.RLock()
_OWNED_SERVICES: dict[str, tuple[subprocess.Popen, str | None]] = {}


def _ollama_environment(
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
) -> dict[str, str]:
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise OllamaBootstrapError("JARVIS Ollama endpoint must be literal loopback HTTP.")
    try:
        port = parsed.port or 11434
    except ValueError as exc:
        raise OllamaBootstrapError("JARVIS Ollama endpoint has an invalid port.") from exc
    if parsed.port == 0:
        raise OllamaBootstrapError("JARVIS Ollama endpoint has an invalid port.")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    if model_store is not None:
        store = Path(model_store).expanduser()
        if not store.is_absolute():
            raise OllamaBootstrapError("Local model folder must be an absolute path.")
        env["OLLAMA_MODELS"] = str(store.resolve())
    return env


@dataclass(frozen=True)
class BrainModelSpec:
    alias: str
    base: str
    role: str
    min_ram_gb: float
    parameters_b: float
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
        parameters_b = item.get("parameters_b")
        if (
            not isinstance(alias, str)
            or not isinstance(base, str)
            or not isinstance(role, str)
            or isinstance(min_ram, bool)
            or not isinstance(min_ram, (int, float))
            or min_ram < 0
            or isinstance(parameters_b, bool)
            or not isinstance(parameters_b, (int, float))
            or parameters_b <= 0
        ):
            raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")
        modelfile = resource_path("models", f"{alias}.Modelfile")
        if not modelfile.is_file():
            raise OllamaBootstrapError(f"Missing Modelfile for {alias}.")
        specs.append(
            BrainModelSpec(alias, base, role, float(min_ram), float(parameters_b), modelfile)
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
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
    popen=subprocess.Popen,
) -> None:
    env = _ollama_environment(base_url, model_store)
    key = env["OLLAMA_HOST"]
    store = env.get("OLLAMA_MODELS")
    with _SERVICE_LOCK:
        owned = _OWNED_SERVICES.get(key)
        if owned is not None and owned[0].poll() is not None:
            _OWNED_SERVICES.pop(key, None)
            owned = None
        ready = wait_for_ollama(base_url, timeout_seconds=2.0)
        if ready:
            if owned is None:
                if model_store is not None:
                    raise OllamaBootstrapError(
                        "An unrelated Ollama service is already using this endpoint. "
                        "Its model folder cannot be changed safely; stop that service "
                        "or use a free local JARVIS endpoint."
                    )
                return
            if owned[1] == store:
                return

        if owned is not None:
            stop_owned_ollama_service(base_url=base_url)

        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            process = popen([executable, "serve"], **kwargs)
        except OSError as exc:
            raise OllamaBootstrapError("Ollama service could not be started.") from exc
        _OWNED_SERVICES[key] = (process, store)

        if not wait_for_ollama(base_url) or process.poll() is not None:
            stop_owned_ollama_service(base_url=base_url)
            raise OllamaBootstrapError("Ollama service did not become ready.")


def stop_owned_ollama_service(*, base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Stop only the Ollama child process created by this Python process."""
    key = _ollama_environment(base_url)["OLLAMA_HOST"]
    with _SERVICE_LOCK:
        owned = _OWNED_SERVICES.pop(key, None)
        if owned is None:
            return False
        process = owned[0]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return True


def _run_ollama(
    executable: str,
    args: list[str],
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
    runner=subprocess.run,
) -> None:
    kwargs = {"env": _ollama_environment(base_url, model_store)}
    try:
        completed = runner(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
            **kwargs,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OllamaBootstrapError("Ollama model provisioning failed.") from exc
    if completed.returncode != 0:
        raise OllamaBootstrapError("Ollama model provisioning failed.")


_PARAMETER_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([BM])\s*$", re.IGNORECASE)


def parameter_size_b(value: str) -> float:
    if not isinstance(value, str):
        raise ValueError("parameter size must be text")
    match = _PARAMETER_SIZE_RE.fullmatch(value)
    if match is None:
        raise ValueError("unsupported Ollama parameter size")
    amount = float(match.group(1))
    return amount if match.group(2).upper() == "B" else amount / 1000.0


def model_matches_manifest(
    model: str,
    expected_parameters_b: float,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    request_post=requests.post,
) -> bool:
    try:
        response = request_post(
            base_url + "/api/show",
            json={"model": model},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
        details = body.get("details") if isinstance(body, dict) else None
        label = details.get("parameter_size") if isinstance(details, dict) else None
        actual = parameter_size_b(label)
    except Exception:
        return False

    # The human-readable size is rounded, so allow a small difference from the
    # advertised family size. A sub-billion model cannot satisfy a 1B core.
    if expected_parameters_b >= 1.0 and actual < 1.0:
        return False
    tolerance = max(0.15, expected_parameters_b * 0.20)
    return abs(actual - expected_parameters_b) <= tolerance


def verify_model_inference(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    request_post=requests.post,
) -> bool:
    """Smoke-test actual text generation; this does not measure intelligence."""
    _ollama_environment(base_url)
    try:
        response = request_post(
            base_url + "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply briefly: JARVIS is ready."}],
                "stream": False,
                "think": False,
                "keep_alive": "5m",
                "options": {"num_ctx": 1024, "num_predict": 24, "temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        return bool(
            isinstance(body, dict)
            and body.get("done") is True
            and isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
            and not message.get("tool_calls")
        )
    except Exception:
        return False


def installed_model_names(
    executable: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
    runner=subprocess.run,
) -> set[str]:
    kwargs = {"env": _ollama_environment(base_url, model_store)}
    try:
        completed = runner(
            [executable, "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs,
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
    core = next((spec for spec in specs if spec.role == "coordinator"), None)
    if core is None:
        raise OllamaBootstrapError("JARVIS Core coordinator is missing from the model manifest.")

    if snapshot is None:
        # Unknown hardware still gets the dedicated JARVIS Core 1B coordinator.
        return (core,)

    capacity = max(snapshot.available_ram_gb, snapshot.total_ram_gb * 0.45)
    eligible = [spec for spec in specs if spec.min_ram_gb <= capacity]

    # JARVIS Core 1B is always provisioned; experts are additive and hidden.
    if core not in eligible:
        eligible.append(core)

    # Keep a low-memory expert fallback where possible.
    lite = next((spec for spec in specs if spec.role == "low_memory_fallback"), None)
    if lite is not None and lite.min_ram_gb <= capacity and lite not in eligible:
        eligible.append(lite)

    # On <=10 GB systems avoid downloading the 7B engineering lane by default.
    if snapshot.total_ram_gb <= 10:
        eligible = [spec for spec in eligible if spec.role != "engineering"]

    order = {"coordinator": 0, "low_memory_fallback": 1, "general_realtime": 2, "engineering": 3}
    return tuple(sorted(eligible, key=lambda spec: order.get(spec.role, 99)))


def provision_brain_models(
    executable: str,
    *,
    snapshot: HardwareSnapshot | None = None,
    progress: Progress | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
    runner=subprocess.run,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    specs = load_brain_manifest()
    selected = select_brain_models(specs, snapshot)
    selected_aliases = {spec.alias for spec in selected}
    skipped = tuple(spec.alias for spec in specs if spec.alias not in selected_aliases)
    existing = installed_model_names(
        executable,
        base_url=base_url,
        model_store=model_store,
        runner=runner,
    )

    completed_aliases: list[str] = []
    total = max(1, len(selected))

    for index, spec in enumerate(selected):
        base_percent = index / total * 100
        if spec.alias in existing and model_matches_manifest(
            spec.alias,
            spec.parameters_b,
            base_url=base_url,
        ):
            _emit(progress, "models", base_percent, f"{spec.alias} already ready")
            completed_aliases.append(spec.alias)
            continue

        _emit(progress, "models", base_percent, f"Preparing {spec.alias}")
        _run_ollama(
            executable,
            ["pull", spec.base],
            base_url=base_url,
            model_store=model_store,
            runner=runner,
        )
        _run_ollama(
            executable,
            ["create", spec.alias, "-f", str(spec.modelfile)],
            base_url=base_url,
            model_store=model_store,
            runner=runner,
        )
        if not model_matches_manifest(spec.alias, spec.parameters_b, base_url=base_url):
            raise OllamaBootstrapError(
                f"{spec.alias} was created but its parameter size could not be verified."
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
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
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
    ensure_ollama_service(
        executable,
        base_url=base_url,
        model_store=model_store,
    )

    _emit(progress, "models", 25, "Preparing JARVIS Brain models")
    provisioned, skipped = provision_brain_models(
        executable,
        snapshot=snapshot,
        progress=progress,
        base_url=base_url,
        model_store=model_store,
        runner=runner,
    )
    core = next(spec for spec in load_brain_manifest() if spec.role == "coordinator")
    _emit(progress, "verification", 95, "Testing JARVIS Core text generation")
    if not verify_model_inference(core.alias, base_url=base_url):
        raise OllamaBootstrapError(
            "JARVIS Core is installed but its text-generation check failed. "
            "Setup is incomplete; check Ollama and available device memory, then retry."
        )
    _emit(progress, "complete", 100, "JARVIS Brain is ready")

    return BootstrapResult(executable, installed, provisioned, skipped)
