"""First-run provisioning for the local JARVIS Brain.

The desktop installer bundles this module, so the owner never needs to install
Python or run pip. Ollama is installed separately through its official Windows
package when the owner explicitly approves local-AI setup.

All subprocess calls use argument arrays; no shell or generated command text is
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from tempfile import TemporaryDirectory
from typing import Callable, Iterable
from urllib.parse import urlparse

import requests

from core.app_paths import resource_path
from core.hardware_profile import HardwareSnapshot
from core.scratch_activation import (
    CORE_ALIAS, ScratchActivationError, assert_activation_unlocked,
    read_receipt, verify_receipt_runtime,
)


Progress = Callable[[str, float, str], None]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

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
        or parsed.hostname not in {"127.0.0.1", "::1"}
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
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    env["OLLAMA_HOST"] = f"{host}:{port}"
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
    base_digest: str
    weights_digest: str
    modelfile_sha256: str
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

    if value.get("schema_version") != 2 or not isinstance(value.get("models"), list):
        raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")

    specs = []
    for item in value["models"]:
        if not isinstance(item, dict):
            raise OllamaBootstrapError("JARVIS Brain model manifest is invalid.")
        alias = item.get("alias")
        base = item.get("base")
        base_digest = item.get("base_digest")
        weights_digest = item.get("weights_digest")
        modelfile_sha256 = item.get("modelfile_sha256")
        role = item.get("role")
        min_ram = item.get("min_ram_gb")
        parameters_b = item.get("parameters_b")
        if (
            not isinstance(alias, str)
            or not isinstance(base, str)
            or not isinstance(base_digest, str)
            or _SHA256_RE.fullmatch(base_digest) is None
            or not isinstance(weights_digest, str)
            or _SHA256_RE.fullmatch(weights_digest) is None
            or not isinstance(modelfile_sha256, str)
            or _SHA256_RE.fullmatch(modelfile_sha256) is None
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
        try:
            # Git may materialize CRLF on Windows and LF in release archives.
            # Pin semantic UTF-8 content so trusted provenance is stable across
            # those equivalent checkouts, while every other byte still binds.
            canonical_modelfile = (
                modelfile.read_text("utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .encode("utf-8")
            )
            actual_modelfile_sha256 = hashlib.sha256(canonical_modelfile).hexdigest()
        except (OSError, UnicodeError) as exc:
            raise OllamaBootstrapError(f"Modelfile for {alias} is unreadable.") from exc
        if actual_modelfile_sha256 != modelfile_sha256:
            raise OllamaBootstrapError(f"Modelfile integrity check failed for {alias}.")
        specs.append(
            BrainModelSpec(
                alias,
                base,
                base_digest,
                weights_digest,
                modelfile_sha256,
                role,
                float(min_ram),
                float(parameters_b),
                modelfile,
            )
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
            encoding="utf-8",
            errors="replace",
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
            encoding="utf-8",
            errors="replace",
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


def _parameter_size_matches(body: object, expected_parameters_b: float) -> bool:
    try:
        details = body.get("details") if isinstance(body, dict) else None
        label = details.get("parameter_size") if isinstance(details, dict) else None
        actual = parameter_size_b(label)
    except (AttributeError, TypeError, ValueError):
        return False
    # The human-readable size is rounded, so allow a small difference from the
    # advertised family size. A sub-billion model cannot satisfy a 1B core.
    if expected_parameters_b >= 1.0 and actual < 1.0:
        return False
    tolerance = max(0.15, expected_parameters_b * 0.20)
    return abs(actual - expected_parameters_b) <= tolerance


def model_matches_manifest(
    model: str,
    expected_parameters_b: float,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    request_post=requests.post,
    request_get=requests.get,
) -> bool:
    if model.removesuffix(":latest") == CORE_ALIAS:
        try:
            assert_activation_unlocked(base_url)
            receipt = read_receipt(base_url)
            if receipt is not None:
                verify_receipt_runtime(receipt, base_url,
                    request_get=request_get, request_post=request_post)
                return True
        except (ScratchActivationError, OSError):
            return False
    try:
        response = request_post(
            base_url + "/api/show",
            json={"model": model},
            timeout=10,
        )
        response.raise_for_status()
        body = response.json()
    except Exception:
        return False
    return _parameter_size_matches(body, expected_parameters_b)


def _installed_model_digest(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    request_get=requests.get,
) -> str | None:
    """Return one exact local Ollama manifest digest, never a fuzzy name match."""
    try:
        response = request_get(base_url + "/api/tags", timeout=10)
        response.raise_for_status()
        body = response.json()
        values = body.get("models") if isinstance(body, dict) else None
        if not isinstance(values, list):
            return None
        accepted = {model}
        if ":" not in model:
            accepted.add(model + ":latest")
        matches = [
            item for item in values
            if isinstance(item, dict)
            and item.get("name", item.get("model")) in accepted
        ]
        if len(matches) != 1:
            return None
        digest = matches[0].get("digest")
        if not isinstance(digest, str):
            return None
        digest = digest.removeprefix("sha256:")
        return digest if _SHA256_RE.fullmatch(digest) else None
    except Exception:
        return None


def _modelfile_expectations(path: Path) -> tuple[str, dict[str, str]]:
    text = path.read_text("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    match = re.search(r'^SYSTEM\s+"""\n(.*?)\n"""\s*$', text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise OllamaBootstrapError(f"Modelfile for {path.stem} has no canonical SYSTEM block.")
    system = "\n" + match.group(1) + "\n"
    parameters: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("PARAMETER "):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or parts[1] in parameters:
            raise OllamaBootstrapError(f"Modelfile for {path.stem} has invalid parameters.")
        parameters[parts[1]] = parts[2].strip()
    if not parameters:
        raise OllamaBootstrapError(f"Modelfile for {path.stem} has no parameters.")
    return system, parameters


def model_matches_pinned_spec(
    spec: BrainModelSpec,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    request_post=requests.post,
    request_get=requests.get,
) -> bool:
    """Verify weights, inherited prompt surface, and JARVIS customization.

    Mutable registry tags and parameter-size-only checks are insufficient for
    a trusted local brain. The shipped manifest pins the registry manifest,
    tensor layer and Modelfile. A deliberately activated scratch Core is the
    sole exception; its private receipt already binds its exact alias digest.
    """
    if spec.alias == CORE_ALIAS:
        try:
            assert_activation_unlocked(base_url)
            receipt = read_receipt(base_url)
            if receipt is not None:
                verify_receipt_runtime(
                    receipt,
                    base_url,
                    request_get=request_get,
                    request_post=request_post,
                )
                return True
        except (ScratchActivationError, OSError):
            return False

    if _installed_model_digest(
        spec.base, base_url=base_url, request_get=request_get,
    ) != spec.base_digest:
        return False
    if _installed_model_digest(
        spec.alias, base_url=base_url, request_get=request_get,
    ) is None:
        return False

    try:
        alias_response = request_post(
            base_url + "/api/show", json={"model": spec.alias}, timeout=10,
        )
        alias_response.raise_for_status()
        alias_body = alias_response.json()
        base_response = request_post(
            base_url + "/api/show", json={"model": spec.base}, timeout=10,
        )
        base_response.raise_for_status()
        base_body = base_response.json()
        if not isinstance(alias_body, dict) or not isinstance(base_body, dict):
            return False
        if not _parameter_size_matches(alias_body, spec.parameters_b):
            return False
        details = alias_body.get("details")
        if not isinstance(details, dict) or details.get("parent_model") != spec.base:
            return False
        capabilities = alias_body.get("capabilities")
        if not isinstance(capabilities, list) or "completion" not in capabilities:
            return False
        # Template and licenses must be inherited verbatim from the pinned base.
        if alias_body.get("template") != base_body.get("template"):
            return False
        if alias_body.get("license") != base_body.get("license"):
            return False
        expected_system, expected_parameters = _modelfile_expectations(spec.modelfile)
        system = alias_body.get("system")
        if not isinstance(system, str) or system.replace("\r\n", "\n") != expected_system:
            return False
        raw_parameters = alias_body.get("parameters")
        if not isinstance(raw_parameters, str):
            return False
        actual_parameters: dict[str, str] = {}
        for line in raw_parameters.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                actual_parameters[parts[0]] = parts[1].strip()
        if any(actual_parameters.get(key) != value for key, value in expected_parameters.items()):
            return False
        marker = "sha256-" + spec.weights_digest
        if marker not in str(alias_body.get("modelfile", "")):
            return False
        if marker not in str(base_body.get("modelfile", "")):
            return False
    except Exception:
        return False
    return True


_IMPORT_ALIAS_RE = re.compile(r"jarvis-import-[a-z0-9][a-z0-9-]{0,63}")


def import_gguf_model(
    executable: str,
    source_path: str | Path,
    *,
    alias: str | None = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_store: str | Path | None = None,
    runner=subprocess.run,
    request_post=requests.post,
) -> str:
    """Import an owner-selected GGUF file into an already ensured runtime.

    This creates an optional local text expert. It neither replaces JARVIS Core
    nor downloads or executes model weights. Ollama validates the full model;
    the initial file check is only a bounded header/regular-file check.
    """
    _ollama_environment(base_url, model_store)
    if not isinstance(source_path, (str, Path)):
        raise OllamaBootstrapError("Local model file must be an absolute GGUF path.")
    source = Path(source_path).expanduser()
    if not source.is_absolute() or source.suffix.lower() != ".gguf":
        raise OllamaBootstrapError("Local model file must be an absolute GGUF path.")
    try:
        info = source.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise OllamaBootstrapError("Choose a regular GGUF file, not a folder or symbolic link.")
        source = source.resolve(strict=True)
        # Ollama expands glob patterns even inside quoted FROM paths. Reject
        # those characters so it cannot import a different file by expansion.
        if any(not char.isprintable() or char in '"*?[]' for char in source.as_posix()):
            raise OllamaBootstrapError("The model path contains unsupported characters; rename the file or folder.")
        with source.open("rb") as handle:
            if info.st_size < 24 or handle.read(4) != b"GGUF":
                raise OllamaBootstrapError("The selected file does not contain a valid GGUF header.")
    except (OSError, ValueError) as exc:
        raise OllamaBootstrapError("The selected GGUF file could not be read.") from exc

    if alias is None:
        stem = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")[:32] or "model"
        identity = f"{source}:{info.st_size}:{info.st_mtime_ns}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        alias = f"jarvis-import-{stem}-{digest}"
    if not isinstance(alias, str) or not _IMPORT_ALIAS_RE.fullmatch(alias):
        raise OllamaBootstrapError("Imported model alias must use the jarvis-import- prefix and lowercase letters, digits or hyphens.")

    with TemporaryDirectory(prefix="jarvis-gguf-") as directory:
        modelfile = Path(directory) / "Modelfile"
        modelfile.write_text(
            f'FROM "{source.as_posix()}"\n'
            'SYSTEM "You are a hidden local JARVIS specialist. Provide factual text analysis; do not claim external actions occurred."\n',
            encoding="utf-8",
        )
        _run_ollama(
            executable, ["create", alias, "-f", str(modelfile)],
            base_url=base_url, model_store=model_store, runner=runner,
        )

    try:
        response = request_post(base_url + "/api/show", json={"model": alias}, timeout=10)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("invalid metadata")
        details = body.get("details")
        try:
            count = parameter_size_b(details.get("parameter_size") if isinstance(details, dict) else None)
        except ValueError:
            model_info = body.get("model_info")
            raw_count = model_info.get("general.parameter_count") if isinstance(model_info, dict) else None
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
                raise ValueError("parameter count unavailable")
            count = raw_count / 1_000_000_000
        if not math.isfinite(count) or count <= 0:
            raise ValueError("invalid parameter count")
        capabilities = body.get("capabilities")
        if capabilities is not None and (
            not isinstance(capabilities, list) or "completion" not in capabilities
        ):
            raise ValueError("text completion unsupported")
    except Exception as exc:
        raise OllamaBootstrapError(
            "Ollama created the imported alias, but a text-capable model with a positive parameter count could not be verified."
        ) from exc
    return alias


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
            encoding="utf-8",
            errors="replace",
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
    # A deliberate scratch activation is never silently replaced with baseline
    # weights, even when its evidence is damaged or its daemon/store changed.
    try:
        assert_activation_unlocked(base_url)
        activation = read_receipt(base_url)
        if activation is not None:
            verify_receipt_runtime(activation, base_url,
                request_get=requests.get, request_post=requests.post)
    except (ScratchActivationError, OSError) as error:
        raise OllamaBootstrapError(f"Scratch Core activation requires inspection: {error}") from error
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
        if spec.alias == CORE_ALIAS:
            try:
                assert_activation_unlocked(base_url)
                current_activation = read_receipt(base_url)
                if activation is not None and current_activation is None:
                    raise ScratchActivationError("The active scratch Core receipt disappeared; refusing baseline replacement.")
                if current_activation is not None:
                    verify_receipt_runtime(current_activation, base_url,
                        request_get=requests.get, request_post=requests.post)
                    completed_aliases.append(spec.alias)
                    _emit(progress, "models", base_percent, f"{spec.alias} verified scratch activation ready")
                    continue
            except (ScratchActivationError, OSError) as error:
                raise OllamaBootstrapError(f"Scratch Core activation requires inspection: {error}") from error
        if spec.alias in existing and model_matches_pinned_spec(
            spec,
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
        if _installed_model_digest(spec.base, base_url=base_url) != spec.base_digest:
            raise OllamaBootstrapError(
                f"{spec.base} does not match JARVIS's pinned model provenance. "
                "The mutable registry tag changed; review and approve a manifest update before use."
            )
        if spec.alias == CORE_ALIAS:
            # Recheck after a potentially long pull, before changing the alias.
            try:
                assert_activation_unlocked(base_url)
                if read_receipt(base_url) is not None:
                    raise ScratchActivationError("Core activation changed while provisioning; retry without replacing it.")
            except (ScratchActivationError, OSError) as error:
                raise OllamaBootstrapError(f"Scratch Core activation requires inspection: {error}") from error
        _run_ollama(
            executable,
            ["create", spec.alias, "-f", str(spec.modelfile)],
            base_url=base_url,
            model_store=model_store,
            runner=runner,
        )
        if not model_matches_pinned_spec(spec, base_url=base_url):
            raise OllamaBootstrapError(
                f"{spec.alias} was created but its pinned provenance could not be verified."
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
