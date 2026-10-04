"""Private evidence for an explicitly activated, independently trained Core.

Receipts attest to checks performed at activation time, not language quality.
The runtime binds them to the current Ollama alias digest and exact tensor count.
No training dependencies, model download or device authority are involved here.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
import uuid

from core.app_paths import user_data_dir


CORE_ALIAS = "jarvis-core-1b"  # Stable routing alias; its name is not a size claim.
SCRATCH_PARAMETERS = 1_543_714_304
SCRATCH_CONTEXT = 4096
_HASH = re.compile(r"[a-f0-9]{64}")
_RECEIPT_LIMIT = 65_536


class ScratchActivationError(ValueError):
    pass


def activation_endpoint(value: str) -> str:
    parsed = urlparse(value)
    try:
        port = parsed.port or 11434
    except ValueError as error:
        raise ScratchActivationError("Activation requires a valid literal loopback Ollama endpoint.") from error
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
            or parsed.port == 0):
        raise ScratchActivationError("Activation requires a literal HTTP loopback Ollama endpoint.")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{port}"


def receipt_path(base_url: str, *, directory: Path | None = None) -> Path:
    endpoint = activation_endpoint(base_url)
    root = directory if directory is not None else user_data_dir() / "scratch-core-activations"
    return root / (hashlib.sha256(endpoint.encode()).hexdigest() + ".json")


def activation_lock_path(base_url: str, *, directory: Path | None = None) -> Path:
    return receipt_path(base_url, directory=directory).with_suffix(".lock")


def assert_activation_unlocked(base_url: str, *, directory: Path | None = None) -> None:
    path = activation_lock_path(base_url, directory=directory)
    if path.exists() or path.is_symlink():
        raise ScratchActivationError(
            "Core activation lock is present. Keep JARVIS closed until activation finishes. "
            "After a crash, inspect the private lock and runtime before removing it manually."
        )


@contextmanager
def activation_lock(base_url: str, *, directory: Path | None = None):
    path = activation_lock_path(base_url, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = uuid.uuid4().hex
    content = json.dumps({"schema_version": 1, "endpoint": activation_endpoint(base_url),
                          "pid": os.getpid(), "token": token}) + "\n"
    lease = {"release": True}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        assert_activation_unlocked(base_url, directory=directory)
        raise ScratchActivationError("Core activation is already locked.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        yield lease
    finally:
        # Never remove another command's lock, even after an unexpected change.
        try:
            if lease["release"] and not path.is_symlink() and path.read_text("utf-8") == content:
                path.unlink()
        except OSError:
            pass


def validate_receipt(receipt: dict, base_url: str) -> dict:
    if not isinstance(receipt, dict):
        raise ScratchActivationError("Scratch Core activation receipt must be an object.")
    expected = {"schema_version": 1, "state": "activated", "alias": CORE_ALIAS,
                "endpoint": activation_endpoint(base_url), "parameters": SCRATCH_PARAMETERS,
                "trainable_parameters": SCRATCH_PARAMETERS, "context_length": SCRATCH_CONTEXT,
                "development_only": False, "initialization": "random_from_config",
                "pretrained_language_weights_loaded": False, "quality_review_acknowledged": True,
                "runtime_smoke_passed": True, "quality_certified": False}
    if any(type(receipt.get(key)) is not type(value) or receipt.get(key) != value
           for key, value in expected.items()):
        raise ScratchActivationError("Scratch Core activation receipt has incompatible production evidence.")
    for key in ("ollama_digest", "gguf_sha256", "training_manifest_sha256", "evaluation_sha256",
                "export_manifest_sha256", "architecture_lock_sha256", "core_template_sha256"):
        if not isinstance(receipt.get(key), str) or not _HASH.fullmatch(receipt[key]):
            raise ScratchActivationError("Scratch Core activation receipt has invalid evidence hashes.")
    if not isinstance(receipt.get("activated_at"), str) or not receipt["activated_at"].strip():
        raise ScratchActivationError("Scratch Core activation receipt has no activation time.")
    return receipt


def read_receipt(base_url: str, *, directory: Path | None = None) -> dict | None:
    path = receipt_path(base_url, directory=directory)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _RECEIPT_LIMIT:
            raise ScratchActivationError("Scratch Core activation receipt is not a bounded regular file.")
        result = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScratchActivationError("Scratch Core activation receipt is unreadable; refusing Core replacement.") from error
    return validate_receipt(result, base_url)


def write_receipt(receipt: dict, base_url: str, *, directory: Path | None = None) -> Path:
    validate_receipt(receipt, base_url)
    return _atomic_private_json(receipt, base_url, directory=directory)


def write_recovery_marker(base_url: str, backup_alias: str | None, *, directory: Path | None = None) -> Path:
    """Persist a deliberately non-ready state if canonical rollback fails."""
    return _atomic_private_json({"schema_version": 1, "state": "recovery_required",
        "endpoint": activation_endpoint(base_url), "alias": CORE_ALIAS,
        "backup_alias": backup_alias, "reason": "Activation rollback could not be verified. Keep JARVIS closed and inspect the backup."},
        base_url, directory=directory)


def restore_receipt_snapshot(previous: dict | None, attempted: dict | None, base_url: str,
                             *, directory: Path | None = None) -> None:
    """Restore only this locked transaction's receipt, then verify the result."""
    current = read_receipt(base_url, directory=directory)
    if current == previous:
        return
    if attempted is None or current != attempted:
        raise ScratchActivationError("Activation receipt changed unexpectedly during rollback.")
    if previous is None:
        receipt_path(base_url, directory=directory).unlink()
    else:
        write_receipt(previous, base_url, directory=directory)
    if read_receipt(base_url, directory=directory) != previous:
        raise ScratchActivationError("The previous activation receipt could not be restored.")


def _atomic_private_json(receipt: dict, base_url: str, *, directory: Path | None = None) -> Path:
    path = receipt_path(base_url, directory=directory)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".activation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def runtime_model_identity(model: str, base_url: str, *, request_get, request_post) -> tuple[str, dict]:
    """Read exact alias digest and actual model geometry from this local daemon."""
    endpoint = activation_endpoint(base_url)
    try:
        response = request_get(endpoint + "/api/tags", timeout=10)
        response.raise_for_status()
        body = response.json()
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            raise ValueError("invalid model list")
        matches = [item for item in models if isinstance(item, dict)
                   and item.get("name", item.get("model")) in {model, model + ":latest"}]
        if len(matches) != 1 or not isinstance(matches[0].get("digest"), str):
            raise ValueError("alias digest missing or ambiguous")
        digest = matches[0]["digest"].removeprefix("sha256:")
        if not _HASH.fullmatch(digest):
            raise ValueError("invalid digest")
        response = request_post(endpoint + "/api/show", json={"model": model}, timeout=10)
        response.raise_for_status()
        metadata = response.json()
        info = metadata.get("model_info") if isinstance(metadata, dict) else None
        count = info.get("general.parameter_count") if isinstance(info, dict) else None
        capabilities = metadata.get("capabilities") if isinstance(metadata, dict) else None
        if type(count) is not int or count != SCRATCH_PARAMETERS:
            raise ValueError("actual parameter count differs")
        if not isinstance(capabilities, list) or "completion" not in capabilities:
            raise ValueError("text completion capability missing")
        return digest, metadata
    except Exception as error:
        raise ScratchActivationError("Cannot verify the scratch Core's exact Ollama digest and parameter count.") from error


def verify_receipt_runtime(receipt: dict, base_url: str, *, request_get, request_post) -> None:
    validate_receipt(receipt, base_url)
    digest, _ = runtime_model_identity(CORE_ALIAS, base_url, request_get=request_get, request_post=request_post)
    if digest != receipt["ollama_digest"]:
        raise ScratchActivationError("Scratch Core alias differs from its activation receipt; refusing pretrained fallback.")
