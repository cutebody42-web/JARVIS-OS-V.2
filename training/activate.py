"""Explicitly promote a verified production scratch export to JARVIS Core.

JARVIS must stay closed. An already running local Ollama service is required;
this command never starts services or downloads pretrained weights. A successful
smoke check proves text generation only, not language quality or device safety.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
import uuid

import requests

from core.app_paths import resource_path
from core.ollama_bootstrap import _run_ollama, find_ollama_executable, verify_model_inference
from core.scratch_activation import (
    CORE_ALIAS, SCRATCH_CONTEXT, SCRATCH_PARAMETERS, ScratchActivationError,
    activation_endpoint, activation_lock, read_receipt, runtime_model_identity,
    restore_receipt_snapshot, verify_receipt_runtime, write_receipt, write_recovery_marker,
)
from training.common import (
    PipelineError, ROOT, architecture_lock, private_output, read_json,
    sha256_file, verify_checkpoint,
)
from training.evaluate import PROBES, verify_evaluation
from training.train import training_policy, verify_corpus


def regular_private_file(path: Path, root: Path) -> Path:
    try:
        if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
            raise PipelineError("Activation inputs must be regular files without symbolic links.")
        result = path.resolve(strict=True)
        result.relative_to(root)
        relative = path.relative_to(root)
        for parent in relative.parents:
            if (root / parent).is_symlink():
                raise PipelineError("Activation input directories cannot be symbolic links.")
        return result
    except (OSError, ValueError) as error:
        raise PipelineError("Activation input is missing, unreadable, or outside its private directory.") from error


def verify_production_export(run_path: Path, export_path: Path) -> dict:
    """Recheck real checkpoint/corpus/evaluation bytes and export evidence."""
    if not run_path.is_absolute() or not export_path.is_absolute():
        raise PipelineError("Activation requires absolute private run and export directories.")
    run, export = private_output(run_path), private_output(export_path)
    if not run.is_dir() or not export.is_dir():
        raise PipelineError("Choose existing private run and export directories.")
    for name in ("training-manifest.json", "evaluation.json"):
        regular_private_file(run / name, run)
        if (run / name).stat().st_size > 4 * 1024 * 1024:
            raise PipelineError("Activation manifests must be bounded JSON files.")
    regular_private_file(export / "export-manifest.json", export)
    if (export / "export-manifest.json").stat().st_size > 4 * 1024 * 1024:
        raise PipelineError("Activation manifests must be bounded JSON files.")
    lock = architecture_lock()
    recorded = read_json(run / "training-manifest.json")
    checkpoints = recorded.get("checkpoint_files")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise PipelineError("Production checkpoint files are unavailable.")
    for name in checkpoints:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise PipelineError("Production checkpoint filenames are invalid.")
        regular_private_file(run / "checkpoint" / name, run)
    regular_private_file(run / "checkpoint" / "config.json", run)
    manifest = verify_checkpoint(run)
    if manifest.get("development_only") is not False:
        raise PipelineError("Only explicitly production scratch runs can replace Core.")
    if manifest.get("architecture_lock_sha256") != sha256_file(ROOT / "architecture.lock.json"):
        raise PipelineError("The production run belongs to another architecture lock.")
    if manifest.get("tokenizer") != lock["tokenizer"]:
        raise PipelineError("The production run tokenizer differs from the pinned recipe.")
    for name in manifest["checkpoint_files"]:
        regular_private_file(run / "checkpoint" / name, run)
    regular_private_file(run / "checkpoint" / "config.json", run)
    corpus = private_output(manifest["corpus"])
    regular_private_file(corpus / "corpus-manifest.json", corpus)
    for split in ("train", "validation", "test"):
        regular_private_file(corpus / f"{split}.tokens", corpus)
    data = verify_corpus(corpus)
    if sha256_file(corpus / "corpus-manifest.json") != manifest.get("corpus_manifest_sha256"):
        raise PipelineError("The production corpus changed after training.")
    training_policy(data, development=False, max_steps=manifest.get("global_steps"),
        sequence_length=manifest.get("sequence_length"), batch_size=manifest.get("batch_size"),
        accumulation=manifest.get("gradient_accumulation"))
    target = lock["expected_parameters"] * lock["planning_tokens_per_parameter"]
    if type(manifest.get("tokens_seen")) is not int or manifest["tokens_seen"] < target:
        raise PipelineError("Recorded actual training tokens did not reach the production planning target.")
    evaluation = verify_evaluation(run, manifest)
    if evaluation.get("development_only") is not False or evaluation.get("split") != "test":
        raise PipelineError("Activation requires production held-out test evaluation.")
    report = read_json(export / "export-manifest.json")
    if report.get("status") != "exported" or report.get("development_only") is not False:
        raise PipelineError("Activation requires a completed production export, never development weights.")
    evidence = {"training_manifest_sha256": sha256_file(run / "training-manifest.json"),
                "evaluation_sha256": sha256_file(run / "evaluation.json")}
    if any(report.get(key) != value for key, value in evidence.items()):
        raise PipelineError("Export evidence belongs to another training run or evaluation.")
    if report.get("checkpoint_files") != manifest["checkpoint_files"] or report.get("behavioral_passes") != len(PROBES):
        raise PipelineError("Export checkpoint or minimum behavioral evidence differs.")
    name = report.get("gguf_file")
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix.lower() != ".gguf":
        raise PipelineError("Export has no safe local GGUF filename.")
    source = regular_private_file(export / name, export)
    # Ollama interprets glob syntax even in quoted FROM paths.
    if any(not char.isprintable() or char in '"*?[]' for char in source.as_posix()):
        raise PipelineError("The GGUF path contains unsupported Modelfile characters.")
    with source.open("rb") as handle:
        if source.stat().st_size < 24 or handle.read(4) != b"GGUF":
            raise PipelineError("Export artifact has no valid GGUF header.")
    gguf = report.get("gguf")
    if (not isinstance(gguf, dict) or type(gguf.get("parameters")) is not int
            or gguf["parameters"] != SCRATCH_PARAMETERS or gguf.get("file_bytes") != source.stat().st_size
            or gguf.get("sha256") != sha256_file(source)):
        raise PipelineError("Exported GGUF bytes/hash/count differ from the recorded production artifact.")
    return {"source": source, "gguf_sha256": gguf["sha256"], **evidence,
            "export_manifest_sha256": sha256_file(export / "export-manifest.json"),
            "architecture_lock_sha256": manifest["architecture_lock_sha256"]}


def core_modelfile(source: Path) -> tuple[str, str, str]:
    template = resource_path("models", f"{CORE_ALIAS}.Modelfile")
    text = template.read_text("utf-8")
    # Use checked-in JARVIS policy, never an old alias's inherited Llama template.
    if len(re.findall(r"(?im)^FROM\s+", text)) != 1 or len(re.findall(r"(?im)^PARAMETER\s+num_ctx\s+", text)) != 1:
        raise PipelineError("The checked-in Core Modelfile has an unexpected structure.")
    system = re.search(r'(?ms)^SYSTEM\s+"""\s*\n(.*?)\n"""\s*$', text)
    if system is None:
        raise PipelineError("The checked-in Core policy prompt could not be preserved.")
    generated = re.sub(r"(?im)^FROM[^\r\n]*", lambda _: f'FROM "{source.as_posix()}"', text, count=1)
    generated = re.sub(r"(?im)^PARAMETER\s+num_ctx\s+[^\r\n]*", f"PARAMETER num_ctx {SCRATCH_CONTEXT}", generated, count=1)
    return generated, system.group(1).strip(), sha256_file(template)


def assert_jarvis_closed(brain_url: str | None = None, *, process_iter=None) -> None:
    """Reject known desktop/sidecar processes; optionally guard a known listener."""
    import psutil
    iterator = process_iter or psutil.process_iter
    try:
        for process in iterator(["pid", "name"]):
            if process.info.get("pid") == os.getpid():
                continue
            name = (process.info.get("name") or "").casefold()
            if name in {"jarvis", "jarvis.exe", "jarvis-shell", "jarvis-shell.exe"} or name.startswith("jarvis-brain"):
                raise PipelineError("JARVIS desktop/Brain is still running; close it before activation.")
            if name.startswith("python"):
                try:
                    arguments = process.cmdline()
                except psutil.NoSuchProcess:
                    continue
                if any(Path(argument).name == "brain_sidecar.py" or "api.jarvis_local_server" in argument for argument in arguments):
                    raise PipelineError("A JARVIS Brain process is still running; close it before activation.")
    except (psutil.AccessDenied, psutil.Error) as error:
        raise PipelineError("Could not inspect JARVIS process state; activation is refused.") from error
    if brain_url is not None:
        origin = activation_endpoint(brain_url)
        parsed = urlparse(origin)
        try:
            connection = socket.create_connection((parsed.hostname, parsed.port), timeout=1)
        except ConnectionRefusedError:
            return
        except OSError as error:
            raise PipelineError("Brain listener closure could not be verified; activation is refused.") from error
        else:
            connection.close()
            raise PipelineError("The configured Brain endpoint is still listening; close it before activation.")


def _tags(base_url: str, request_get) -> list[dict]:
    response = request_get(base_url + "/api/tags", timeout=10)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("models"), list) or len(body["models"]) > 4096:
        raise PipelineError("Existing Ollama service returned invalid model metadata.")
    return body["models"]


def _alias_digest(alias: str, base_url: str, request_get) -> str | None:
    matches = [item for item in _tags(base_url, request_get) if isinstance(item, dict)
               and item.get("name", item.get("model")) in {alias, alias + ":latest"}]
    if not matches:
        return None
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str):
        raise PipelineError("Ollama alias digest is absent or ambiguous.")
    digest = matches[0]["digest"].removeprefix("sha256:")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise PipelineError("Ollama alias digest is invalid.")
    return digest


def _unload(alias: str, base_url: str, request_get, request_post) -> None:
    response = request_post(base_url + "/api/generate",
        json={"model": alias, "stream": False, "keep_alive": 0}, timeout=180)
    response.raise_for_status()
    response = request_get(base_url + "/api/ps", timeout=10)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("models"), list):
        raise PipelineError("Ollama resident models could not be verified after unload.")
    if any(isinstance(item, dict) and item.get("name", item.get("model")) in {alias, alias + ":latest"}
           for item in body["models"]):
        raise PipelineError("Core alias is still resident; activation refuses a stale runtime cache.")


def _verify_candidate(alias: str, base_url: str, system: str, request_get, request_post) -> str:
    digest, metadata = runtime_model_identity(alias, base_url,
        request_get=request_get, request_post=request_post)
    info = metadata["model_info"]
    if (info.get("general.architecture") != "qwen2" or type(info.get("qwen2.context_length")) is not int
            or info["qwen2.context_length"] != SCRATCH_CONTEXT):
        raise PipelineError("Ollama scratch architecture/context differs from the locked export.")
    if metadata.get("system", "").strip() != system:
        raise PipelineError("The staged Core policy prompt was not preserved.")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, str):
        raise PipelineError("The staged Core context parameters are unavailable.")
    context = re.findall(r"(?m)^num_ctx\s+(\d+)\s*$", parameters)
    if context != [str(SCRATCH_CONTEXT)]:
        raise PipelineError("The staged Core context limit was not preserved.")
    template = metadata.get("template")
    if not isinstance(template, str) or "<|im_start|>" not in template or "<|start_header_id|>" in template:
        raise PipelineError("The staged model has no compatible Qwen chat template.")
    if not verify_model_inference(alias, base_url=base_url, request_post=request_post):
        raise PipelineError("The scratch Core did not produce actual bounded text generation.")
    return digest


def activate(run: Path, export: Path, *, acknowledge_quality_review: bool,
             acknowledge_jarvis_closed: bool, ollama_url: str = "http://127.0.0.1:11435",
             brain_url: str | None = None, executable: str | None = None,
             runner=subprocess.run, request_get=requests.get, request_post=requests.post,
             receipt_directory: Path | None = None, closed_check=assert_jarvis_closed) -> dict:
    if not acknowledge_quality_review or not acknowledge_jarvis_closed:
        raise PipelineError("Explicit quality-review and JARVIS-closed acknowledgments are required.")
    base_url = activation_endpoint(ollama_url)
    if urlparse(base_url).hostname != "127.0.0.1":
        raise PipelineError("The activation CLI requires the desktop's literal 127.0.0.1 Ollama endpoint.")
    closed_check(brain_url)
    evidence = verify_production_export(run, export)
    generated, system, template_digest = core_modelfile(evidence["source"])
    executable = executable or find_ollama_executable()
    if not executable:
        raise PipelineError("Install Ollama separately and start the intended local model store first.")
    nonce = uuid.uuid4().hex[:16]
    staged, backup = f"jarvis-scratch-stage-{nonce}", f"jarvis-core-backup-{nonce}"
    promoted = False
    backup_created = False
    existed = False
    receipt_committed = False
    receipt = None
    def command(*arguments):
        _run_ollama(executable, list(arguments), base_url=base_url, runner=runner)
    with activation_lock(base_url, directory=receipt_directory) as lease:
        closed_check(brain_url)
        previous_digest = _alias_digest(CORE_ALIAS, base_url, request_get)
        existed = previous_digest is not None
        previous_receipt = read_receipt(base_url, directory=receipt_directory)
        if previous_receipt is not None:
            verify_receipt_runtime(previous_receipt, base_url, request_get=request_get, request_post=request_post)
        try:
            with TemporaryDirectory(prefix="jarvis-core-activation-") as temporary:
                modelfile = Path(temporary) / "Modelfile"
                modelfile.write_text(generated, "utf-8")
                command("create", staged, "-f", str(modelfile))
            staged_digest = _verify_candidate(staged, base_url, system, request_get, request_post)
            if sha256_file(evidence["source"]) != evidence["gguf_sha256"]:
                raise PipelineError("GGUF bytes changed while staging; Core was not promoted.")
            _unload(staged, base_url, request_get, request_post)
            closed_check(brain_url)
            if existed:
                command("cp", CORE_ALIAS, backup)
                backup_created = True
                if _alias_digest(backup, base_url, request_get) != previous_digest:
                    raise PipelineError("Backup digest differs from the original Core; refusing promotion.")
                _unload(CORE_ALIAS, base_url, request_get, request_post)
            # Mark an attempted promotion before the CLI returns: even a failed
            # or timed-out copy may have already changed the canonical alias.
            promoted = True
            lease["release"] = False
            command("cp", staged, CORE_ALIAS)
            digest = _verify_candidate(CORE_ALIAS, base_url, system, request_get, request_post)
            if digest != staged_digest:
                raise PipelineError("Canonical Core digest differs from the verified staged model.")
            _unload(CORE_ALIAS, base_url, request_get, request_post)
            receipt = {"schema_version": 1, "state": "activated", "alias": CORE_ALIAS,
                "endpoint": base_url, "parameters": SCRATCH_PARAMETERS,
                "trainable_parameters": SCRATCH_PARAMETERS, "context_length": SCRATCH_CONTEXT,
                "development_only": False, "initialization": "random_from_config",
                "pretrained_language_weights_loaded": False, "quality_review_acknowledged": True,
                "runtime_smoke_passed": True, "quality_certified": False,
                "ollama_digest": digest, **{key: value for key, value in evidence.items() if key != "source"},
                "core_template_sha256": template_digest,
                "activated_at": datetime.now(timezone.utc).isoformat(), "backup_alias": backup if backup_created else None}
            path = write_receipt(receipt, base_url, directory=receipt_directory)
            receipt_committed = True
            lease["release"] = True
        except BaseException as error:
            if promoted and receipt is not None:
                # os.replace may have committed before write_receipt returns
                # (including a Ctrl+C during finalization). Keep a completely
                # verified new receipt/model pair rather than restoring only
                # the alias and leaving contradictory evidence behind.
                try:
                    fully_committed = (read_receipt(base_url, directory=receipt_directory) == receipt
                        and _alias_digest(CORE_ALIAS, base_url, request_get) == receipt["ollama_digest"])
                except Exception:
                    fully_committed = False
                if fully_committed:
                    lease["release"] = True
                    receipt_committed = True
                    if not isinstance(error, Exception):
                        raise
                    raise PipelineError("Scratch Core and receipt were committed; inspect them before restarting JARVIS after interrupted finalization.") from error
            if promoted:
                try:
                    if _alias_digest(CORE_ALIAS, base_url, request_get) is not None:
                        _unload(CORE_ALIAS, base_url, request_get, request_post)
                    if backup_created:
                        if _alias_digest(backup, base_url, request_get) != previous_digest:
                            raise PipelineError("The recovery backup digest changed.")
                        command("cp", backup, CORE_ALIAS)
                    else:
                        command("rm", CORE_ALIAS)
                    if _alias_digest(CORE_ALIAS, base_url, request_get) != previous_digest:
                        raise PipelineError("The prior canonical Core digest/absence was not restored.")
                    restore_receipt_snapshot(previous_receipt, receipt, base_url, directory=receipt_directory)
                    lease["release"] = True
                except BaseException as rollback_error:
                    try:
                        write_recovery_marker(base_url, backup if backup_created else None, directory=receipt_directory)
                        lease["release"] = True
                    except BaseException:
                        # Retain the cooperative lock if even the recovery
                        # marker cannot be persisted. Never silently fall back.
                        lease["release"] = False
                    if not isinstance(rollback_error, Exception):
                        raise
                    if not isinstance(error, Exception):
                        raise error
                    raise PipelineError(
                        f"Activation failed and rollback needs manual inspection. Keep JARVIS closed; backup alias: {backup if backup_created else 'none'}."
                    ) from rollback_error
            if not isinstance(error, Exception):
                raise
            raise PipelineError("Activation failed; the prior Core selection was preserved/restored. Inspect the cause before retrying.") from error
        finally:
            try:
                command("rm", staged)
            except Exception:
                pass  # A leftover staging alias never authorizes Core replacement.
    if not receipt_committed:
        raise PipelineError("No scratch Core activation receipt was committed.")
    return {"activated": True, "alias": CORE_ALIAS, "parameters": SCRATCH_PARAMETERS,
            "receipt": str(path), "backup_alias": receipt["backup_alias"], "quality_certified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--brain-url", help="Optional known Brain origin; any listening service there blocks activation.")
    parser.add_argument("--ollama-executable")
    parser.add_argument("--acknowledge-quality-review", action="store_true")
    parser.add_argument("--acknowledge-jarvis-closed", action="store_true")
    args = parser.parse_args()
    try:
        report = activate(args.run, args.export, ollama_url=args.ollama_url, brain_url=args.brain_url,
            executable=args.ollama_executable, acknowledge_quality_review=args.acknowledge_quality_review,
            acknowledge_jarvis_closed=args.acknowledge_jarvis_closed)
    except (PipelineError, ScratchActivationError, OSError, requests.RequestException, KeyError, TypeError) as error:
        parser.exit(1, f"Core activation did not complete: {error}\n")
    print(f"Activated verified scratch Core: {report['alias']} ({report['parameters']} parameters).")
    print(f"Private receipt: {report['receipt']}. Restart the updated JARVIS application; quality remains uncertified.")


if __name__ == "__main__":
    main()
