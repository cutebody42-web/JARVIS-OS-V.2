"""Dependency-free dataset, provenance and checkpoint checks.

Nothing here reads owner files, downloads models, or grants device authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import unicodedata


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
SPLITS = ("train", "validation", "test")
ALLOWED_LICENSES = {"CC0-1.0", "Apache-2.0", "MIT", "LicenseRef-Owner-Private"}
ALLOWED_SOURCES = {"authored_synthetic", "project_documentation", "owner_provided"}


class PipelineError(ValueError):
    """A check failed before a model can be marked as trained or exported."""


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"Cannot read JSON: {path.name}") from error
    if not isinstance(result, dict):
        raise PipelineError(f"Expected JSON object: {path.name}")
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architecture_parameters(config: dict) -> int:
    """Qwen2 tied embeddings + biased QKV + unbiased attention/MLP outputs."""
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    if hidden % heads or config.get("tie_word_embeddings") is not True or config.get("attention_bias") is not True:
        raise PipelineError("Unsupported scratch architecture parameter assumptions.")
    kv_width = config["num_key_value_heads"] * (hidden // heads)
    attention = 2 * hidden * hidden + 2 * hidden * kv_width + hidden + 2 * kv_width
    mlp = 3 * hidden * config["intermediate_size"]
    norms = 2 * hidden
    return config["vocab_size"] * hidden + config["num_hidden_layers"] * (attention + mlp + norms) + hidden


def architecture_lock() -> dict:
    lock = read_json(ROOT / "architecture.lock.json")
    tokenizer = lock.get("tokenizer", {})
    revision = tokenizer.get("revision") if isinstance(tokenizer, dict) else None
    if lock.get("schema_version") != 1 or not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise PipelineError("Tokenizer must have a pinned commit revision.")
    if tokenizer.get("license") != "apache-2.0" or sha256_file(ROOT / "TOKENIZER-LICENSE") != tokenizer.get("license_sha256"):
        raise PipelineError("The pinned tokenizer license is missing or changed.")
    if architecture_parameters(lock["architecture"]) != lock.get("expected_parameters") or lock["expected_parameters"] < 1_500_000_000:
        raise PipelineError("Scratch architecture must actually reach the verified 1.5B count.")
    return lock


def private_root() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "JARVIS" / "training"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "jarvis" / "training"


def verify_tokenizer_vocabulary(size: int) -> None:
    """The pinned text vocabulary is smaller than the padded embedding matrix."""
    lock = architecture_lock()
    expected = lock["tokenizer"]["vocab_size"]
    if type(size) is not int or size != expected or size > lock["architecture"]["vocab_size"]:
        raise PipelineError("Tokenizer vocabulary differs from the pinned text vocabulary or exceeds model embeddings.")


def private_output(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if result == REPOSITORY or REPOSITORY in result.parents:
        raise PipelineError("Model outputs and caches must be outside the Git repository.")
    return result


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def prompt_text(record: dict) -> str:
    return record["text"] if "text" in record else record["messages"][1]["content"]


def _tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise PipelineError("Expected a dataset record object.")
    for field in ("id", "group_id", "category"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise PipelineError(f"Missing {field} in dataset record.")
    if record.get("split") not in SPLITS:
        raise PipelineError("Every record must declare train, validation or test.")
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in ALLOWED_SOURCES:
        raise PipelineError("Missing recognized data provenance.")
    if provenance.get("license") not in ALLOWED_LICENSES:
        raise PipelineError("Dataset license is missing or unsupported.")
    if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise PipelineError("Dataset source must be explicit.")
    if not isinstance(provenance.get("reviewed"), bool):
        raise PipelineError("Every record must declare its review status.")
    if provenance["kind"] == "owner_provided" and provenance.get("owner_authorized") is not True:
        raise PipelineError("Owner-provided data needs recorded owner authorization.")
    if provenance.get("license") == "LicenseRef-Owner-Private" and provenance["kind"] != "owner_provided":
        raise PipelineError("The private owner license is only for authorized owner data.")
    source_digest = provenance.get("source_sha256")
    if provenance["kind"] != "authored_synthetic" and (
        not isinstance(source_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", source_digest)
    ):
        raise PipelineError("Imported source material must include its provenance SHA-256.")
    if ("text" in record) == ("messages" in record):
        raise PipelineError("Use either a text document or messages, not both.")
    if "text" in record:
        contents = [record["text"]]
    else:
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise PipelineError("Use one system, user, assistant training turn per record.")
        if [item.get("role") if isinstance(item, dict) else None for item in messages] != ["system", "user", "assistant"]:
            raise PipelineError("Expected system, user, assistant message order.")
        contents = [message.get("content") for message in messages]
    for content in contents:
        if not isinstance(content, str) or not content.strip() or len(content) > 20_000:
            raise PipelineError("Invalid or excessive message content.")
        if any(marker in content for marker in ("<|im_start|>", "<|im_end|>")):
            raise PipelineError("Raw chat control tokens are not accepted in data.")
    if not normalize(prompt_text(record)):
        raise PipelineError("Dataset prompt/document has no usable text.")


def iter_records(paths: list[Path]):
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise PipelineError(f"Invalid JSON in {path.name} at line {line_number}.") from error
                    validate_record(record)
                    yield record
        except (OSError, UnicodeError) as error:
            raise PipelineError(f"Dataset is unreadable: {path.name}.") from error


def load_dataset(path: Path) -> list[dict]:
    """Require declared splits and rights; reject duplicate/near-duplicate prompts.

    Topic families belong to one split. Common system instructions are excluded
    from leakage checks. Semantic leakage still requires a human dataset review.
    """
    records: list[dict] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    group_splits: dict[str, str] = {}
    for record in iter_records([path]):
        if record["id"] in ids:
            raise PipelineError("Duplicate dataset ID.")
        ids.add(record["id"])
        split = record.get("split")
        if split not in SPLITS:
            raise PipelineError("Every record must declare train, validation or test.")
        previous = group_splits.setdefault(record["group_id"], split)
        if previous != split:
            raise PipelineError("A topic/source group crosses dataset splits.")
        prompt = normalize(prompt_text(record))
        if not prompt or prompt in prompts:
            raise PipelineError("Duplicate normalized dataset prompt.")
        prompts.add(prompt)
        record_tokens = _tokens(prompt)
        answer = normalize(record["messages"][-1]["content"]) if "messages" in record else ""
        for previous_record in records:
            if previous_record["split"] == split:
                continue
            previous_tokens = _tokens(prompt_text(previous_record))
            union = record_tokens | previous_tokens
            if len(union) >= 8 and len(record_tokens & previous_tokens) / len(union) >= 0.8:
                raise PipelineError("Near-duplicate prompt leaks across splits.")
            previous_answer = normalize(previous_record["messages"][-1]["content"]) if "messages" in previous_record else ""
            if len(answer.split()) >= 8 and answer == previous_answer:
                raise PipelineError("Duplicate answer leaks across splits.")
        records.append(record)
    if not records or any(not any(item["split"] == split for item in records) for split in SPLITS):
        raise PipelineError("The dataset needs nonempty train, validation and test splits.")
    return records


def dataset_summary(records: list[dict]) -> dict:
    return {
        "split_counts": {split: sum(item["split"] == split for item in records) for split in SPLITS},
        "categories": sorted({item["category"] for item in records}),
        "unreviewed_records": sum(not item["provenance"]["reviewed"] for item in records),
        "synthetic_records": sum(item["provenance"]["kind"] == "authored_synthetic" for item in records),
        "note": "Lexical leakage checks do not establish semantic independence or data quality.",
    }


def safetensor_parameters(path: Path) -> int:
    """Inspect actual tensor shapes without importing torch or reading weights."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise PipelineError("Truncated safetensors file.")
        size = struct.unpack("<Q", prefix)[0]
        if size > 16 * 1024 * 1024 or size < 2:
            raise PipelineError("Invalid safetensors header size.")
        raw_header = handle.read(size)
        if len(raw_header) != size:
            raise PipelineError("Truncated safetensors header.")
        try:
            header = json.loads(raw_header)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise PipelineError("Invalid safetensors header.") from error
    if not isinstance(header, dict):
        raise PipelineError("Invalid safetensors tensor map.")
    total = 0
    ranges = []
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
              "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    payload_size = path.stat().st_size - 8 - size
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise PipelineError("Invalid safetensors tensor shape.")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise PipelineError("Invalid safetensors tensor offsets.")
        if not 0 <= offsets[0] <= offsets[1] <= payload_size:
            raise PipelineError("Safetensors offsets point outside the actual checkpoint.")
        count = math.prod(shape)
        width = widths.get(tensor.get("dtype"))
        if width is None or offsets[1] - offsets[0] != count * width:
            raise PipelineError("Safetensors tensor shape/dtype does not match actual weight bytes.")
        ranges.append(tuple(offsets))
        total += count
    position = 0
    for start, end in sorted(ranges):
        if start != position:
            raise PipelineError("Safetensors tensor offsets overlap or leave gaps.")
        position = end
    if position != payload_size:
        raise PipelineError("Safetensors tensor bytes do not cover the checkpoint payload.")
    return total


def verify_checkpoint(run: Path, *, allow_development: bool = False, allow_partial: bool = False) -> dict:
    manifest = read_json(run / "training-manifest.json")
    lock = architecture_lock()
    allowed_statuses = {"trained", "development_only"} if allow_development else {"trained"}
    if allow_partial:
        allowed_statuses.add("partial_training")
    if manifest.get("status") not in allowed_statuses or manifest.get("global_steps", 0) <= 0:
        raise PipelineError("This run does not contain a completed training result.")
    if manifest.get("initialization") != "random_from_config" or manifest.get("pretrained_language_weights_loaded") is not False:
        raise PipelineError("This is not a recorded training run from scratch.")
    if manifest.get("parameters") != lock["expected_parameters"] or manifest.get("trainable_parameters") != lock["expected_parameters"]:
        raise PipelineError("Not all verified 1.5B parameters were trainable.")
    if manifest.get("sampled_update_nonzero") is not True:
        raise PipelineError("There is no recorded nonzero training update.")
    checkpoints = manifest.get("checkpoint_files")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise PipelineError("Actual trained checkpoint files are missing.")
    total = 0
    for name, digest in checkpoints.items():
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise PipelineError("Invalid checkpoint filename.")
        path = run / "checkpoint" / name
        if sha256_file(path) != digest:
            raise PipelineError("Trained weights changed after the recorded run.")
        total += safetensor_parameters(path)
    if total != lock["expected_parameters"]:
        raise PipelineError("Actual checkpoint tensor count does not match the 1.5B architecture.")
    config = read_json(run / "checkpoint" / "config.json")
    for field, expected in lock["architecture"].items():
        if field != "use_cache" and config.get(field) != expected:
            raise PipelineError(f"Checkpoint architecture changed: {field}.")
    return manifest
