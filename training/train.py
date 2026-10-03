"""Train every JARVIS 1.5B neural parameter from random initialization on CUDA.

No pretrained language weights are downloaded or used. The pinned tokenizer is
prepared separately. Small runs require --development and cannot be exported
as a completed production training run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
from pathlib import Path
import shutil
import struct

from training.common import (
    PipelineError, ROOT, SPLITS, architecture_lock, private_output, private_root,
    read_json, sha256_file, verify_checkpoint, verify_tokenizer_vocabulary, write_json,
)
from training.preflight import inspect


def verify_corpus(corpus: Path) -> dict:
    lock = architecture_lock()
    manifest = read_json(corpus / "corpus-manifest.json")
    if manifest.get("architecture_lock_sha256") != sha256_file(ROOT / "architecture.lock.json"):
        raise PipelineError("Corpus was prepared for another architecture lock.")
    if manifest.get("tokenizer") != lock["tokenizer"] or manifest.get("token_dtype") != "uint32-little-endian":
        raise PipelineError("Corpus tokenizer or token encoding differs from the locked recipe.")
    for split in SPLITS:
        path = corpus / f"{split}.tokens"
        count = manifest.get("token_counts", {}).get(split)
        digest = manifest.get("token_files", {}).get(split)
        if type(count) is not int or count <= 0 or path.stat().st_size != count * 4:
            raise PipelineError(f"Invalid {split} token count or actual token file size.")
        if sha256_file(path) != digest:
            raise PipelineError(f"Prepared {split} token data changed after preparation.")
    return manifest


def training_policy(manifest: dict, *, development: bool, max_steps: int,
                    sequence_length: int, batch_size: int, accumulation: int) -> None:
    lock = architecture_lock()
    if any(type(value) is not int or value <= 0 for value in (max_steps, sequence_length, batch_size, accumulation)):
        raise PipelineError("Steps, sequence length, batch and accumulation must be positive integers.")
    if sequence_length < 2 or sequence_length > lock["architecture"]["max_position_embeddings"]:
        raise PipelineError("Sequence length must fit the locked architecture and include a next-token target.")
    counts = manifest.get("token_counts", {})
    if any(type(counts.get(split)) is not int or counts[split] < sequence_length for split in SPLITS):
        raise PipelineError("Every split needs at least one full sequence before model allocation.")
    if counts["train"] < sequence_length * batch_size:
        raise PipelineError("The training split needs at least one complete batch before model allocation.")
    if not development:
        target = lock["expected_parameters"] * lock["planning_tokens_per_parameter"]
        if manifest.get("unreviewed_records") != 0:
            raise PipelineError("Production training requires a reviewed corpus.")
        if manifest.get("token_counts", {}).get("train", 0) < target:
            raise PipelineError("The corpus is below the planning token target; use --development for a small pipeline run.")
        if max_steps * sequence_length * batch_size * accumulation < target:
            raise PipelineError("The planned optimizer updates are below the training token target.")


class TokenBlocks:
    """Memory-map private token shards without reading the full corpus into RAM."""

    def __init__(self, path: Path, sequence_length: int, vocab_size: int):
        self.file = path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.length = len(self.mapping) // (4 * sequence_length)
        if self.length == 0:
            self.close()
            raise PipelineError("Each split needs at least one full sequence; reduce the sequence length for development data.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import torch
        start = index * self.sequence_length * 4
        ids = struct.unpack(f"<{self.sequence_length}I", self.mapping[start:start + self.sequence_length * 4])
        if any(token >= self.vocab_size for token in ids):
            raise PipelineError("Prepared tokens exceed the locked model vocabulary.")
        return torch.tensor(ids, dtype=torch.long)

    def close(self):
        self.mapping.close()
        self.file.close()


def sampled_weights(model) -> list[float]:
    return next(model.parameters()).detach().flatten()[:32].float().cpu().tolist()


def sample_digest(sample: list[float]) -> str:
    return hashlib.sha256(json.dumps(sample, allow_nan=False).encode()).hexdigest()


def save_run(model, tokenizer, optimizer, output: Path, manifest: dict, torch) -> None:
    checkpoint = output / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(checkpoint)
    shutil.copyfile(ROOT / "TOKENIZER-LICENSE", checkpoint / "TOKENIZER-LICENSE")
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, output / "optimizer.pt")
    manifest["checkpoint_files"] = {path.name: sha256_file(path) for path in sorted(checkpoint.glob("*.safetensors"))}
    manifest["optimizer_sha256"] = sha256_file(output / "optimizer.pt")
    manifest["checkpoint_config_sha256"] = sha256_file(checkpoint / "config.json")
    write_json(output / "training-manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=private_root() / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        output, corpus = private_output(args.output), private_output(args.corpus)
        lock = architecture_lock()
        data = verify_corpus(corpus)
        training_policy(data, development=args.development, max_steps=args.max_steps,
                        sequence_length=args.sequence_length, batch_size=args.batch_size,
                        accumulation=args.gradient_accumulation)
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.checkpoint_every <= 0:
            raise PipelineError("Learning rate and checkpoint interval must be positive.")
        if output.exists():
            raise PipelineError("Choose a new private output directory; existing runs are preserved.")
        preflight = inspect(output, development=args.development)
        if not preflight["eligible_for_this_recipe"]:
            print(json.dumps(preflight, indent=2))
            raise PipelineError("Training preflight failed before model allocation.")
    except (PipelineError, OSError) as error:
        parser.exit(1, f"Training did not start: {error}\n")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable in the installed PyTorch; no model was allocated.\n")
    eligible_devices = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        if total / 2**30 >= lock["minimum_gpu_memory_gib"] and free / 2**30 >= 36:
            eligible_devices.append((free, index))
    if not eligible_devices:
        parser.exit(1, "No CUDA-visible GPU meets this recipe's actual memory budget; no model was allocated.\n")
    torch.cuda.set_device(max(eligible_devices)[1])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(corpus / "tokenizer", local_files_only=True, trust_remote_code=False)
    verify_tokenizer_vocabulary(len(tokenizer))
    resumed = None
    if args.resume:
        resume = private_output(args.resume)
        resumed = verify_checkpoint(resume, allow_development=args.development, allow_partial=True)
        if resumed.get("corpus_manifest_sha256") != sha256_file(corpus / "corpus-manifest.json"):
            parser.exit(1, "Resume requires the same verified corpus.\n")
        if resumed["global_steps"] >= args.max_steps:
            parser.exit(1, "Resume max-steps must exceed the recorded completed updates.\n")
        model = Qwen2ForCausalLM.from_pretrained(resume / "checkpoint", local_files_only=True, torch_dtype=torch.float32)
    else:
        # Constructor from configuration creates random weights. It never calls
        # from_pretrained or a language-model download for a new scratch run.
        model = Qwen2ForCausalLM(Qwen2Config(**lock["architecture"]))
    model.requires_grad_(True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != lock["expected_parameters"] or trainable != parameters:
        parser.exit(1, "The allocated model does not have all verified 1.5B weights trainable.\n")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    initial_step = resumed["global_steps"] if resumed else 0
    seen = resumed.get("tokens_seen", 0) if resumed else 0
    if resumed:
        if sha256_file(resume / "optimizer.pt") != resumed.get("optimizer_sha256"):
            parser.exit(1, "Resume optimizer state changed after the checkpoint.\n")
        state = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    dataset = TokenBlocks(corpus / "train.tokens", args.sequence_length, len(tokenizer))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    if len(loader) == 0:
        dataset.close()
        parser.exit(1, "The training split is smaller than one complete batch.\n")
    iterator = iter(loader)
    before = sampled_weights(model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
    output.mkdir(parents=True, mode=0o700)
    manifest = {
        "schema_version": 1, "model_name": lock["model_name"], "status": "training",
        "initialization": "random_from_config", "pretrained_language_weights_loaded": False,
        "parameters": parameters, "trainable_parameters": trainable,
        "tokenizer": lock["tokenizer"], "architecture_lock_sha256": sha256_file(ROOT / "architecture.lock.json"),
        "corpus": str(corpus), "corpus_manifest_sha256": sha256_file(corpus / "corpus-manifest.json"),
        "global_steps": initial_step, "tokens_seen": seen, "seed": args.seed,
        "sequence_length": args.sequence_length, "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate,
        "development_only": args.development, "resumed_from": str(args.resume) if args.resume else None,
        "sampled_initial_weights_sha256": sample_digest(before), "sampled_update_nonzero": False,
        "hardware": preflight, "started_at": datetime.now(timezone.utc).isoformat(),
        "quality_certified": False, "behavior_evaluated": False,
    }
    write_json(output / "training-manifest.json", manifest)
    model.train()
    try:
        for step in range(initial_step, args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(args.gradient_accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = batch.to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(input_ids=batch, labels=batch).loss
                if not torch.isfinite(loss).item():
                    raise PipelineError("Training loss became nonfinite; the run is incomplete.")
                loss_sum += loss.detach().float().item()
                scaler.scale(loss / args.gradient_accumulation).backward()
                seen += batch.numel()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm).item():
                raise PipelineError("Training gradients became nonfinite; the run is incomplete.")
            scaler.step(optimizer)
            scaler.update()
            after = sampled_weights(model)
            manifest.update(global_steps=step + 1, tokens_seen=seen,
                            last_train_loss=loss_sum / args.gradient_accumulation,
                            sampled_update_nonzero=before != after,
                            sampled_final_weights_sha256=sample_digest(after))
            print(json.dumps({"step": step + 1, "tokens_seen": seen, "train_loss": manifest["last_train_loss"]}), flush=True)
            if (step + 1) % args.checkpoint_every == 0 and step + 1 < args.max_steps:
                manifest["status"] = "development_only" if args.development else "partial_training"
                save_run(model, tokenizer, optimizer, output, manifest, torch)
        if not manifest["sampled_update_nonzero"]:
            raise PipelineError("No nonzero neural-weight update was observed.")
        target = lock["expected_parameters"] * lock["planning_tokens_per_parameter"]
        if not args.development and seen < target:
            raise PipelineError("Actual processed training tokens did not reach the production planning target.")
        manifest["status"] = "development_only" if args.development else "trained"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_run(model, tokenizer, optimizer, output, manifest, torch)
        verify_checkpoint(output, allow_development=args.development)
    finally:
        dataset.close()
    print(f"Saved actual trained weights in {output}; quality and deployment remain unevaluated.")


if __name__ == "__main__":
    main()
