# JARVIS training from scratch

This is an explicitly invoked training pipeline for a **1,543,714,304-parameter**
JARVIS language model. It defines an architecture and the steps needed to train,
evaluate, and export real weights. The repository does **not** contain a trained
JARVIS checkpoint, and installing this code does not create one. Parameter count
alone does not establish intelligence, useful answers, or safe device control.

The locked Qwen2 architecture starts with random neural weights; all of them are
trainable. It reuses only the text tokenizer from the pinned, Apache-2.0-licensed
`Qwen/Qwen2.5-1.5B-Instruct` revision recorded in
[`architecture.lock.json`](architecture.lock.json). The tokenizer license is
preserved in [`TOKENIZER-LICENSE`](TOKENIZER-LICENSE). Reusing this tokenizer is
disclosed separately from training the language weights from scratch.

The pinned tokenizer has **151,665 distinct text token IDs** (highest ID
151,664), while the neural architecture uses **151,936 embedding rows**. The
271 extra rows are padding in the locked network geometry; they are not extra
tokenizer entries. Preparation, training, and evaluation check the actual text
tokenizer length against 151,665. Keep the padded embedding size unchanged so
the verified neural parameter count stays correct.

## Starter data and corpus rights

[`data/starter.jsonl`](data/starter.jsonl) contains 18 independently authored,
CC0-1.0, synthetic English and Egyptian Arabic examples: six each in `train`,
`validation`, and `test`. Each record includes its source, license, explicit
split, topic group, and review status. These are **development-only examples**
for exercising the pipeline. They are unreviewed by a human and are far too small
to train a broadly useful assistant. They contain no owner recordings, biometric
data, files, or private conversation history.

Topic groups belong to one split. The starter training groups cover temperature
arithmetic, sensitive-action approval, and model memory. Validation groups cover
spatial language, evidence for perception, and summarization. Test groups cover
duration arithmetic, offline capability, and storage selection. Keep related
source documents and paraphrases together when extending the data; do not move
evaluation examples into training to improve scores.

Additional JSONL shards must use the same record schema. A record contains either
`text` or exactly three `messages` with `system`, `user`, and `assistant` roles.
`provenance.kind` must be `authored_synthetic`, `project_documentation`, or
`owner_provided`; permitted licenses are `CC0-1.0`, `Apache-2.0`, `MIT`, and
`LicenseRef-Owner-Private`. Imported material needs its original `source_sha256`.
Owner data requires `owner_authorized: true`; the private owner license is only
valid for that source type. Mark `reviewed: true` only after an actual review.
Do not upload private corpora or model artifacts to GitHub.

Validation rejects duplicate IDs, normalized duplicate prompts, groups crossing
splits, and certain cross-split near duplicates or repeated answers. Preparation
uses a disk-backed index for larger shards. These lexical checks cannot prove
semantic independence, licensing compliance, or dataset quality; review those
properties separately.

## Compute and private storage

Run these commands from the repository root. The validation and preflight tools
use the Python standard library and do not load weights or download anything:

```bash
python -m training.validate
python -m training.preflight --development
```

The full-weight CUDA AdamW recipe requires a GPU with at least 48 GiB VRAM and
36 GiB currently free, at least 16 GiB host/cgroup RAM, and 80 GiB free private
storage for development or 200 GiB for a production run. CPU training is rejected.
This makes full training unsuitable for many laptops; the laptop can later run
a compatible quantized model after genuine training and evaluation elsewhere.
Preflight reports the actual detected limits and missing dependencies.

The architecture lock uses 20 tokens per parameter as a planning heuristic.
Production corpus and processed-token checks enforce that target, reviewed data,
and a run with all weights trained. The FLOP and time figures are estimates, not
measured throughput or guarantees of quality. Benchmark the training hardware
before scheduling a full run.

Use Python 3.11 or 3.12 and a separate virtual environment. Install the correct
CUDA build of PyTorch 2.14.1 using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then
install the pinned dependencies in `training/requirements.txt`, including
Transformers 5.18.0. A training environment is separate from JARVIS's application
dependencies. The optional architecture check below can use the matching CPU
build of PyTorch instead.

All caches, token files, checkpoints, evaluation transcripts, and GGUF output
must stay outside this Git repository. Default paths are
`~/.local/share/jarvis/training` on Linux (or `$XDG_DATA_HOME/jarvis/training`) and
`%LOCALAPPDATA%\JARVIS\training` on Windows. Output directories inside the
repository are rejected even when reached through a symlink. Use a new directory
for corpus preparation and a new run directory unless explicitly resuming.

## Development workflow

The examples below use Bash. On Windows, pass equivalent absolute private paths
to each `--corpus`, `--run`, and `--output` argument.

```bash
JARVIS_TRAIN_ROOT="$HOME/.local/share/jarvis/training"
python -m training.preflight --output "$JARVIS_TRAIN_ROOT/runs" --development --require-ready
python -m training.validate --data training/data/starter.jsonl
python -m training.prepare --data training/data/starter.jsonl --output "$JARVIS_TRAIN_ROOT/starter-corpus" --download-tokenizer
python -m training.train --corpus "$JARVIS_TRAIN_ROOT/starter-corpus" --output "$JARVIS_TRAIN_ROOT/starter-run" --max-steps 2 --sequence-length 64 --batch-size 1 --gradient-accumulation 8 --learning-rate 0.0002 --seed 42 --development
python -m training.evaluate --run "$JARVIS_TRAIN_ROOT/starter-run" --max-batches 2 --allow-development
```

`--download-tokenizer` explicitly permits the pinned tokenizer download, never
language-model weights. Omit it when the pinned tokenizer is already cached for
offline preparation. Preparation writes uint32 little-endian token streams, a
saved tokenizer, the license, and a corpus manifest with source and token hashes.
It performs no neural initialization or training.

Training uses random initialization, CUDA, full-weight AdamW, fixed seed,
gradient accumulation, and the requested step limit. `--checkpoint-every N`
controls periodic recovery checkpoints; `--resume PATH` requests an existing
recovery checkpoint with compatible corpus/configuration evidence. A completed
run records trained checkpoint hashes, parameter counts, steps, and a sampled
nonzero weight update. A development run remains `development_only`; a few
successful optimizer steps do not make its answers useful.

Evaluation verifies the checkpoint and evaluates the held-out test split. It
records loss/perplexity evidence and generated behavioral outputs for review.
Development checkpoints require `--allow-development`. These results are
evidence for inspection, not an intelligence certification. Review generated
responses and conduct broader independent evaluations before product use.
This evaluator requires at least 8 GiB free CUDA memory. Production export also
requires passing all four minimum behavioral probes; passing them is not a
substitute for a wider evaluation.

For a production run, supply a genuinely reviewed, licensed corpus at the scale
reported by preflight, prepare it in a new private directory, and omit
`--development`. Select `--max-steps` and training settings that meet the
processed-token target on the measured hardware. The starter corpus cannot
satisfy production gates. Test data is reserved for final evaluation rather
than repeated training decisions.

## Local GGUF export

Export uses an existing local checkout of trusted `llama.cpp` conversion and
quantization tools. It does not clone tools, fetch weights, or register a model
with Ollama automatically. Run it only after checkpoint verification and
evaluation have completed. Use a clean tracked checkout and a locally built
`llama-quantize` executable for quantized export. The checkout must include
`convert_hf_to_gguf.py` and `gguf-py/gguf`; supported quantizer locations are
`build/bin/llama-quantize`, `build/bin/llama-quantize.exe`, and
`build/bin/Release/llama-quantize.exe`. Install that inspected revision's
documented conversion dependencies in the isolated export environment first;
they are separate from the pinned training requirements:

```bash
python -m training.export --run "$JARVIS_TRAIN_ROOT/starter-run" --llama-cpp /absolute/path/to/llama.cpp --output "$JARVIS_TRAIN_ROOT/starter-export" --quantization Q4_K_M --allow-development
```

Production exports omit `--allow-development`. Export checks the recorded
checkpoint/evaluation hashes before conversion, records tool revision/hashes,
and verifies the resulting GGUF tensor parameter count. Available output choices
are `Q4_K_M`, `Q5_K_M`, `Q8_0`, and `F16`. A resulting GGUF is a model
artifact, not proof of good answers, owner recognition, mobile biometric approval,
or laptop readiness. Development exports retain their development status.
After a suitable model has been evaluated, import its GGUF through JARVIS's
explicit local-model workflow and verify actual Ollama inference on the target
laptop. No training stage grants device-action authority or replaces JARVIS's
owner approval, face-enrollment, or companion-phone checks.

## Offline regression checks

```bash
python -m unittest discover -s tests -p 'test_scratch_training.py' -v
```

These tests use tiny synthetic files and standard-library mocks to check
provenance, split leakage, the locked parameter count, output boundaries, and
checkpoint evidence. They allocate no real model and perform no training,
weight download, GPU inference, or physical laptop/phone verification.

After installing the pinned model libraries in an isolated CPU environment,
optionally check the actual Transformers API and model geometry:

```bash
python -m training.inspect_architecture --output /private/path/meta-report.json
```

Choose your own private output path outside the repository. This check constructs
the model on meta tensors, which have no neural weight storage. It reports actual
library versions and parameter geometry without downloading weights, creating
a checkpoint, training, or running model inference.

The independent CPU workflow
[`scratch-model-contracts.yml`](../.github/workflows/scratch-model-contracts.yml)
runs the offline guards and this real library architecture check and preserves
the meta report as an artifact. Passing that workflow establishes the checked
code and architecture contracts; it does not establish learning quality or
intelligence.
