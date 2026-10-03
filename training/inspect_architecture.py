"""Check the actual Transformers model geometry on meta tensors, without weights.

This validates library compatibility and parameter geometry. Meta tensors have
no neural weight storage; this script never trains, saves a checkpoint, loads
pretrained weights, or certifies useful model behavior.
"""

import argparse
import json
from pathlib import Path

from training.common import PipelineError, architecture_lock, private_output, write_json


def inspect() -> dict:
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    lock = architecture_lock()
    with torch.device("meta"):
        model = Qwen2ForCausalLM(Qwen2Config(**lock["architecture"]))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    meta_only = all(parameter.device.type == "meta" for parameter in model.parameters())
    if not meta_only or parameters != lock["expected_parameters"] or trainable != parameters:
        raise PipelineError("The actual installed model library does not match the locked trainable meta architecture.")
    return {
        "parameters": parameters, "trainable_parameters": trainable,
        "all_neural_tensors_on_meta": meta_only,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "tokenizer_vocab_size": lock["tokenizer"]["vocab_size"],
        "padded_embedding_rows": lock["architecture"]["vocab_size"],
        "neural_weights_allocated": False, "pretrained_weights_loaded": False,
        "training_performed": False, "checkpoint_saved": False, "intelligence_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect()
        if args.output:
            path = private_output(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, report)
    except (PipelineError, ImportError) as error:
        parser.exit(1, f"Meta architecture inspection failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
