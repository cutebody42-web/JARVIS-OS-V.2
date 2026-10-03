"""Export verified scratch-trained weights with existing local llama.cpp tools.

No model downloads, tool checkout, Ollama registration or automatic activation.
Development exports remain explicitly development-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from training.common import PipelineError, architecture_lock, private_output, sha256_file, verify_checkpoint, write_json
from training.evaluate import verify_evaluation


def inspect_gguf(path: Path, llama_cpp: Path) -> dict:
    """Use the converter's GGUF reader to inspect real tensor byte mappings."""
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise PipelineError("The converter output is not a GGUF file.")
    script = (
        "import json,sys; sys.path.insert(0,sys.argv[2]); "
        "from gguf import GGUFReader; reader=GGUFReader(sys.argv[1]); "
        "print(json.dumps({'parameters':sum(int(t.n_elements) for t in reader.tensors),"
        "'tensor_count':len(reader.tensors)}))"
    )
    result = subprocess.run([sys.executable, "-c", script, str(path), str(llama_cpp / "gguf-py")],
                            check=False, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise PipelineError("The local GGUF reader could not verify the exported tensors; check llama.cpp conversion dependencies.")
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PipelineError("The GGUF inspector returned invalid metadata.") from error
    if not isinstance(metadata, dict) or metadata.get("parameters") != architecture_lock()["expected_parameters"]:
        raise PipelineError("The actual GGUF tensor parameter count differs from the trained 1.5B checkpoint.")
    metadata["sha256"] = sha256_file(path)
    metadata["file_bytes"] = path.stat().st_size
    return metadata


def run_checked(command: list[str], log: Path) -> None:
    with log.open("wb") as handle:
        result = subprocess.run(command, check=False, stdout=handle, stderr=subprocess.STDOUT, timeout=14400)
    if result.returncode:
        raise PipelineError(f"The local conversion tool failed; inspect {log.name} in the private export directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantization", choices=("Q4_K_M", "Q5_K_M", "Q8_0", "F16"), default="Q4_K_M")
    parser.add_argument("--allow-development", action="store_true")
    args = parser.parse_args()
    try:
        run, output = private_output(args.run), private_output(args.output)
        manifest = verify_checkpoint(run, allow_development=args.allow_development)
        evaluation = verify_evaluation(run, manifest, allow_development=args.allow_development)
        tools = args.llama_cpp.expanduser().resolve(strict=True)
        converter = tools / "convert_hf_to_gguf.py"
        if not converter.is_file() or not (tools / "gguf-py" / "gguf").is_dir():
            raise PipelineError("Choose an existing local llama.cpp checkout with converter and GGUF reader.")
        quantizers = [tools / "build" / "bin" / name for name in ("llama-quantize", "llama-quantize.exe")]
        quantizers.append(tools / "build" / "bin" / "Release" / "llama-quantize.exe")
        quantizer = next((path for path in quantizers if path.is_file()), None)
        if args.quantization != "F16" and quantizer is None:
            raise PipelineError("Build the local llama-quantize executable before quantized export.")
        revision = subprocess.check_output(["git", "-C", str(tools), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        dirty = subprocess.check_output(["git", "-C", str(tools), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=10).strip()
        if dirty:
            raise PipelineError("The llama.cpp tracked sources are modified; use an inspected clean checkout for reproducible conversion.")
        if output.exists():
            raise PipelineError("Choose a new private export directory; existing artifacts are preserved.")
        output.mkdir(parents=True, mode=0o700)
        development = manifest.get("development_only") is True
        name = f"jarvis-scratch-1_5b-{args.quantization.lower()}" + ("-development" if development else "") + ".gguf"
        final = output / name
        intermediate = output / "weights-f16.gguf"
        report = {
            "status": "incomplete", "development_only": development,
            "training_manifest_sha256": sha256_file(run / "training-manifest.json"),
            "evaluation_sha256": sha256_file(run / "evaluation.json"),
            "checkpoint_files": manifest["checkpoint_files"], "llama_cpp_revision": revision,
            "converter_sha256": sha256_file(converter), "quantization": args.quantization,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "behavioral_passes": sum(item["passed"] for item in evaluation["behavioral_probes"]),
            "quality_certified": False, "activated_in_jarvis": False,
        }
        write_json(output / "export-manifest.json", report)
        run_checked([sys.executable, str(converter), str(run / "checkpoint"), "--outfile", str(intermediate), "--outtype", "f16"], output / "converter.log")
        inspect_gguf(intermediate, tools)
        if args.quantization == "F16":
            intermediate.rename(final)
        else:
            report["quantizer_sha256"] = sha256_file(quantizer)
            run_checked([str(quantizer), str(intermediate), str(final), args.quantization], output / "quantizer.log")
        report.update(status="exported", gguf_file=final.name, gguf=inspect_gguf(final, tools))
        write_json(output / "export-manifest.json", report)
        if intermediate.exists():
            intermediate.unlink()
    except (PipelineError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Export did not complete: {error}\n")
    print(f"Exported {final}; import and actual device inference remain explicit separate steps.")


if __name__ == "__main__":
    main()
