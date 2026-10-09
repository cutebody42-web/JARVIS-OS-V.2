# Owner-local wake-word activation

NEXUS can use a local wake-word detector as an **activation signal only**. A wake-word hit does not authorize an action, change policy, or bypass the Sovereign Owner Kernel / Action Gateway.

## External runtime research

The adapter was designed against `dscripka/openWakeWord` commit `368c03716d1e92591906a84949bc477f3a834455` (Apache-2.0), release line 0.6.x. Upstream uses 16 kHz PCM audio and its examples commonly process 1280-sample chunks. The ONNX path uses three model artifacts:

- the chosen wake-word model (for example an owner-provided `wakeword.onnx`)
- `melspectrogram.onnx`
- `embedding_model.onnx`

JARVIS does **not** call openWakeWord's model downloader and does not contain a model URL. All three files must already exist locally.

## Model directory

A minimal owner-managed directory is:

```text
wakeword-model/
├── wakeword.onnx
├── melspectrogram.onnx
└── embedding_model.onnx
```

Construct the detector with `WakeWordModelFiles.from_directory(...)`. The files are resolved and validated before inference. Paths must remain inside the configured directory, must be regular `.onnx` files, and must be non-empty.

## Audio contract

`LocalWakeWordDetector.process()` accepts mono signed 16-bit PCM samples at 16 kHz. It is intentionally **not** a microphone owner. The UI/runtime decides when microphone capture is enabled and feeds bounded frames to the detector. Push-to-talk therefore remains available even when wake-word activation is disabled or unavailable.

The detector accepts between 10 ms and 5 seconds per call; a typical streaming frame is 1280 samples (~80 ms).

## Privacy and authority

- no model download
- no HTTP client or cloud audio upload
- no background microphone loop in the detector itself
- no shell/process execution
- no permissions or policy mutation
- detection result is only `{detected, score, model_name, threshold}`

Any command produced after activation must still pass through the existing owner verification and action policy path.

## Optional runtime install

```bash
pip install -r requirements-local-wakeword.txt
```

The base installer is intentionally not forced to carry this optional runtime until the product-level microphone UX and hardware validation gate are enabled.

## Upstream update policy

Review newer openWakeWord releases manually. Do not automatically merge upstream runtime/model-download behavior. Re-check license, dependency changes, model/resource expectations, audio contracts, and offline behavior before updating the pinned optional runtime version.
