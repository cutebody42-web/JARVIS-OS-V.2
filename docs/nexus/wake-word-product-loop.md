# Wake-word product loop

The desktop product can keep an owner-enabled, **local-only** wake-word loop active while JARVIS is running. This is an activation surface, not an authority surface.

## Lifecycle

1. JARVIS starts with wake-word microphone capture **off**, including after a restart. A previously saved model folder does not reopen the microphone automatically.
2. The owner configures an absolute local model folder containing `wakeword.onnx`, `melspectrogram.onnx`, and `embedding_model.onnx`.
3. The owner explicitly clicks **Enable microphone** in the Local Wake Word control.
4. The Tauri WebView captures one mono microphone stream and converts it locally to signed 16-bit 16 kHz PCM.
5. Bounded 1280-sample chunks are posted only to the bearer-token protected loopback JARVIS Brain endpoint.
6. `WakeWordSession` feeds those frames to the owner-local ONNX detector. It applies a cooldown so one utterance cannot create a burst of activations.
7. On activation, the WebView microphone is closed before the existing Windows System.Speech one-shot dictation runtime opens the default audio device.
8. The transcript is submitted through the normal `/v1/message` Brain path. Any action still uses the existing Owner Kernel, Action Gateway, receipts, and approval rules.
9. JARVIS speaks the reply locally. If the owner has not disabled wake-word mode, the WebView reopens the wake stream after speech completes.

## Privacy

- no cloud audio upload
- no wake-model download
- no wake microphone in the Python detector itself
- only loopback bearer-authenticated PCM frames
- bounded in-browser queue; old chunks are dropped instead of allowing unbounded latency/memory growth
- capture stops on disable, component unmount, activation, or inference error
- server restart always requires a fresh owner enable action

## Authority

A detector result carries `authority: activation_only`. It cannot call tools, approve a request, change policy, mutate permissions, or bypass owner verification. The product bridge deliberately routes the resulting transcript into the same Brain endpoint used by typed input.

## Model/runtime setup

Install the optional local runtime with:

```bash
pip install -r requirements-local-wakeword.txt
```

Then configure the owner-local model folder in the desktop Local Wake Word control. The folder is validated before enablement.

## Hardware acceptance

Cloud CI can verify API authentication, state transitions, bounded PCM handling, UI type/build integration, Windows packaging, and the deterministic detector contracts. It cannot certify the target laptop microphone, WebView2 permission prompt, acoustic false-positive/false-negative rates, room noise performance, or physical speaker-to-microphone feedback behavior. Those remain physical acceptance checks on the target Dell/HP hardware.
