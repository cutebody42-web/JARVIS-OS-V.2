# JARVIS validation and remaining installation gates

Reviewed on 4 October 2026 against the `product/jarvis-brain` development line.

The current changes repair model-store switching, premature readiness during
setup, Core parameter/inference verification, stale face presence, enrollment
integrity, expired/replayed phone approvals, concurrent approval processing,
secure signing-key persistence, and mobile requests blocked by WebView CSP.
The mobile transport stays native and is restricted to the invitation or saved
desktop origin and fixed companion routes. Redirects and oversized responses
are rejected. The UI reports Core readiness and permits changing model storage
after initial setup. An unnecessary lint dependency chain with a high-severity
advisory was removed while retaining React, hooks, TypeScript and accessibility
checks.

## Local verification

- Full Python discovery: **598 passed**, zero failures/errors/skips.
- Actual running Brain HTTP service: **20/20 checks passed**, including
  authentication, malformed input, unavailable hardware, missing model setup,
  stale pairing approval, and setup error recovery.
- TypeScript checking, ESLint, static web production build: passed.
- Native-bridge JavaScript tests: **4 passed**; mobile pairing, chat, pending
  approvals and decisions use the Rust transport and retain signed-response
  verification.
- `npm audit --audit-level=high`: **zero vulnerabilities**.
- Actual Chromium at 1440×1000 and 390×844: zero page errors or horizontal
  overflow. Desktop UI used the running HTTP Brain; tests exercised invalid
  model-path recovery and unavailable-camera feedback. Mobile showed the
  unpaired state and disabled pairing without available fingerprint support.

The local HTTP/browser probes explicitly inject temporary test secret storage
and a substitute native bridge. They do not establish functioning OS keychains,
real model inference, camera recognition or physical phone fingerprint input.
Native Rust compilation and fresh Windows/Android packages must also pass on the
published revision before it replaces the previous installation candidate.

## Previous candidate evidence

Source `1506249ab113297f117d4f15259e2c4e1a860717` has successful
[Windows installer](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/runs/36863469039),
[Android APK](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/runs/36863468961),
and [real cloud runtime](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/runs/36863469014)
checks. Its evidence establishes Windows VM installation/process startup and
separate packaged Brain health/status, Android emulator launch/deep-link and
screenshot, and overlapping nonempty Ollama inference requests for a 1.2B Core
and a 4.7B expert. These artifacts precede the fixes described above.

Those checks did not exercise real phone fingerprint authentication, owner face
recognition, actual microphone capture, all native UI actions, or the owner's
laptop. Two overlapping inference requests do not prove simultaneous GPU
execution or reasoning quality.

## Still required for the owner's requested completion

1. Provide laptop OS/hardware and an authorized device connection, or run the
   packaged installer locally. Verify startup, real conversation, chosen model
   storage, restart persistence and practical performance on that machine.
2. Enroll the owner's face and test recognition, blank camera, and an unmatched
   person. Test actual microphone capture, speaker output and interruption.
3. Install the fresh Android build on the real phone, pair over Tailscale, verify
   chat and approve/reject/cancel a supported exact pending action with its
   enrolled fingerprint.
   The existing biometric library may accept other Android biometric modalities;
   a dedicated fingerprint-sensor path is being implemented and must be rebuilt
   and tested before claiming fingerprint-only approval.
4. Independently trained JARVIS model weights are **not supplied**. Core currently
   uses pretrained Llama 3.2 through an Ollama Modelfile. Application routing and
   authority logic have no neural parameter count. A new trained 1B JARVIS
   checkpoint needs a training specification, licensed data, compute and
   held-out evaluation; a renamed model cannot satisfy that requirement.
5. Existing restricted action/Windows broker contracts remain in force. Passing
   regressions does not establish unrestricted desktop automation or feature
   parity for every legacy adapter.

See [installation instructions](JARVIS-INSTALL.md). Keep PR #25 draft until the
fresh published-revision build/runtime gates pass; keep physical acceptance
explicitly pending until device observations are recorded.
