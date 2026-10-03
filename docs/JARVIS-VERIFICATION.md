# JARVIS validation and remaining installation gates

Reviewed on 3 October 2026 against the `product/jarvis-brain` development line.

The current changes repair model-store switching, premature readiness during
setup, Core parameter/inference verification, stale face presence, enrollment
integrity, expired/replayed phone approvals, concurrent approval processing,
secure signing-key persistence, and mobile requests blocked by WebView CSP.
The mobile transport stays native and is restricted to the invitation or saved
desktop origin and fixed companion routes. Redirects and oversized responses
are rejected. The UI reports Core readiness, permits changing model storage after initial
setup, and imports a selected GGUF expert while retaining its own coordinator. An unnecessary lint dependency chain with a high-severity
advisory was removed while retaining React, hooks, TypeScript and accessibility
checks.

## Local verification

- Full Python discovery: **779 passed**, zero failures/errors/skips.
- Actual running Brain HTTP service: **28/28 checks passed**, including
  authentication, malformed input, unavailable hardware, missing model setup,
  stale pairing approval, and setup error recovery.
- TypeScript checking, ESLint, static web production build: passed.
- Native-bridge JavaScript tests: **17 passed**; mobile pairing, chat, pending
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

The packaged Windows updater now stages a pinned GitHub release installer and
an exact previous-build recovery installer, verifies hashes/PE headers, binds
phone approval to the update digest, waits for desktop and Brain shutdown,
and checks the installed binary's embedded version and commit. Failed checks
attempt verified application recovery. Fifty-eight focused repair/update tests
and 26 additional local-control/early-entry tests passed with substitute OS
installer adapters. These establish code behavior, not a physical NSIS upgrade
or complete recovery of unrelated Windows settings.

Confirmed sidecar termination clears its cached connection and allows at most
three replacement starts. UI read recovery is shared and retries once; effectful
POST requests are never replayed. Update handoff pauses recovery. The 17 web
tests include nine recovery cases, and two isolated Rust lifecycle tests passed;
the full native compilation gate remains required.

The public preview publisher requires nine successful checks for the exact
commit, including iOS simulator and scratch architecture checks. It verifies
Windows/Android payload hashes and publishes their installer/APK plus a pinned
update manifest. It does not publish an unsigned simulator build as a phone IPA.

## Real inference after the warm-model fix

The production local-AI job in
[run 37156236899](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/runs/37156236899)
passed for source `3af55b52cb6aedf2f56174e31fbdcfb64f972e43`. It exercises
`JarvisBrain.respond` with the actual Core and a hidden expert, including the
final locally routed answer. The preceding revision's council succeeded but
its final route failed because Ollama returned warm model names with `:latest`.
Canonical alias matching fixed that observed failure. This remains inference
with the existing pretrained Core, not independently trained JARVIS weights.

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

1. Provide the laptop OS and an authorized device connection, or run the
   packaged installer locally. Verify startup, real conversation, chosen model
   storage, restart persistence and practical performance on that machine.
2. Enroll the owner's face and test recognition, blank camera, and an unmatched
   person. Test actual microphone capture, speaker output and interruption.
3. Install the fresh Android build on the real phone, pair over Tailscale, verify
   chat and approve/reject/cancel a supported exact pending action with its
   enrolled fingerprint.
   A dedicated Android fingerprint-sensor path is implemented, with worker-thread
   native verification, bounded retries, timeout and lifecycle cancellation.
   Android-target Rust and real sensor API Kotlin compilation passed; physical
   fingerprint operation still requires the owner device.
4. Independently trained JARVIS model weights are **not supplied**. Core currently
   uses pretrained Llama 3.2 through an Ollama Modelfile. Application routing and
   authority logic have no neural parameter count. A new trained 1.5B JARVIS
   checkpoint needs licensed data, training compute and held-out evaluation.
   The new [scratch pipeline](../training/README.md) defines a
   **1,543,714,304-parameter** architecture, random neural initialization,
   full-weight optimization, private corpus preparation, checkpoint integrity,
   held-out evaluation and inspected GGUF export. Only a pinned text tokenizer
   is reused. Its 18-record CC0 starter corpus is development data, not enough
   to train a useful assistant. The current environment has 8 GiB RAM, no CUDA
   GPU and approximately 24 GiB free storage; its preflight rejects this recipe
   before model allocation. No training job or learned checkpoint exists.
   An isolated CPU PyTorch/Transformers construction on the `meta` device
   verified exactly 1,543,714,304 actual model parameters, all trainable,
   without allocating neural weights. This verifies geometry and library
   compatibility; it does not verify language skill or training throughput.
5. Existing restricted action/Windows broker contracts remain in force. Passing
   regressions does not establish unrestricted desktop automation or feature
   parity for every legacy adapter.
6. iOS source and an unsigned simulator build/launch workflow are supplied. Real
   iPhone/iPad installation still needs Apple signing and device access. Touch ID
   is required by the fingerprint approval policy; Face ID-only devices cannot
   approve through that sensor path. See [iOS details](JARVIS-IOS.md).

The owner supplied a Dell Latitude 3420 (8 GB RAM, 256 GB storage, 11th-generation
Core i5/Xe-class graphics), HP EliteBook 660 G11 (16 GB RAM, 512 GB storage,
Core Ultra 5/Intel graphics), and Honor X9D Android phone. Hardware adaptation
prioritizes Core and limits expert loading under pressure; it cannot guarantee
zero latency or turn these inference devices into full-weight training hardware.
None of these physical devices has been accessed or installed from this workspace.

See [installation instructions](JARVIS-INSTALL.md). Keep PR #25 draft until the
fresh published-revision build/runtime gates pass; keep physical acceptance
explicitly pending until device observations are recorded.
