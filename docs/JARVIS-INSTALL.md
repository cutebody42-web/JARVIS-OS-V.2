# Install JARVIS on your laptop and Android phone

The `main` branch contains the consolidated packaged local JARVIS application.
The Windows installer includes its Python Brain runtime; **you do not need Python,
pip, a Gemini key, or a source checkout** to use this version. The Android app is
a companion to the laptop Brain, so the laptop must stay running for phone chat.

## Download a validated public preview

When all nine checks pass for the current source commit, the preview publisher
creates a prerelease on the repository's [Releases page](https://github.com/cutebody42-web/JARVIS-OS-V.2/releases).
Choose the newest **JARVIS preview**, expand **Assets**, and download
`JARVIS-Setup.exe` for the Windows laptop and `JARVIS-Companion.apk` for the
Android phone. These public release files do not require signing in to GitHub
or extracting an Actions artifact ZIP.

If you downloaded the source ZIP, double-click `Install-JARVIS.cmd` in the
extracted repository's main folder. It downloads the newest published preview,
checks its release binding, file size and SHA-256, and opens the standard Windows
installer. An older source ZIP will not gain new files automatically: download
the latest ZIP from `main`, or download `JARVIS-Setup.exe` directly above.
The older `scripts/setup_jarvis.bat` starts the legacy Python application; use
the packaged installer for the local JARVIS product described in this guide.

The tag has the form `jarvis-preview-<commit-prefix>`. Its `jarvis-update.json`
pins the full source commit, Windows installer SHA-256, changed paths, and major
update classification. The release notes link the successful checks for that
same commit. A published preview still requires acceptance on your physical
devices and does not contain independently trained JARVIS weights.

The nine checks include the existing product/build/runtime and regression gates,
an unsigned iOS simulator compile/install/launch check, and CPU contract checks
for the scratch-model architecture and training guards. The public preview
assets are Windows and Android only.

## Laptop: install and initialize

1. Run the downloaded `JARVIS-Setup.exe`, finish the installer, and open **JARVIS**
   from the Start menu.
2. On the **Initialize local intelligence** screen, choose your model folder
   using **Choose folder**, or leave it blank to use Ollama's default storage.
   The folder format is explained below.
3. Click **Initialize Local Brain**. This authorizes installing Ollama if it is
   missing and preparing the required model weights. Automatic Ollama
   installation needs Windows Package Manager (`winget`). If Windows does not
   have it, install Ollama from [ollama.com/download/windows](https://ollama.com/download/windows),
   reopen JARVIS, and retry initialization.
4. Keep internet access available while setup downloads any missing base
   weights. Wait until **JARVIS Brain** shows **LOCAL** and the message composer
   appears. Setup can also download hidden experts depending on your laptop's
   available RAM; the installer itself does not contain the language-model
   weights.
5. Type a short request and verify that you receive a useful reply. Then close
   and reopen JARVIS and verify that the Brain becomes ready again.

### Your HP and Dell installations

Use the **HP with 16 GB RAM as the primary laptop Brain** for the Honor companion.
Install the same Windows x64 preview on the Dell with 8 GB RAM if you also want
local JARVIS there. Leave automatic model selection enabled: the Dell skips the
7B engineering download by default, while the HP can prepare more experts when
memory permits. Actual response speed depends on free RAM and the processor;
close heavy applications before testing on the Dell.

These are independent local installations. Installing JARVIS on both PCs does
not automatically merge their memory. The current phone companion stores one
paired desktop at a time and does not switch automatically between HP and Dell.
Pair the Honor with the primary HP and keep that laptop awake while using it.
To switch, tap **Connect another laptop** on the phone, review the current
laptop, and confirm **Disconnect laptop**. This clears the phone's displayed
conversation and pending approvals while preserving its signing identity and
the laptop's stored data. Scan the other laptop's new QR and complete its
fingerprint and desktop-approval flow. Disconnecting locally does not revoke
the phone's existing entry on the former laptop.

### Reuse local model storage correctly

**Choose folder** selects an **Ollama model store**, the directory containing
`blobs` and `manifests`. On a standard Windows Ollama installation this is
usually `%USERPROFILE%\.ollama\models`; choose your custom `OLLAMA_MODELS`
directory if you already use one. Select that root directory, not `blobs`, an
individual model file, or Ollama's application installation folder.

JARVIS runs its local Ollama endpoint at `127.0.0.1:11435` and remembers the
chosen storage path. Existing models in a compatible store appear in **Local
models**. Initial setup still prepares any missing JARVIS aliases and their base
weights. Once the Brain is ready, open **Change local model folder** in **Local
models**, choose another Ollama store, and click **Use this folder** to provision
the required aliases there. This folder picker does not import individual model
files or directories of Hugging Face weights.

### Import a local GGUF expert

First finish local Brain setup.
Then click **Import GGUF expert** in **Local models** and select your existing
`.gguf` file in the native file picker. JARVIS imports that selected file into
the active Ollama store, creates a `jarvis-import-…` alias, and selects it as
the optional council expert. **JARVIS Core remains the coordinator.** To return
to automatic specialist routing, choose **Automatic expert selection**.

The selected file must exist as a regular file with a GGUF header. Ollama must
support its model architecture and report a text-capable model with a positive
parameter count. A file extension or header alone does not establish model
compatibility or useful inference; an import error is shown if validation fails.
Use a model that fits your laptop's available memory and verify a real answer
after importing it. This path accepts a selected GGUF file, not arbitrary Hugging
Face weight folders. Actual inference from the owner's imported GGUF weights
remains a real-device acceptance check.

Leave **Automatic expert selection** selected for normal use. JARVIS chooses
hidden specialists and can consult multiple experts concurrently when memory
headroom permits. The dropdown is an optional override for an installed model;
it is not required for each conversation and does not guarantee every model
will remain loaded simultaneously.

### What “JARVIS Core 1B” means in this build

`jarvis-core-1b` is an Ollama model created from **`llama3.2:1b`**, with JARVIS's
coordinator instructions and parameters. Its base weights belong to the Llama
3.2 1B family. The optional experts are `qwen3:1.7b`, `qwen3.5:4b`, and
`qwen2.5-coder:7b`, exposed internally through JARVIS aliases.

This repository does **not** contain an independently trained one-billion-
parameter JARVIS model. The authority, memory, routing, and tool code are
application logic; they do not themselves have a neural parameter count.
The [scratch-model training pipeline](../training/README.md) defines a
**1,543,714,304-parameter architecture** with random model-weight initialization,
preparation, training, evaluation, and export stages. Its CPU contract checks
validate architecture/configuration and training guards; they do not train a
1.5B model. Training and validating useful separate JARVIS weights still require
an appropriate GPU run and a sufficient licensed corpus. A parameter count
alone does not establish answer quality.

## Teach JARVIS your face

1. In Windows **Settings → Privacy & security → Camera**, allow camera access
   for desktop applications. Close other apps that are using the camera.
2. Sit alone in front of the laptop's default camera in even lighting, then
   click **Enroll owner face** in the **Owner identity** panel. Hold still while
   JARVIS collects multiple samples of the same face.
3. Click **Recognize me** and verify that **Owner face** becomes **RECOGNIZED**.
4. Test again after reopening JARVIS. Cover the camera or move out of frame,
   retry recognition, and verify that you are not reported as recognized.

Enrollment stores numeric templates in the operating system's secret store;
camera frames are discarded. Face recognition identifies the enrolled owner
but does not replace the phone fingerprint approval required for sensitive
actions. Owner recognition on your actual camera is only verified after you
perform these checks.

## Phone: install and pair

1. Install [Tailscale](https://tailscale.com/download) on both the laptop and
   phone, sign both into the same Tailscale network, and connect both devices.
   Being on the same Wi-Fi alone does not enable the automatic gateway.
2. **Restart JARVIS on the laptop after connecting Tailscale.** The **Device
   Link** panel should report that the secure companion gateway is online.
   JARVIS automatically binds its signed gateway to the laptop's Tailscale IPv4
   address on port `8765`. Leave the advanced endpoint override blank.
3. Open the downloaded `JARVIS-Companion.apk` on the phone.
   Allow installation from the app you used to open that file when Android asks.
4. Enroll and enable a fingerprint in Android's security settings. Open the
   JARVIS companion and check that it displays **FINGERPRINT READY**.
5. On the laptop, click **Create pairing QR**. On the phone, tap **Scan desktop
   QR**, allow camera access, scan the current code, and complete the phone's
   biometric prompt. Codes expire; create a fresh one if necessary.
6. Approve the pending phone in the laptop's **Device Link** panel. Wait for the
   phone to show the connected chat interface, then send a short message.

The preview APK is debug-signed. Candidate builds do not currently share a
configured permanent Android signing key, so Android may reject an APK upgrade
over an older candidate because its signing certificate differs. In that case,
uninstall the older JARVIS companion before installing the new APK, then pair it
again. Uninstalling removes the companion's local signing identity and desktop
pairing; retain the running laptop's memory rather than treating the phone as a
backup of it.

The phone sends signed requests to the running laptop Brain; Tailscale supplies
the private network connection. Sensitive approval uses the enrolled phone
fingerprint. A phone PIN or facial recognition does not substitute for it.
The companion uses a dedicated Android fingerprint-sensor path
rather than a generic biometric prompt; this behavior must still be checked on
your physical phone.
For an action that requests approval, read the exact pending action on the
phone, then choose **Approve with fingerprint** or **Reject**. Cancelling the
biometric prompt must leave the action unapproved.

If **Gateway offline** persists, confirm both Tailscale connections and restart
the laptop app. Check that your Tailscale access rules and Windows firewall
allow the phone to reach the laptop on port `8765`. A QR cannot repair a missing
network route. Gateway availability now requires a running local listener; a
QR still does not establish phone-to-laptop reachability. Keep JARVIS open and
the laptop awake while using the companion.

## iPhone and iPad status

The [iOS build workflow](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/workflows/build-ios.yml)
builds an unsigned simulator companion, installs it on an iPhone simulator, and
checks launch. Its `JARVIS-iOS-Simulator-not-for-phones` artifact is a simulator
application, not an installer for your iPhone or iPad. There is no physical iOS
installation asset in the Windows/Android preview release.

Physical iOS installation requires an Apple signing identity/team and device
provisioning or an appropriate signed distribution build. Actual pairing and
biometric approval must then be tested on that device. Simulator startup does
not certify Touch ID or fingerprint availability, and an iPhone with only Face
ID cannot meet a fingerprint-only approval requirement.

## Verify on your real devices

The [cloud runtime validation run](https://github.com/cutebody42-web/JARVIS-OS-V.2/actions/runs/36863469014)
contains evidence from real local inference, a Windows VM, and an Android
emulator. Those environments cannot prove that your microphone, camera,
fingerprint sensor, or device-specific drivers work. Record the installed
`build-info.txt` commit with these observations; do not mark a missing check as
passed.

| Check | What you should observe |
| --- | --- |
| Install/start/restart | Installed app opens, local Brain initializes, and reopening preserves setup. |
| Three text requests | Useful replies to a factual question, a planning request, and a coding request; no invented claim that an external action happened. |
| Optional imported model | Import a compatible GGUF through the file picker, see its alias selected in Local models, and verify a real response; switching back to automatic selection retains JARVIS Core. |
| Actual laptop voice | Select the default microphone in Windows, allow desktop microphone access, press JARVIS's microphone button, speak, and verify the recognized text and audible reply. Tap again to stop listening or interrupt speech. |
| Camera identity | Owner enrolls and is recognized; an empty or covered camera is rejected. |
| Phone connection | Pairing requires desktop approval; phone text reaches the same laptop Brain and still works after reopening the companion. |
| Sensitive approval | For a supported pending action, rejection or biometric cancellation does not execute it; approving with the enrolled fingerprint authorizes only the displayed exact action. |
| Lost connection | Disconnect phone Tailscale: requests fail visibly; reconnect and retry successfully. |

Native laptop voice currently uses Windows `System.Speech` with **English
(United States)** recognition by default. Install the corresponding Windows
speech components if voice recognition is unavailable. This is push-to-talk;
an always-listening wake word is not implemented. The Android companion in
this build provides text chat and approvals; it does not expose a voice-input
button. Arabic voice recognition and a separately trained JARVIS model are not
certified features of this candidate.

The physical hardware gate remains documented in
[the Windows hardware checklist](nexus/windows-hardware-gate.md). Installation
and source tests alone do not close that gate.
