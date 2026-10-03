# JARVIS iOS companion

The shared Tauri companion now has an iOS build path, QR/deep-link pairing,
signed chat transport, and native Touch ID approval. It connects to the same
running laptop Brain through Tailscale; it does not run the laptop's language
models on the phone. iOS 15 or later is required.

## Fingerprint capability

JARVIS owner authentication uses `LocalAuthentication` with a fresh `LAContext`,
the biometrics-only policy, and an explicit **Touch ID** hardware check. Face ID,
a passcode, and a previously cached successful authentication cannot provide
fingerprint approval. Cancellation, interruption, lockout, and timeout fail the
pending authentication. Native Rust signs an approval only after this sensor
callback succeeds.

**An iPhone with Face ID and no Touch ID cannot currently pair as an approving
JARVIS companion.** This follows the project's fingerprint requirement. A Touch
ID-capable iPhone or iPad with an enrolled fingerprint is required for that role.
The fingerprint button remains unavailable when the operating system reports
unsupported hardware or missing enrollment; this is not a simulated success.

The device signing seed is stored in the iOS Keychain with
`WhenUnlockedThisDeviceOnly` accessibility and without iCloud synchronization.
Desktop trust remains in the native application sandbox. A mobile key read
failure does not silently replace an existing identity.

## Build and install

The **Build JARVIS iOS companion** GitHub Actions workflow builds an unsigned
simulator application on macOS, installs it on an iPhone simulator, checks that
the launched process remains alive, opens the JARVIS URL scheme, and uploads a
screenshot plus source/architecture/hash evidence. Its artifact is named
`JARVIS-iOS-Simulator-not-for-phones`. It is **not an installable iPhone IPA** and
does not establish real Touch ID or laptop connectivity. The workflow must pass
for the relevant source commit before a simulator build is considered verified.

Installing on a physical Apple device requires a Mac with Xcode and an Apple
development team/provisioning profile. For development, check out the intended
revision, install stable Rust and Node 24, and run:

```sh
cd web
npm ci
export APPLE_DEVELOPMENT_TEAM="your-real-Apple-team-ID"
npm run tauri -- ios init
npm run tauri -- ios dev
```

Select the connected Touch ID-capable device when prompted. Xcode must authorize
that device for the specified team; enable Developer Mode on the device when
iOS requests it. For distribution, build with real signing and publish through
the owner's Apple Developer/TestFlight/App Store process. This repository does
not invent a team identifier or claim Apple signing credentials are configured.

Once installed, connect the device and laptop to the same Tailscale network,
restart desktop JARVIS so its gateway starts, and follow
[the phone pairing instructions](JARVIS-INSTALL.md#phone-install-and-pair).
Allow camera access for QR scanning. Verify chat, restart persistence, exact
fingerprint approval, cancellation, and lost-network recovery on the physical
device before accepting it as ready.
