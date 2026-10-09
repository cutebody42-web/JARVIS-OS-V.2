# JARVIS iOS production signing

The normal `build-ios.yml` workflow intentionally builds and launches an **unsigned simulator** app. It proves the Tauri/Xcode product shell and deep-link path without inventing an Apple developer identity.

Physical iPhone/iPad distribution uses the separate **Build signed JARVIS iOS release** workflow. That workflow follows Tauri's supported manual iOS signing environment variables and exports with `app-store-connect`.

## Required GitHub Actions secrets

Configure these repository secrets from an enrolled Apple Developer account:

- `IOS_CERTIFICATE` — base64 of the exported Apple Distribution `.p12` certificate and private key.
- `IOS_CERTIFICATE_PASSWORD` — password used when exporting the `.p12`.
- `IOS_MOBILE_PROVISION` — base64 of an App Store Connect provisioning profile for bundle ID `ai.jarvis.app`.
- `APPLE_DEVELOPMENT_TEAM` — the 10-character Apple Developer Team ID.
- `IOS_CERT_SHA256` — SHA-256 fingerprint of the exact distribution certificate expected to sign JARVIS.

Never commit the certificate, private key, profile, passwords, or API credentials to the repository.

## Build and verification

1. Open **Actions → Build signed JARVIS iOS release**.
2. Run it from the exact release commit on `main`.
3. The workflow validates the required secret shape, initializes the generated Tauri Xcode project, and builds with `--export-method app-store-connect` for an arm64 iOS device target.
4. It opens the produced IPA and verifies:
   - the code signature with `codesign --verify --deep --strict`;
   - bundle ID `ai.jarvis.app`;
   - the leaf signing certificate SHA-256 against `IOS_CERT_SHA256`;
   - the embedded provisioning profile's Apple team ID;
   - the profile application identifier `<TEAM_ID>.ai.jarvis.app`.
5. Only after all checks pass is `JARVIS-iOS-Signed-Release` uploaded as an Actions artifact.

`build-info.txt` travels with the IPA and records source commit, bundle/team identity, export method, signing certificate fingerprint, IPA SHA-256, and the still-unverified physical-device gates.

## Physical acceptance is separate

A successful signed IPA proves the software signing/export path. It does **not** prove installation or biometric behavior on the owner's phone. Physical acceptance still requires:

- install/provisioning on the target iPhone;
- launch and deep-link test on that phone;
- pairing to the intended JARVIS Brain;
- biometric approve/reject/cancel validation;
- Touch ID specifically if fingerprint approval is a hard requirement.

A Face ID-only device cannot be claimed to have passed a Touch ID requirement. The workflow therefore records `touch_id_verified=false` and `physical_device_verified=false`; only a real-device acceptance run can change those claims.

## Alternative automatic signing

Tauri also supports App Store Connect API-key based automatic signing. This repository uses the manual certificate/profile path for the production artifact because it allows the workflow to pin and verify the exact signing certificate and provisioning identity explicitly. A later migration to automatic signing should preserve equivalent identity verification rather than weakening this gate.
