# JARVIS Android production signing

JARVIS keeps the ordinary Android companion workflow debug-signed for disposable CI/emulator validation. Distribution builds use the separate **Build signed JARVIS Android release** workflow and must use one stable owner-controlled keystore across releases.

## Security boundary

The keystore is never committed to this repository. The release workflow creates `web/src-tauri/gen/android/keystore.properties` only inside the ephemeral Actions runner and deletes it with the runner. The generated Gradle project is patched after `tauri android init` by `scripts/configure_android_signing.py`.

A production run fails closed unless all five repository Actions secrets are configured:

- `ANDROID_KEY_BASE64` — base64 of the private JKS/PKCS12 keystore file.
- `ANDROID_KEY_ALIAS` — alias of the upload/release key.
- `ANDROID_KEY_PASSWORD` — private-key password.
- `ANDROID_STORE_PASSWORD` — keystore password.
- `ANDROID_CERT_SHA256` — SHA-256 fingerprint of the expected signing certificate. Colons and letter case are ignored when comparing it.

The workflow verifies the resulting APK with Android `apksigner` and rejects a production artifact if the certificate fingerprint differs from `ANDROID_CERT_SHA256`.

## Configure the secrets

Create the keystore outside the repository and keep an offline backup. Android package upgrades depend on retaining the same signing identity.

Example local fingerprint inspection:

```bash
keytool -list -v -keystore /secure/path/jarvis-release.jks -alias YOUR_ALIAS
```

Encode the keystore as one base64 value before saving it as `ANDROID_KEY_BASE64`. Do not paste the raw keystore or passwords into issues, commits, logs, or configuration files.

## Build a production artifact

1. Open **Actions → Build signed JARVIS Android release**.
2. Run the workflow from the exact release commit on `main`.
3. The job validates the secrets, generates private `keystore.properties`, patches the generated Tauri Gradle application, builds a release APK, verifies its signature, and checks its certificate fingerprint.
4. Download `JARVIS-Android-Signed-Release` only from that successful workflow run.
5. Keep `build-info.txt` with the APK; it records the source commit, APK SHA-256, build type, signing mode, and signing-certificate SHA-256.

The workflow also runs on pull requests with a throwaway CI-only keystore. That proves the signing/build machinery without exposing or depending on production secrets. CI-signed artifacts are not uploaded for distribution.

## One-time transition from old debug APKs

Existing public preview APKs were debug-signed. Android will **not** install a production-signed APK as an in-place update over a package signed by a different debug certificate. The migration to the permanent production key therefore requires a one-time uninstall/reinstall (after preserving any data that is not already recoverable from JARVIS pairing/sync). After that transition, future APKs signed by the same permanent keystore can participate in normal same-package upgrade checks.

Never rotate or lose the signing key casually. A different certificate creates a different upgrade trust identity even when the package name is unchanged.

## What this does not certify

This pipeline proves that JARVIS can create and cryptographically verify a release-signed APK. It does not claim that production secrets exist until the owner configures them, does not upload to Google Play, and does not substitute for physical installation/upgrade acceptance on the Honor phone.
