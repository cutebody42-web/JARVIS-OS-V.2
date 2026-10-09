# JARVIS Android production signing

JARVIS keeps the ordinary Android companion workflow debug-signed for disposable CI/emulator validation. Distribution builds use the separate **Build signed JARVIS Android release** workflow and must use one stable owner-controlled keystore across releases.

## Security boundary

The keystore is never committed to this repository and never reaches the build runner. A production dispatch first builds an unsigned APK on an ephemeral runner, records its SHA-256 digest, and transfers only that APK plus its checksum to a fresh signing runner. The signing runner does not check out the source tree or run application and build scripts. The workflow definition is itself repository-controlled code, however, and its reviewed signing commands necessarily handle the key. Approval must therefore include the exact commit and this workflow's diff.

A production run fails closed unless all five secrets are configured in the protected `android-production-signing` GitHub Environment:

- `ANDROID_KEY_BASE64` — base64 of the private JKS/PKCS12 keystore file.
- `ANDROID_KEY_ALIAS` — alias of the upload/release key.
- `ANDROID_KEY_PASSWORD` — private-key password.
- `ANDROID_STORE_PASSWORD` — keystore password.
- `ANDROID_CERT_SHA256` — SHA-256 fingerprint of the expected signing certificate. Colons and letter case are ignored when comparing it.

Before the signing step receives secrets in its environment, the isolated runner requires exactly one regular APK, validates its bounded size and fixed checksum record, rejects an already signed input, and checks the manifest and native libraries with Android `apkanalyzer`. The accepted identity is `ai.jarvis.app`, version name `2.0.0`, version code `2000000`, non-debuggable, and ARM64-only. After signing, the workflow repeats the identity checks, verifies 16 KiB ZIP alignment and the APK signature, and rejects a certificate fingerprint that differs from `ANDROID_CERT_SHA256`.

Production dispatch is accepted only from `main`. Configure `android-production-signing` with a `main`-only deployment branch policy and a required owner reviewer so approval is needed before its secrets become available. Referencing an Environment in YAML does not prove those repository settings exist; verify them in GitHub before the first production run. The secrets are exposed only to one reviewed signing-and-verification step on the isolated runner, so dependency installation, Android project generation, and application/build scripts cannot read them. Pull-request runs receive only a throwaway CI key generated on their build runner.

## Configure the secrets

Create the keystore outside the repository and keep an offline backup. Android package upgrades depend on retaining the same signing identity.

Example local fingerprint inspection:

```bash
keytool -list -v -keystore /secure/path/jarvis-release.jks -alias YOUR_ALIAS
```

Create the `android-production-signing` Environment under **Settings → Environments**, restrict deployment branches to `main`, enable required-reviewer protection, and store all five values as environment secrets. Encode the keystore as one base64 value before saving it as `ANDROID_KEY_BASE64`. Do not also copy these values to repository-level secrets, and never paste the raw keystore or passwords into issues, commits, logs, or configuration files. The repository ignores common keystore extensions and the generated Android `keystore.properties`, but ignore rules are only a last line of defense; create and back up the key outside the checkout.

On Windows PowerShell, copy the keystore's base64 representation without creating another file in the repository:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes('C:\secure\jarvis-release.jks')) | Set-Clipboard
```

The root Tauri version is the Android `versionName`; `bundle.android.versionCode` is the monotonically increasing Android upgrade number. This release intentionally binds `2.0.0` to `2000000`. Increment both deliberately before a later production release, then update the workflow's expected values in the same reviewed change.

## Build a production artifact

1. Open **Actions → Build signed JARVIS Android release**.
2. Run the workflow from the exact release commit on `main`.
3. The build job creates an unsigned release APK and a checksum without receiving production secrets.
4. Inspect the exact commit and workflow diff, then approve the protected `android-production-signing` Environment.
5. A fresh runner validates the unsigned package identity without secrets, exposes the keystore only to the signing step, then repeats identity, alignment, signature, and certificate checks without the private key.
6. Download `JARVIS-Android-Signed-Release` only from that successful workflow run.
7. Keep `build-info.txt` with the APK; it records the source commit, signed and unsigned APK hashes, transfer-artifact hash, package/version identity, build type, signing mode, tool version, and signing-certificate SHA-256.

The workflow also runs on pull requests with a throwaway CI-only keystore. That proves the signing/build machinery without exposing or depending on production secrets. CI-signed artifacts are not uploaded for distribution.

## One-time transition from old debug APKs

Existing public preview APKs were debug-signed. Android will **not** install a production-signed APK as an in-place update over a package signed by a different debug certificate. The migration to the permanent production key therefore requires a one-time uninstall/reinstall (after preserving any data that is not already recoverable from JARVIS pairing/sync). After that transition, future APKs signed by the same permanent keystore can participate in normal same-package upgrade checks.

Never rotate or lose the signing key casually. A different certificate creates a different upgrade trust identity even when the package name is unchanged.

## What this does not certify

This pipeline proves that JARVIS can create and cryptographically verify a release-signed APK with the expected static identity. It cannot enforce GitHub Environment protection settings from inside the repository, prove that production secrets exist until the owner configures them, make the build fully reproducible, audit all application behavior, upload to Google Play, or substitute for physical installation/upgrade acceptance on the Honor phone.
