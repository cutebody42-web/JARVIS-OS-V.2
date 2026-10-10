"""Static security contract for paired-phone fingerprint approval."""

from pathlib import Path
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "web" / "src" / "lib" / "jarvis-runtime.ts"
SHELL = ROOT / "web" / "src" / "components" / "jarvis-product-shell.tsx"
MOBILE_NATIVE = ROOT / "web" / "src-tauri" / "src" / "mobile_identity.rs"
CAPABILITY = ROOT / "web" / "src-tauri" / "capabilities" / "mobile.json"
CARGO = ROOT / "web" / "src-tauri" / "Cargo.toml"
NATIVE_APP = ROOT / "web" / "src-tauri" / "src" / "lib.rs"
FINGERPRINT_PLUGIN = ROOT / "web" / "src-tauri" / "plugins" / "tauri-plugin-jarvis-fingerprint"
FINGERPRINT_RUST = FINGERPRINT_PLUGIN / "src" / "lib.rs"
FINGERPRINT_ANDROID = FINGERPRINT_PLUGIN / "android" / "src" / "main" / "java" / "FingerprintPlugin.kt"
FINGERPRINT_MANIFEST = FINGERPRINT_PLUGIN / "android" / "src" / "main" / "AndroidManifest.xml"
FINGERPRINT_IOS = FINGERPRINT_PLUGIN / "ios" / "Sources" / "FingerprintPlugin.swift"


def rust_function(source: str, name: str) -> str:
    """Read a module-level function, including its nested platform branches."""
    start = source.index(f"fn {name}(")
    return source[start:].split("\n}\n", 1)[0]


class ProductShellBiometricContractTests(unittest.TestCase):
    def test_ios_manifest_tools_support_the_declared_deployment_target(self):
        manifest = (FINGERPRINT_PLUGIN / "ios" / "Package.swift").read_text(encoding="utf-8")
        tools = re.search(r"swift-tools-version:(\d+)\.(\d+)", manifest)
        deployment = re.search(r"\.iOS\(\.v(\d+)\)", manifest)
        self.assertIsNotNone(tools)
        self.assertIsNotNone(deployment)
        # PackageDescription introduced the iOS 15 enum in Swift tools 5.5.
        # A current Xcode compiler still checks the manifest's declared API.
        if int(deployment[1]) >= 15:
            self.assertGreaterEqual((int(tools[1]), int(tools[2])), (5, 5))
        config = json.loads((ROOT / "web" / "src-tauri" / "tauri.ios.conf.json").read_text())
        self.assertEqual(int(deployment[1]), int(config["bundle"]["iOS"]["minimumSystemVersion"].split(".")[0]))

    def test_sensitive_approval_requires_native_fingerprint_without_pin_fallback(self):
        source = RUNTIME.read_text(encoding="utf-8")
        native = MOBILE_NATIVE.read_text(encoding="utf-8")

        # WebView code asks the native signer for a decision; it does not get to
        # assert biometric success or substitute a generic biometric prompt.
        self.assertIn('"mobile_sign_approval_decision"', source)
        self.assertNotIn('biometryType: "fingerprint"', source)
        self.assertNotIn("user_verified", source)

        decision = rust_function(native, "mobile_sign_approval_decision")
        self.assertIn("pub async fn mobile_sign_approval_decision", native)
        self.assertIn("spawn_blocking", decision)
        self.assertIn("signed_pending_response", decision)
        self.assertIn("sign_fingerprint_approval(", decision)
        signing = rust_function(native, "sign_fingerprint_approval")
        verify_pending = signing.index("verified_approval_list_payload(")
        authenticate = signing.index("authenticate_owner_fingerprint(")
        payload = signing.index("let payload = json!(")
        sign = signing.index("sign_companion_request(")
        self.assertLess(verify_pending, authenticate)
        self.assertLess(authenticate, payload)
        self.assertLess(payload, sign)
        # A rejected/cancelled native callback must propagate out before any
        # verified payload or signature is constructed.
        self.assertIn("authenticate_owner_fingerprint(&app, &prompt)?;", signing[authenticate:payload])
        self.assertIn("SHA-256", signing[:authenticate])
        self.assertIn("{summary}", signing[:authenticate])
        self.assertIn('"action_digest": action_digest', signing[payload:sign])
        self.assertNotIn("actionDigest: approval.action_digest", source)
        self.assertIn("signedPendingResponse: context.signedPendingResponse", source)
        self.assertIn("<small>SHA-256 {approval.action_digest}</small>", SHELL.read_text(encoding="utf-8"))

        authentication = rust_function(native, "authenticate_owner_fingerprint")
        self.assertIn('#[cfg(any(target_os = "android", target_os = "ios"))]', authentication)
        self.assertIn("app.fingerprint().authenticate", authentication)
        self.assertNotIn(".biometric()", authentication)

    def test_ios_success_requires_fresh_touch_id_without_passcode_or_face_id(self):
        ios = FINGERPRINT_IOS.read_text(encoding="utf-8")
        self.assertIn("context.biometryType == .touchID", ios)
        self.assertIn("context.touchIDAuthenticationAllowableReuseDuration = 0", ios)
        self.assertIn('context.localizedFallbackTitle = ""', ios)
        self.assertIn("context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics", ios)
        self.assertNotIn("evaluatePolicy(.deviceOwnerAuthentication,", ios)
        self.assertIn("success && context.biometryType == .touchID", ios)
        self.assertEqual(ios.count('session.invoke.resolve(["userVerified": true'), 1)
        self.assertIn("guard active?.id == session.id else { return }", ios)
        self.assertIn("UIApplication.didEnterBackgroundNotification", ios)
        self.assertIn(".now() + 30", ios)
        self.assertIn("session.context.invalidate()", ios)

    def test_ios_key_is_device_only_keychain_and_failures_do_not_regenerate_it(self):
        native = MOBILE_NATIVE.read_text(encoding="utf-8")
        app = NATIVE_APP.read_text(encoding="utf-8")
        signing = rust_function(native, "signing_key")
        self.assertIn("apple_native_keyring_store::protected::Store::new()", app)
        self.assertIn('("access-policy", "WhenUnlockedThisDeviceOnly")', signing)
        self.assertIn("Err(keyring_core::Error::NoEntry)", signing)
        self.assertIn("Err(error) => Err(", signing)
        self.assertIn("IDENTITY_LOCK.lock()", signing)
        self.assertIn('"ios_keychain_this_device_only"', native)

    def test_android_success_requires_the_fingerprint_sensor_callback(self):
        rust = FINGERPRINT_RUST.read_text(encoding="utf-8")
        android = FINGERPRINT_ANDROID.read_text(encoding="utf-8")
        self.assertIn('run_mobile_plugin("authenticate", AuthenticateRequest { reason })', rust)
        self.assertIn('!proof.user_verified || proof.biometry_type != "fingerprint"', rust)
        self.assertIn("android.hardware.fingerprint.FingerprintManager", android)
        self.assertIn("!sensor.hasEnrolledFingerprints()", android)
        self.assertIn("sensor()!!.authenticate(null, session.cancellation", android)
        self.assertIn("override fun onAuthenticationSucceeded", android)
        self.assertEqual(android.count("finish(session, null)"), 1)
        self.assertNotIn("BiometricPrompt(", android)
        self.assertNotIn("BIOMETRIC_WEAK", android)
        self.assertNotIn("DEVICE_CREDENTIAL", android)

    def test_fingerprint_lifecycle_cancels_without_reusing_old_success(self):
        android = FINGERPRINT_ANDROID.read_text(encoding="utf-8")
        finish = android.split("private fun finish(", 1)[1].split("override fun onPause", 1)[0]
        self.assertIn("if (active !== session) return", finish)
        self.assertLess(finish.index("active = null"), finish.index("session.cancellation.cancel()"))
        self.assertLess(finish.index("active = null"), finish.index("session.dialog?.dismiss()"))
        self.assertIn("handler.removeCallbacks", finish)
        self.assertIn("dialog.setOnCancelListener", android)
        self.assertIn("dialog.setOnDismissListener", android)
        self.assertIn("override fun onPause(activity: AppCompatActivity)", android)
        self.assertIn("override fun onDestroy(activity: AppCompatActivity)", android)
        self.assertIn("session.failedAttempts >= 3", android)
        self.assertIn("handler.postDelayed(timeout, 30000)", android)

    def test_pairing_authentication_uses_a_worker_and_native_fingerprint(self):
        source = RUNTIME.read_text(encoding="utf-8")
        native = MOBILE_NATIVE.read_text(encoding="utf-8")
        presence = source.split("async function requireMobileOwnerPresence(", 1)[1].split(
            "export function encodePairingDeepLink", 1
        )[0]
        self.assertIn("if (!status.fingerprint)", presence)
        self.assertIn('"mobile_verify_owner_presence"', presence)
        self.assertNotIn("biometric.authenticate", presence)
        for command in ("mobile_fingerprint_status", "mobile_verify_owner_presence"):
            with self.subTest(command=command):
                self.assertIn(f"pub async fn {command}", native)
                self.assertIn("spawn_blocking", rust_function(native, command))

    def test_pairing_preflights_fingerprint_before_qr(self):
        source = SHELL.read_text(encoding="utf-8")
        self.assertIn("mobileBiometricStatus()", source)
        self.assertIn("biometric?.fingerprint !== true", source)
        self.assertIn("SECURITY / FINGERPRINT READY", source)
        self.assertIn("does not fall back to a phone PIN", source)

    def test_native_mobile_build_includes_the_sensor_plugin_and_permissions(self):
        capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))
        cargo = CARGO.read_text(encoding="utf-8")
        native = NATIVE_APP.read_text(encoding="utf-8")
        manifest = FINGERPRINT_MANIFEST.read_text(encoding="utf-8")
        # Generic UI biometric permission grants no signing authority. Sensitive
        # authentication goes through each platform's native-only sensor path.
        self.assertIn("biometric:default", capability["permissions"])
        self.assertIn('tauri-plugin-biometric = "2.4.0"', cargo)
        self.assertIn('tauri-plugin-jarvis-fingerprint = { path = "plugins/tauri-plugin-jarvis-fingerprint" }', cargo)
        self.assertIn("builder.plugin(tauri_plugin_jarvis_fingerprint::init())", native)
        self.assertIn("android.permission.USE_FINGERPRINT", manifest)
        self.assertIn("android.permission.USE_BIOMETRIC", manifest)


if __name__ == "__main__":
    unittest.main()
