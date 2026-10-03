"""Static security contract for paired-phone fingerprint approval."""

from pathlib import Path
import json
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


def rust_function(source: str, name: str) -> str:
    """Read a module-level function, including its nested platform branches."""
    start = source.index(f"fn {name}(")
    return source[start:].split("\n}\n", 1)[0]


class ProductShellBiometricContractTests(unittest.TestCase):
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
        self.assertIn("sign_fingerprint_approval(app, approval_id, approved)", decision)
        signing = rust_function(native, "sign_fingerprint_approval")
        authenticate = signing.index("authenticate_owner_fingerprint(")
        payload = signing.index("let payload = json!(")
        sign = signing.index("sign_companion_request(")
        self.assertLess(authenticate, payload)
        self.assertLess(payload, sign)
        # A rejected/cancelled native callback must propagate out before any
        # verified payload or signature is constructed.
        self.assertIn("})?;", signing[authenticate:payload])

        authentication = rust_function(native, "authenticate_owner_fingerprint")
        android = authentication.split('#[cfg(target_os = "android")]', 1)[1].split(
            '#[cfg(target_os = "ios")]', 1
        )[0]
        self.assertIn("app.fingerprint().authenticate", android)
        self.assertNotIn(".biometric()", android)
        ios = authentication.split('#[cfg(target_os = "ios")]', 1)[1]
        self.assertIn("BiometryType::TouchID", ios)
        self.assertIn("allow_device_credential: false", ios)

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
        # iOS Touch ID still uses the generic platform plugin. Android-sensitive
        # authorization is routed through the separate native-only sensor path.
        self.assertIn("biometric:default", capability["permissions"])
        self.assertIn('tauri-plugin-biometric = "2.4.0"', cargo)
        self.assertIn('tauri-plugin-jarvis-fingerprint = { path = "plugins/tauri-plugin-jarvis-fingerprint" }', cargo)
        self.assertIn("builder.plugin(tauri_plugin_jarvis_fingerprint::init())", native)
        self.assertIn("android.permission.USE_FINGERPRINT", manifest)
        self.assertIn("android.permission.USE_BIOMETRIC", manifest)


if __name__ == "__main__":
    unittest.main()
