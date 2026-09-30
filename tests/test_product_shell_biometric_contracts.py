"""Static security contract for paired-phone fingerprint approval."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "web" / "src" / "lib" / "jarvis-runtime.ts"
SHELL = ROOT / "web" / "src" / "components" / "jarvis-product-shell.tsx"
MOBILE_NATIVE = ROOT / "web" / "src-tauri" / "src" / "mobile_identity.rs"
CAPABILITY = ROOT / "web" / "src-tauri" / "capabilities" / "mobile.json"
CARGO = ROOT / "web" / "src-tauri" / "Cargo.toml"


class ProductShellBiometricContractTests(unittest.TestCase):
    def test_sensitive_approval_requires_native_fingerprint_without_pin_fallback(self):
        source = RUNTIME.read_text(encoding="utf-8")
        native = MOBILE_NATIVE.read_text(encoding="utf-8")

        # Pairing still preflights fingerprint in the UI, but sensitive action
        # approval must be enforced again inside the native signing command.
        self.assertIn("status.biometryType !== 1", source)
        self.assertIn("allowDeviceCredential: false", source)
        self.assertIn("confirmationRequired: true", source)
        self.assertIn('"mobile_sign_approval_decision"', source)
        self.assertNotIn('biometryType: "fingerprint"', source)

        decision = native.index("pub fn mobile_sign_approval_decision")
        status = native.index(".biometric().status()", decision)
        authenticate = native.index(".biometric().authenticate(", decision)
        sign = native.index("sign_companion_request(", decision)
        self.assertLess(status, authenticate)
        self.assertLess(authenticate, sign)
        self.assertIn("allow_device_credential: false", native[decision:sign])
        self.assertIn("BiometryType::TouchID", native[decision:sign])

    def test_pairing_preflights_fingerprint_before_qr(self):
        source = SHELL.read_text(encoding="utf-8")
        self.assertIn("mobileBiometricStatus()", source)
        self.assertIn("biometric?.fingerprint !== true", source)
        self.assertIn("SECURITY / FINGERPRINT READY", source)
        self.assertIn("does not fall back to a phone PIN", source)

    def test_native_mobile_build_includes_biometric_plugin_and_permission(self):
        capability = CAPABILITY.read_text(encoding="utf-8")
        cargo = CARGO.read_text(encoding="utf-8")
        self.assertIn('"biometric:default"', capability)
        self.assertIn('tauri-plugin-biometric = "2.4.0"', cargo)


if __name__ == "__main__":
    unittest.main()
