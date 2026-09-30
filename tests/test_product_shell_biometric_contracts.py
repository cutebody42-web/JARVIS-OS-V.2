"""Static security contract for paired-phone fingerprint approval."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "web" / "src" / "lib" / "jarvis-runtime.ts"
SHELL = ROOT / "web" / "src" / "components" / "jarvis-product-shell.tsx"
CAPABILITY = ROOT / "web" / "src-tauri" / "capabilities" / "mobile.json"
CARGO = ROOT / "web" / "src-tauri" / "Cargo.toml"


class ProductShellBiometricContractTests(unittest.TestCase):
    def test_sensitive_approval_requires_fingerprint_without_pin_fallback(self):
        source = RUNTIME.read_text(encoding="utf-8")
        self.assertIn("status.biometryType !== 1", source)
        self.assertIn("allowDeviceCredential: false", source)
        self.assertIn("confirmationRequired: true", source)
        self.assertIn('biometryType: "fingerprint"', source)

        decision = source.index("export async function mobileDecideApproval")
        presence = source.index("await requireMobileOwnerPresence(", decision)
        signature = source.index('"mobile_sign_approval_decision"', decision)
        self.assertLess(presence, signature)

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
