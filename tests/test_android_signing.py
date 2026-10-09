import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.configure_android_signing import (
    AndroidSigningConfigError,
    MARKER,
    configure_file,
    configure_gradle,
)


_FIXTURE = '''import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
}

android {
    namespace = "com.jarvis.companion"

    defaultConfig {
        applicationId = "com.jarvis.companion"
        minSdk = 24
        targetSdk = 36
    }

    buildTypes {
        getByName("debug") {
            isDebuggable = true
        }
        getByName("release") {
            isMinifyEnabled = true
        }
    }
}
'''


class AndroidSigningPatchTests(unittest.TestCase):
    def test_release_signing_is_scoped_and_idempotent(self):
        configured = configure_gradle(_FIXTURE)
        self.assertIn(MARKER, configured)
        self.assertIn("import java.util.Properties", configured)
        self.assertIn("import java.io.FileInputStream", configured)
        self.assertEqual(configured.count('signingConfig = signingConfigs.getByName("release")'), 1)

        debug_start = configured.index('getByName("debug")')
        release_start = configured.index('getByName("release")', debug_start)
        assignment = configured.index('signingConfig = signingConfigs.getByName("release")')
        self.assertGreater(assignment, release_start)
        self.assertNotIn("signingConfig", configured[debug_start:release_start])
        self.assertEqual(configure_gradle(configured), configured)

    def test_private_properties_are_required_at_gradle_time(self):
        configured = configure_gradle(_FIXTURE)
        self.assertIn('rootProject.file("keystore.properties")', configured)
        self.assertIn('throw GradleException("JARVIS release signing requires keystore.properties")', configured)
        self.assertIn('getProperty("keyPassword")', configured)
        self.assertIn('getProperty("storePassword")', configured)

    def test_existing_release_signing_is_not_silently_overwritten(self):
        fixture = _FIXTURE.replace(
            'getByName("release") {\n            isMinifyEnabled = true',
            'getByName("release") {\n            signingConfig = signingConfigs.getByName("debug")\n            isMinifyEnabled = true',
        )
        with self.assertRaises(AndroidSigningConfigError):
            configure_gradle(fixture)

    def test_unexpected_generated_layout_fails_closed(self):
        with self.assertRaises(AndroidSigningConfigError):
            configure_gradle('android { defaultConfig { minSdk = 24 } }')
        with self.assertRaises(AndroidSigningConfigError):
            configure_gradle('')

    def test_file_patch_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.gradle.kts"
            path.write_text(_FIXTURE, encoding="utf-8")
            configure_file(path)
            first = path.read_text(encoding="utf-8")
            configure_file(path)
            self.assertEqual(path.read_text(encoding="utf-8"), first)


class AndroidSigningWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.workflow = (
            cls.root / ".github"
            / "workflows"
            / "build-android-signed.yml"
        ).read_text(encoding="utf-8")

    def test_production_dispatch_is_restricted_to_main(self):
        self.assertIn(
            "if: github.event_name != 'workflow_dispatch' || github.ref == 'refs/heads/main'",
            self.workflow,
        )

    def test_production_secrets_are_not_job_wide(self):
        build_job = self.workflow.split("  build-release:", 1)[1].split(
            "  production-sign:", 1
        )[0]
        for name in (
            "ANDROID_KEY_BASE64",
            "ANDROID_KEY_ALIAS",
            "ANDROID_KEY_PASSWORD",
            "ANDROID_STORE_PASSWORD",
            "ANDROID_CERT_SHA256",
        ):
            self.assertNotIn(f"secrets.{name}", build_job)

    def test_production_signing_uses_a_fresh_approval_gated_runner(self):
        signer = self.workflow.split("  production-sign:", 1)[1]
        self.assertIn("needs: build-release", signer)
        self.assertIn("environment: android-production-signing", signer)
        self.assertNotIn("actions/checkout", signer)
        self.assertLess(
            signer.index("Download unsigned APK without executing application code"),
            signer.index("ANDROID_KEY_BASE64: ${{ secrets.ANDROID_KEY_BASE64 }}"),
        )
        self.assertIn("--ks-pass env:ANDROID_STORE_PASSWORD", signer)
        self.assertIn("--key-pass env:ANDROID_KEY_PASSWORD", signer)
        self.assertIn("ANDROID_CERT_SHA256: ${{ secrets.ANDROID_CERT_SHA256 }}", signer)

    def test_release_identity_matches_tauri_configuration(self):
        config = json.loads(
            (self.root / "web/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
        )
        for field, value in (
            ("PACKAGE_ID", config["identifier"]),
            ("VERSION_NAME", config["version"]),
            ("VERSION_CODE", str(config["bundle"]["android"]["versionCode"])),
        ):
            self.assertEqual(self.workflow.count(f'"${field}" != "{value}"'), 2)
        signer = self.workflow.split("  production-sign:", 1)[1]
        self.assertLess(
            signer.index("Validate unsigned release identity before exposing secrets"),
            signer.index("ANDROID_KEY_BASE64: ${{ secrets.ANDROID_KEY_BASE64 }}"),
        )
        self.assertIn("persist-credentials: false", self.workflow)

    def test_workflow_dependencies_are_commit_pinned_and_cli_is_locked(self):
        uses = re.findall(r"^\s*(?:-\s+)?uses:\s*(actions/[^@\s]+)@([^\s#]+)", self.workflow, re.MULTILINE)
        self.assertTrue(uses)
        self.assertTrue(all(re.fullmatch(r"[a-f0-9]{40}", revision) for _, revision in uses))
        self.assertNotIn("npx --yes", self.workflow)
        self.assertIn("npx --no-install tauri android init", self.workflow)
        self.assertIn("npx --no-install tauri android build", self.workflow)

    def test_all_build_workflows_use_the_lockfile_cli(self):
        package = json.loads((self.root / "web" / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((self.root / "web" / "package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(package["scripts"]["tauri"], "tauri")
        self.assertEqual(package["devDependencies"]["@tauri-apps/cli"], "2.12.0")
        self.assertEqual(
            lock["packages"]["node_modules/@tauri-apps/cli"]["version"],
            "2.12.0",
        )
        for name in (
            "build-android.yml",
            "build-android-signed.yml",
            "build-windows.yml",
            "real-cloud-validation.yml",
        ):
            workflow = (self.root / ".github" / "workflows" / name).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("npx --yes @tauri-apps/cli", workflow, name)
            self.assertNotIn("npx @tauri-apps/cli", workflow, name)


class AndroidUnsignedValidationTests(unittest.TestCase):
    """Execute the real transfer guard, with Android analysis tools stubbed.

    The APK fixture is deliberately not a real build; these cases check transfer
    handling and manifest rejection, not Android compilation or cryptography.
    """

    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if not cls.bash and os.name == "nt":
            candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if candidate.is_file():
                cls.bash = str(candidate)
        if not cls.bash:
            raise unittest.SkipTest("Bash is required to execute the workflow guard")
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github/workflows/build-android-signed.yml").read_text(encoding="utf-8")
        step = workflow.split("      - name: Validate unsigned release identity before exposing secrets\n", 1)[1]
        cls.script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])

    def run_guard(self, *, checksum=None, extra_file=False, **values):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer = root / "unsigned"
            transfer.mkdir()
            apk = b"fixture APK; Android analysis is stubbed"
            (transfer / "JARVIS-Companion-release-unsigned.apk").write_bytes(apk)
            record = hashlib.sha256(apk).hexdigest() + "  JARVIS-Companion-release-unsigned.apk\n"
            (transfer / "SHA256SUMS").write_bytes((record if checksum is None else checksum).encode())
            if extra_file:
                (transfer / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TEST_PACKAGE": "ai.jarvis.app",
                "TEST_DEBUGGABLE": "false",
                "TEST_VERSION_NAME": "2.0.0",
                "TEST_VERSION_CODE": "2000000",
                "TEST_ABIS": "/lib/arm64-v8a/libapp.so",
                "TEST_SIGNED": "false",
            })
            env.update(values)
            prelude = '''
export GITHUB_WORKSPACE="$PWD" GITHUB_OUTPUT="$PWD/guard-output"
export APKSIGNER=stub_apksigner APKANALYZER=stub_apkanalyzer
stub_apksigner() { [ "$TEST_SIGNED" = "true" ]; }
stub_apkanalyzer() {
  case "$1 $2" in
    "manifest application-id") printf '%s\\n' "$TEST_PACKAGE" ;;
    "manifest debuggable") printf '%s\\n' "$TEST_DEBUGGABLE" ;;
    "manifest version-name") printf '%s\\n' "$TEST_VERSION_NAME" ;;
    "manifest version-code") printf '%s\\n' "$TEST_VERSION_CODE" ;;
    "files list") printf '%s\\n' "$TEST_ABIS" ;;
    *) return 2 ;;
  esac
}
'''
            result = subprocess.run(
                [self.bash, "--noprofile", "--norc"],
                input=prelude + self.script,
                text=True,
                capture_output=True,
                cwd=root,
                env=env,
                timeout=20,
            )
            output = root / "guard-output"
            return result, output.read_text() if output.exists() else ""

    def test_valid_transfer_records_verified_digest(self):
        result, output = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(output, r"^sha256=[a-f0-9]{64}\n$")

    def test_unsafe_or_corrupt_transfers_fail_before_output(self):
        for arguments in (
            {"extra_file": True},
            {"checksum": "0" * 64 + "  JARVIS-Companion-release-unsigned.apk\n"},
            {"checksum": "0" * 64 + "  ../outside.apk\n"},
            {"checksum": "x" * 4096},
            {"TEST_SIGNED": "true"},
        ):
            with self.subTest(arguments=arguments):
                result, output = self.run_guard(**arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")

    def test_unexpected_release_identity_fails_before_output(self):
        for arguments in (
            {"TEST_PACKAGE": "attacker.app"},
            {"TEST_DEBUGGABLE": "true"},
            {"TEST_VERSION_NAME": "1.0.0"},
            {"TEST_VERSION_CODE": "1"},
            {"TEST_ABIS": "/lib/x86_64/libapp.so"},
            {"TEST_ABIS": "/lib/arm64-v8a/libapp.so\n/lib/x86_64/libapp.so"},
            {"TEST_ABIS": ""},
        ):
            with self.subTest(arguments=arguments):
                result, output = self.run_guard(**arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
