"""Guard the native runtime resource boundary across desktop and mobile builds."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WindowsPackagingTests(unittest.TestCase):
    def test_windows_dll_and_installer_hook_are_platform_scoped(self):
        common = json.loads((ROOT / "web/src-tauri/tauri.conf.json").read_text("utf-8"))
        windows = json.loads((ROOT / "web/src-tauri/tauri.windows.conf.json").read_text("utf-8"))
        self.assertNotIn("libsodium.dll", json.dumps(common))
        self.assertNotIn("installerHooks", json.dumps(common))
        self.assertEqual(common["bundle"]["android"]["versionCode"], 2000000)
        self.assertEqual(windows["bundle"]["targets"], ["nsis"])
        self.assertEqual(windows["bundle"]["externalBin"], ["binaries/jarvis-brain"])
        self.assertIs(windows["bundle"]["windows"]["bundleVCRuntime"], True)
        self.assertIn("resources/libsodium.dll", windows["bundle"]["resources"])
        nsis = windows["bundle"]["windows"]["nsis"]
        self.assertEqual(nsis["installMode"], "currentUser")
        self.assertTrue((ROOT / "web/src-tauri" / nsis["installerHooks"]).is_file())
        # An overlay must not replace shared identity, mobile metadata or build hooks.
        self.assertEqual(set(windows) - {"$schema"}, {"bundle"})
        self.assertEqual(set(windows["bundle"]), {"targets", "externalBin", "resources", "windows"})

    def staging_script(self, workflow_name):
        workflow = (ROOT / ".github/workflows" / workflow_name).read_text("utf-8")
        if workflow_name == "real-cloud-validation.yml":
            workflow = workflow.split("  windows-real-runtime:\n", 1)[1].split(
                "  android-real-runtime:\n", 1
            )[0]
        self.assertLess(workflow.index("$sodiumDll ="), workflow.index("tauri build"))
        return textwrap.dedent(
            "          $sodiumDll =" + workflow.split("$sodiumDll =", 1)[1].split("\n\n", 1)[0]
        )

    def test_both_windows_builds_stage_the_same_native_runtime(self):
        installer = self.staging_script("build-windows.yml")
        runtime = self.staging_script("real-cloud-validation.yml")
        self.assertEqual(installer, runtime)
        self.assertIn('"installed\\x64-windows\\bin\\libsodium.dll"', installer)
        self.assertIn('"web/src-tauri/resources/libsodium.dll"', installer)

    @unittest.skipUnless(os.name == "nt", "Execute the Windows staging script on Windows")
    def test_staging_copies_runtime_bytes_and_rejects_missing_dll(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell unavailable")
        script = self.staging_script("real-cloud-validation.yml")
        for present in (True, False):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "vcpkg/installed/x64-windows/bin/libsodium.dll"
                source.parent.mkdir(parents=True)
                if present:
                    source.write_bytes(b"fixture native runtime bytes")
                result = subprocess.run(
                    [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                     "$ErrorActionPreference = 'Stop'; $vcpkgRoot = Join-Path $PWD 'vcpkg';\n" + script],
                    cwd=root, capture_output=True, text=True, timeout=20,
                )
                target = root / "web/src-tauri/resources/libsodium.dll"
                if present:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(target.read_bytes(), source.read_bytes())
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
