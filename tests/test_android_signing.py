import tempfile
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


if __name__ == "__main__":
    unittest.main()
