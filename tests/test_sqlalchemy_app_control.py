from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import sqlalchemy_app_control as repair


def completed(
    returncode: int, *, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


APP_CONTROL_FAILURE = """Traceback (most recent call last):
  File "C:\\venv\\Lib\\site-packages\\sqlalchemy\\sql\\visitors.py", line 36, in <module>
    from ._util_cy import anon_map as anon_map
ImportError: DLL load failed while importing _util_cy: An Application Control policy has blocked this file.
"""


class SQLAlchemyAppControlDetectionTests(unittest.TestCase):
    def test_recognizes_sqlalchemy_extension_block_on_windows(self):
        self.assertTrue(
            repair.is_sqlalchemy_app_control_failure(APP_CONTROL_FAILURE, platform="win32")
        )

    def test_rejects_same_message_off_windows(self):
        self.assertFalse(
            repair.is_sqlalchemy_app_control_failure(APP_CONTROL_FAILURE, platform="linux")
        )

    def test_rejects_unrelated_dll_block(self):
        output = APP_CONTROL_FAILURE.replace("_util_cy", "unrelated_native_module")
        self.assertFalse(repair.is_sqlalchemy_app_control_failure(output, platform="win32"))

    def test_rejects_matching_module_without_sqlalchemy_trace(self):
        output = (
            "ImportError: DLL load failed while importing _util_cy: "
            "An Application Control policy has blocked this file."
        )
        self.assertFalse(repair.is_sqlalchemy_app_control_failure(output, platform="win32"))

    def test_rejects_ordinary_sqlalchemy_import_error(self):
        output = APP_CONTROL_FAILURE.replace(
            "An Application Control policy has blocked this file.",
            "The specified module could not be found.",
        )
        self.assertFalse(repair.is_sqlalchemy_app_control_failure(output, platform="win32"))


class SQLAlchemyAppControlRepairTests(unittest.TestCase):
    def test_successful_import_does_not_install(self):
        with (
            patch.object(
                repair.subprocess,
                "run",
                return_value=completed(0, stdout="2.0.44\n"),
            ) as run,
            patch.object(repair.subprocess, "check_call") as install,
        ):
            changed = repair.ensure_sqlalchemy_import(
                "venv-python", root=Path("repo"), platform="win32"
            )

        self.assertFalse(changed)
        install.assert_not_called()
        self.assertEqual(run.call_count, 1)

    def test_recognized_failure_installs_source_build_and_rechecks(self):
        with (
            patch.object(
                repair.subprocess,
                "run",
                side_effect=[
                    completed(1, stderr=APP_CONTROL_FAILURE),
                    completed(0, stdout="2.0.44\n"),
                    completed(0, stdout="2.0.44\n"),
                ],
            ) as run,
            patch.object(repair.subprocess, "check_call") as install,
            patch.dict(
                repair.os.environ,
                {"REQUIRE_SQLALCHEMY_CEXT": "1", "KEEP_ME": "yes"},
                clear=True,
            ),
        ):
            changed = repair.ensure_sqlalchemy_import(
                "venv-python", root=Path("repo"), platform="win32"
            )

        self.assertTrue(changed)
        self.assertEqual(run.call_count, 3)
        command = install.call_args.args[0]
        self.assertEqual(command[:4], ["venv-python", "-m", "pip", "install"])
        self.assertIn("--force-reinstall", command)
        self.assertIn("--no-cache-dir", command)
        self.assertIn("--no-deps", command)
        no_binary = command.index("--no-binary")
        self.assertEqual(command[no_binary + 1], "SQLAlchemy")
        self.assertEqual(command[-1], "SQLAlchemy==2.0.44")
        install_env = install.call_args.kwargs["env"]
        self.assertEqual(install_env["DISABLE_SQLALCHEMY_CEXT"], "1")
        self.assertEqual(install_env["KEEP_ME"], "yes")
        self.assertNotIn("REQUIRE_SQLALCHEMY_CEXT", install_env)

    def test_unrelated_failure_is_reported_without_install(self):
        failure = APP_CONTROL_FAILURE.replace("_util_cy", "other_extension")
        with (
            patch.object(
                repair.subprocess,
                "run",
                return_value=completed(1, stderr=failure),
            ),
            patch.object(repair.subprocess, "check_call") as install,
        ):
            with self.assertRaisesRegex(
                repair.SQLAlchemyImportError, "fallback was not attempted"
            ):
                repair.ensure_sqlalchemy_import(
                    "venv-python", root=Path("repo"), platform="win32"
                )
        install.assert_not_called()

    def test_failed_verification_after_reinstall_is_reported(self):
        with (
            patch.object(
                repair.subprocess,
                "run",
                side_effect=[
                    completed(1, stderr=APP_CONTROL_FAILURE),
                    completed(0, stdout="2.0.44\n"),
                    completed(1, stderr="ImportError: still broken"),
                ],
            ),
            patch.object(repair.subprocess, "check_call"),
        ):
            with self.assertRaisesRegex(
                repair.SQLAlchemyImportError, "still cannot be imported"
            ):
                repair.ensure_sqlalchemy_import(
                    "venv-python", root=Path("repo"), platform="win32"
                )

    def test_invalid_installed_version_is_not_passed_to_pip(self):
        with (
            patch.object(
                repair.subprocess,
                "run",
                side_effect=[
                    completed(1, stderr=APP_CONTROL_FAILURE),
                    completed(0, stdout="2.0.44 @ https://attacker.invalid/pkg\n"),
                ],
            ),
            patch.object(repair.subprocess, "check_call") as install,
        ):
            with self.assertRaisesRegex(
                repair.SQLAlchemyImportError, "Cannot safely select"
            ):
                repair.ensure_sqlalchemy_import(
                    "venv-python", root=Path("repo"), platform="win32"
                )
        install.assert_not_called()


class BootstrapWiringTests(unittest.TestCase):
    def test_windows_launcher_runs_helper_before_starting_jarvis(self):
        source = (repair.ROOT / "scripts" / "start_jarvis.bat").read_text(
            encoding="utf-8"
        )
        helper = source.index("python scripts\\sqlalchemy_app_control.py")
        launch = source.index("python main.py")
        self.assertLess(helper, launch)
        self.assertIn("if errorlevel 1 exit /b 1", source[helper:launch])

    def test_cross_platform_setup_uses_helper_after_requirements(self):
        source = (repair.ROOT / "scripts" / "setup_jarvis.py").read_text(
            encoding="utf-8"
        )
        requirements = source.index('"-r", "requirements.txt"')
        helper = source.index("ensure_sqlalchemy_import(str(PYTHON), root=ROOT)")
        editable = source.index('"-e", "."')
        self.assertLess(requirements, helper)
        self.assertLess(helper, editable)


if __name__ == "__main__":
    unittest.main()
