"""Packaged provenance and early updater modes without starting the Brain."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import brain_sidecar
from core.build_info import read_build_info
from core.update_manager import UpdateOutcome, UpdateState


INFO = {"version": "2.0.1", "commit_sha": "a" * 40}


class BuildInfoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "build-info.json"

    def test_provenance_preserves_exact_commit_and_drops_unrelated_fields(self):
        self.path.write_text(json.dumps({**INFO, "internal_path": "ignored"}), encoding="utf-8")
        self.assertEqual(read_build_info(self.path), INFO)

    def test_invalid_or_missing_provenance_is_rejected(self):
        invalid = (
            [], {}, {**INFO, "version": "latest"}, {**INFO, "version": 2},
            {**INFO, "commit_sha": "short"}, {**INFO, "commit_sha": "A" * 40},
            {**INFO, "commit_sha": "g" * 40}, {**INFO, "commit_sha": None},
        )
        for value in invalid:
            with self.subTest(value=value):
                self.path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_build_info(self.path)
        self.path.write_text("not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_build_info(self.path)
        self.path.unlink()
        with self.assertRaises(OSError):
            read_build_info(self.path)

    def test_provenance_byte_budget_rejects_unbounded_input(self):
        self.path.write_bytes(b" " * 16385)
        with self.assertRaisesRegex(ValueError, "too large"):
            read_build_info(self.path)


class SidecarUpdateModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metadata = self.root / "build-info.json"
        self.metadata.write_text(json.dumps(INFO), encoding="utf-8")
        self.output = self.root / "self-test.json"
        self.host = Mock(side_effect=AssertionError("early update modes must not start a normal host"))
        self.server = Mock(side_effect=AssertionError("early update modes must not start a server"))
        self.parser = Mock(side_effect=AssertionError("early update modes must not require normal host arguments"))
        self.gateway = Mock(side_effect=AssertionError("early update modes must not inspect companion networking"))
        for target, value in (
            ("brain_sidecar.LocalBrainHost", self.host),
            ("brain_sidecar.uvicorn.run", self.server),
            ("brain_sidecar._parser", self.parser),
            ("brain_sidecar.resolve_companion_gateway", self.gateway),
        ):
            context = patch(target, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch("core.build_info.resource_path", return_value=self.metadata)
        context.start()
        self.addCleanup(context.stop)
        context = patch("core.ollama_bootstrap.load_brain_manifest", return_value={})
        self.manifest = context.start()
        self.addCleanup(context.stop)

    def assert_no_host(self):
        self.host.assert_not_called()
        self.server.assert_not_called()
        self.parser.assert_not_called()
        self.gateway.assert_not_called()

    def test_self_test_writes_file_with_exact_packaged_version_and_commit(self):
        code = brain_sidecar.main(["--update-self-test", str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.output.read_text("utf-8")), {"ok": True, **INFO})
        self.manifest.assert_called_once_with()
        self.assert_no_host()

    def test_invalid_or_missing_metadata_never_creates_a_success_result(self):
        for value in ("not json", json.dumps({**INFO, "commit_sha": "wrong"}), None):
            with self.subTest(metadata=value):
                if value is None:
                    self.metadata.unlink(missing_ok=True)
                else:
                    self.metadata.write_text(value, encoding="utf-8")
                self.assertEqual(brain_sidecar.main(["--update-self-test", str(self.output)]), 1)
                self.assertFalse(self.output.exists())
        self.assert_no_host()

    def test_invalid_brain_manifest_cannot_report_self_test_success(self):
        self.manifest.side_effect = ValueError("invalid packaged manifest")
        self.assertEqual(brain_sidecar.main(["--update-self-test", str(self.output)]), 1)
        self.assertFalse(self.output.exists())
        self.assert_no_host()

    def test_self_test_does_not_overwrite_existing_files_or_follow_links(self):
        self.output.write_text("existing private data", encoding="utf-8")
        self.assertEqual(brain_sidecar.main(["--update-self-test", str(self.output)]), 1)
        self.assertEqual(self.output.read_text("utf-8"), "existing private data")
        link = self.root / "linked-result.json"
        link.symlink_to(self.output)
        self.assertEqual(brain_sidecar.main(["--update-self-test", str(link)]), 2)
        self.assertEqual(self.output.read_text("utf-8"), "existing private data")
        self.assert_no_host()

    def test_self_test_rejects_relative_output_and_wrong_argument_counts(self):
        for argv in (
            ["--update-self-test"], ["--update-self-test", str(self.output), "extra"],
            ["--update-self-test", "relative-result.json"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(brain_sidecar.main(argv), 2)
                self.assertFalse(self.output.exists())
        self.assert_no_host()

    def test_install_mode_delegates_pinned_state_and_parent_without_normal_boot(self):
        checkpoint = "b" * 32
        applied = UpdateOutcome(UpdateState.APPLIED, "verified installed")
        with patch("core.update_manager.run_windows_update_helper", return_value=applied) as helper:
            self.assertEqual(brain_sidecar.main(["--install-update", str(self.root), checkpoint, "42"]), 0)
            helper.assert_called_once_with(str(self.root), checkpoint, 42)
        self.assert_no_host()

    def test_install_failure_or_recovery_is_not_reported_as_update_success(self):
        for state in (UpdateState.FAILED, UpdateState.ROLLED_BACK):
            with self.subTest(state=state):
                with patch("core.update_manager.run_windows_update_helper", return_value=UpdateOutcome(state, "not applied")):
                    self.assertEqual(brain_sidecar.main(["--install-update", str(self.root), "b" * 32, "42"]), 1)
        with patch("core.update_manager.run_windows_update_helper", side_effect=ValueError("unsafe checkpoint")):
            self.assertEqual(brain_sidecar.main(["--install-update", str(self.root), "b" * 32, "42"]), 1)
        self.assert_no_host()

    def test_invalid_install_arguments_never_reach_the_helper(self):
        checkpoint = "b" * 32
        invalid = (
            ["--install-update"], ["--install-update", str(self.root), checkpoint, "42", "extra"],
            ["--install-update", str(self.root), checkpoint, "not-a-pid"],
            ["--install-update", str(self.root), checkpoint, "0"],
            ["--install-update", str(self.root), checkpoint, "-1"],
            ["--install-update", "relative-state", checkpoint, "42"],
        )
        with patch("core.update_manager.run_windows_update_helper") as helper:
            for argv in invalid:
                with self.subTest(argv=argv):
                    self.assertEqual(brain_sidecar.main(argv), 2)
            helper.assert_not_called()
        self.assert_no_host()


if __name__ == "__main__":
    unittest.main()
