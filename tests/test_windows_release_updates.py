"""Verified packaged update staging, owner approval and recovery contracts.

These tests exercise the lifecycle with real staged bytes and a fake NSIS
adapter. They do not claim physical Windows installation or phone biometrics.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.nexus.event_store import EventStore
from core.nexus.owner_approval import OwnerApprovalManager
from core.update_manager import (
    GitHubReleaseSource, UpdateArtifactError, UpdateClass, UpdatePlan,
    UpdateState, WindowsReleaseUpdateCoordinator,
)


def executable(marker: bytes) -> bytes:
    result = bytearray(128)
    result[:2] = b"MZ"
    result[60:64] = (64).to_bytes(4, "little")
    result[64:68] = b"PE\x00\x00"
    result.extend(marker)
    return bytes(result)


def release(version="2.1.0", commit="a" * 40, marker=b"new"):
    artifact = executable(marker)
    return UpdatePlan(version, commit, hashlib.sha256(artifact).hexdigest(),
                      ("core/update_manager.py",), "Verified update", UpdateClass.MAJOR,
                      "v" + version + "-" + commit[:8], "JARVIS-Setup.exe"), artifact


class Source:
    def __init__(self):
        self.new, new_artifact = release()
        self.old, old_artifact = release("2.0.0", "b" * 40, b"old")
        self.artifacts = {self.new.digest(): new_artifact, self.old.digest(): old_artifact}

    def find_version(self, version, *, commit_sha=None):
        if version != self.old.version or commit_sha != self.old.commit_sha:
            raise UpdateArtifactError("No recovery build")
        return self.old

    def download(self, plan, destination):
        destination.write_bytes(self.artifacts[plan.digest()])


class WindowsReleaseUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = Source()
        self.sidecar = self.root / "installed-brain.exe"
        self.sidecar.write_bytes(b"existing trusted sidecar")
        self.approvals = OwnerApprovalManager(EventStore(self.root / "nexus", "desktop"))
        self.updates = WindowsReleaseUpdateCoordinator(
            self.source, state_dir=self.root, installed_version=self.source.old.version,
            installed_sidecar=self.sidecar, installed_commit_sha=self.source.old.commit_sha,
            owner_approvals=self.approvals,
        )

    def tearDown(self):
        self.temp.cleanup()

    def ready(self):
        stage = self.updates.prepare(self.source.new)
        self.assertEqual(stage.state, UpdateState.AWAITING_APPROVAL)
        self.approvals.decide(stage.approval_id, peer_id="phone", approved=True, user_verified=True)
        commands = []
        with patch("core.update_manager.sys.platform", "win32"), patch("core.update_manager.sys.frozen", True, create=True):
            handoff = self.updates.authorize_handoff(stage.checkpoint_id, approval_id=stage.approval_id,
                                                    parent_pid=42, launcher=commands.append)
        self.assertEqual(handoff.state, UpdateState.APPLYING)
        self.assertEqual(commands[0][1:], ["--install-update", str(self.root), stage.checkpoint_id, "42"])
        return stage

    def test_prepare_checks_candidate_and_recovery_before_requesting_approval(self):
        stage = self.updates.prepare(self.source.new)
        self.assertEqual(stage.state, UpdateState.AWAITING_APPROVAL)
        folder = self.root / "updates" / stage.checkpoint_id
        self.assertTrue((folder / "candidate.exe").is_file())
        self.assertTrue((folder / "recovery.exe").is_file())
        self.assertEqual(self.approvals.get(stage.approval_id).action_digest, self.source.new.digest())

    def test_success_verifies_the_installed_binary_version_and_commit(self):
        stage = self.ready()
        installed = []
        probes = []
        def self_test(path, version, commit):
            probes.append((path, version, commit))
            return True
        result = self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                              wait_for_exit=lambda pid: pid in {42, os.getpid()},
                                              run_installer=installed.append, self_test=self_test)
        self.assertEqual(result.state, UpdateState.APPLIED)
        self.assertEqual([path.name for path in installed], ["candidate.exe"])
        self.assertEqual(probes, [(self.sidecar, "2.1.0", "a" * 40)])
        self.assertEqual(self.updates.history()[0]["state"], "applied")

    def test_failed_new_binary_recovers_and_verifies_the_previous_version(self):
        stage = self.ready()
        installed = []
        result = self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                              wait_for_exit=lambda _: True, run_installer=installed.append,
                                              self_test=lambda path, version, commit: commit == self.source.old.commit_sha)
        self.assertEqual(result.state, UpdateState.ROLLED_BACK)
        self.assertEqual([path.name for path in installed], ["candidate.exe", "recovery.exe"])

    def test_failed_recovery_is_reported_without_claiming_rollback(self):
        stage = self.ready()
        result = self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                              wait_for_exit=lambda _: True, run_installer=lambda _: None,
                                              self_test=lambda *args: False)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertIn("recovery failed", result.message)

    def test_desktop_shutdown_is_required_before_mutating_the_installation(self):
        stage = self.ready()
        installed = []
        result = self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                              wait_for_exit=lambda _: False, run_installer=installed.append,
                                              self_test=lambda *args: True)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertEqual(installed, [])

    def test_original_sidecar_must_exit_before_installer_can_replace_it(self):
        stage = self.ready()
        installed = []
        result = self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                              wait_for_exit=lambda pid: pid == 42,
                                              run_installer=installed.append,
                                              self_test=lambda *args: True)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertEqual(installed, [])

    def test_symlinked_update_state_cannot_escape_the_owner_state_directory(self):
        target = self.root / "elsewhere"
        target.mkdir()
        other = self.root / "second-owner"
        other.mkdir()
        try:
            (other / "updates").symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("Symlink creation is unavailable on this platform")
        with self.assertRaises(UpdateArtifactError):
            WindowsReleaseUpdateCoordinator(self.source, state_dir=other,
                                            installed_version=self.source.old.version,
                                            installed_sidecar=self.sidecar)

    def test_altered_candidate_does_not_consume_owner_approval(self):
        stage = self.updates.prepare(self.source.new)
        self.approvals.decide(stage.approval_id, peer_id="phone", approved=True, user_verified=True)
        (self.root / "updates" / stage.checkpoint_id / "candidate.exe").write_bytes(b"tampered")
        with patch("core.update_manager.sys.platform", "win32"), patch("core.update_manager.sys.frozen", True, create=True):
            with self.assertRaises(UpdateArtifactError):
                self.updates.authorize_handoff(stage.checkpoint_id, approval_id=stage.approval_id,
                                              parent_pid=42, launcher=lambda _: None)
        self.assertFalse(self.approvals.get(stage.approval_id).consumed)

    def test_unapproved_helper_is_rejected(self):
        stage = self.updates.prepare(self.source.new)
        with self.assertRaises(PermissionError):
            self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                         wait_for_exit=lambda _: True, run_installer=lambda _: None,
                                         self_test=lambda *args: True)

    def test_authorized_helper_cannot_be_replayed(self):
        stage = self.ready()
        for first in (True, False):
            if first:
                self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                             wait_for_exit=lambda _: True, run_installer=lambda _: None,
                                             self_test=lambda *args: True)
            else:
                with self.assertRaises(PermissionError):
                    self.updates.complete_handoff(stage.checkpoint_id, parent_pid=42,
                                                 wait_for_exit=lambda _: True, run_installer=lambda _: None,
                                                 self_test=lambda *args: True)

    def test_recovery_artifact_must_match_the_exact_installed_commit(self):
        self.updates.installed_commit_sha = "f" * 40
        result = self.updates.prepare(self.source.new)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertEqual(self.approvals.pending(), ())

    def test_same_version_with_a_new_commit_is_still_an_update(self):
        replacement, payload = release("2.0.0", "c" * 40, b"same version new code")
        self.source.artifacts[replacement.digest()] = payload
        result = self.updates.prepare(replacement)
        self.assertEqual(result.state, UpdateState.AWAITING_APPROVAL)

    def test_unknown_old_build_provenance_blocks_installation(self):
        self.updates.installed_commit_sha = None
        result = self.updates.prepare(self.source.new)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertIn("no commit provenance", result.message)

    def test_correct_digest_of_a_non_pe_download_is_still_rejected(self):
        invalid = b"This is not an executable"
        altered = UpdatePlan.from_dict({**self.source.new.to_dict(), "artifact_sha256": hashlib.sha256(invalid).hexdigest()})
        self.source.artifacts[altered.digest()] = invalid
        result = self.updates.prepare(altered)
        self.assertEqual(result.state, UpdateState.FAILED)
        self.assertEqual(self.approvals.pending(), ())

    def test_non_windows_source_checkout_cannot_install_nsis(self):
        stage = self.updates.prepare(self.source.new)
        with patch("core.update_manager.sys.platform", "linux"), self.assertRaises(UpdateArtifactError):
            self.updates.authorize_handoff(stage.checkpoint_id, approval_id=stage.approval_id,
                                          parent_pid=42, launcher=lambda _: None)


class ReleaseSourceTests(unittest.TestCase):
    def setUp(self):
        self.source = GitHubReleaseSource()
        self.plan, self.artifact = release()
        self.manifest = {"name": "jarvis-update.json", "browser_download_url":
                         f"https://github.com/{self.source.repository}/releases/download/{self.plan.release_tag}/jarvis-update.json"}
        self.asset = {"name": "JARVIS-Setup.exe", "browser_download_url":
                      f"https://github.com/{self.source.repository}/releases/download/{self.plan.release_tag}/JARVIS-Setup.exe",
                      "digest": "sha256:" + self.plan.artifact_sha256}
        self.release = {"draft": False, "tag_name": self.plan.release_tag, "assets": [self.manifest, self.asset],
                        "published_at": "2026-10-04T07:10:45Z"}

    def metadata(self, url):
        if url.endswith("jarvis-update.json"):
            return self.plan.to_dict()
        if "/git/ref/tags/" in url:
            return {"object": {"type": "commit", "sha": self.plan.commit_sha}}
        return self.release

    def test_release_tag_must_identify_the_manifest_commit(self):
        with patch.object(self.source, "_json", side_effect=self.metadata):
            self.assertEqual(self.source._release_plan(self.release), self.plan)
        def changed(url):
            if "/git/ref/tags/" in url:
                return {"object": {"type": "commit", "sha": "f" * 40}}
            return self.metadata(url)
        with patch.object(self.source, "_json", side_effect=changed), self.assertRaises(UpdateArtifactError):
            self.source._release_plan(self.release)

    def test_discover_returns_the_published_verified_release(self):
        def metadata(url):
            if "/releases?" in url:
                return [{"draft": True, "assets": []}, self.release]
            return self.metadata(url)
        with patch.object(self.source, "_json", side_effect=metadata):
            self.assertEqual(self.source.discover(), self.plan)

    def test_discover_uses_publication_time_when_api_lists_older_release_first(self):
        older = {**self.release, "published_at": "2026-10-04T06:35:40Z", "tag_name": "older"}
        with patch.object(self.source, "_json", return_value=[older, self.release]), \
                patch.object(self.source, "_release_plan", return_value=self.plan) as release_plan:
            self.assertEqual(self.source.discover(), self.plan)
        release_plan.assert_called_once_with(self.release)

    def test_discover_compares_timezone_offsets_as_instants(self):
        older = {**self.release, "published_at": "2026-10-04T10:00:00+03:00", "tag_name": "older"}
        with patch.object(self.source, "_json", return_value=[older, self.release]), \
                patch.object(self.source, "_release_plan", return_value=self.plan) as release_plan:
            self.assertEqual(self.source.discover(), self.plan)
        release_plan.assert_called_once_with(self.release)

    def test_discover_rejects_unknown_publication_time_without_older_fallback(self):
        for timestamp in (None, 42, "", "not-a-date", "2026-10-04T07:10:45",
                          "2026-02-30T07:10:45Z", "2026-10-04T07:10:45+99:00"):
            with self.subTest(timestamp=timestamp):
                invalid = {**self.release, "published_at": timestamp}
                with patch.object(self.source, "_json", return_value=[self.release, invalid]), \
                        patch.object(self.source, "_release_plan") as release_plan, \
                        self.assertRaises(UpdateArtifactError):
                    self.source.discover()
                release_plan.assert_not_called()

    def test_discover_does_not_fall_back_when_latest_manifest_is_invalid(self):
        older = {**self.release, "published_at": "2026-10-04T06:35:40Z", "tag_name": "older"}
        with patch.object(self.source, "_json", return_value=[older, self.release]), \
                patch.object(self.source, "_release_plan", side_effect=UpdateArtifactError("invalid manifest")) as release_plan, \
                self.assertRaises(UpdateArtifactError):
            self.source.discover()
        release_plan.assert_called_once_with(self.release)

    def test_recovery_lookup_skips_unrelated_pinned_builds_before_fetching_manifests(self):
        def metadata(url):
            if "/releases?" in url:
                return [{**self.release, "target_commitish": "f" * 40},
                        {**self.release, "target_commitish": self.plan.commit_sha}]
            return self.metadata(url)
        with patch.object(self.source, "_json", side_effect=metadata):
            self.assertEqual(self.source.find_version(self.plan.version, commit_sha=self.plan.commit_sha), self.plan)

    def test_external_asset_urls_cannot_enter_the_download_path(self):
        unsafe = {**self.asset, "browser_download_url": "https://example.com/installer.exe"}
        with self.assertRaises(UpdateArtifactError):
            self.source._asset_url(unsafe, self.plan.release_tag)

    def test_redirects_to_untrusted_hosts_are_rejected(self):
        class Response:
            status_code = 302
            headers = {"Location": "https://example.com/installer.exe"}
            def close(self):
                pass
        with patch.object(self.source.session, "get", return_value=Response()) as get:
            with self.assertRaises(UpdateArtifactError):
                self.source._response(self.asset["browser_download_url"])
            self.assertEqual(get.call_count, 1)

    def test_download_rechecks_sha256_and_removes_corrupted_partial_file(self):
        class Response:
            def iter_content(self, *args):
                yield b"corrupted installer"
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary, patch.object(self.source, "_json", side_effect=self.metadata), patch.object(self.source, "_response", return_value=Response()):
            target = Path(temporary) / "candidate.exe"
            with self.assertRaises(UpdateArtifactError):
                self.source.download(self.plan, target)
            self.assertFalse(target.exists())

    def test_public_download_auth_removes_session_tokens(self):
        import requests
        request = requests.Request("GET", self.asset["browser_download_url"],
                                   headers={"Authorization": "Bearer secret", "Cookie": "session=secret"}).prepare()
        self.source.session.auth(request)
        self.assertNotIn("Authorization", request.headers)
        self.assertNotIn("Cookie", request.headers)
        self.assertTrue(self.source.session.trust_env)


if __name__ == "__main__":
    unittest.main()
