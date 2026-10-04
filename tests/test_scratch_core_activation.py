"""Offline activation contracts; tiny files/fake Ollama, never real inference."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from core import scratch_activation as receipt
from core.ollama_bootstrap import OllamaBootstrapError, model_matches_manifest, provision_brain_models
from training import activate
from training.common import PipelineError, ROOT, sha256_file, write_json
from training.evaluate import PROBES


URL = "http://127.0.0.1:11435"


def valid_receipt():
    return {"schema_version": 1, "state": "activated", "alias": receipt.CORE_ALIAS,
        "endpoint": URL, "parameters": receipt.SCRATCH_PARAMETERS,
        "trainable_parameters": receipt.SCRATCH_PARAMETERS, "context_length": 4096,
        "development_only": False, "initialization": "random_from_config",
        "pretrained_language_weights_loaded": False, "quality_review_acknowledged": True,
        "runtime_smoke_passed": True, "quality_certified": False, "activated_at": "2026-10-04T00:00:00Z",
        **{key: "a" * 64 for key in ("ollama_digest", "gguf_sha256", "training_manifest_sha256",
            "evaluation_sha256", "export_manifest_sha256", "architecture_lock_sha256", "core_template_sha256")}}


def response(value):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)


class FakeOllama:
    """Only transaction metadata and calls are simulated, no model exists."""
    def __init__(self, *, existing=True):
        self.aliases = {receipt.CORE_ALIAS: "b" * 64} if existing else {}
        self.calls = []
        self.modelfile = None
        self.system = None
        self.fail_canonical_probe = False
        self.fail_rollback = False
        self.resident = []

    def run(self, args, **kwargs):
        self.calls.append(args[1:])
        self.assert_loopback(kwargs)
        operation = args[1]
        if operation == "create":
            self.modelfile = Path(args[4]).read_text("utf-8")
            self.system = self.modelfile.split('SYSTEM """\n', 1)[1].rsplit('"""', 1)[0].strip()
            self.aliases[args[2]] = "a" * 64
        elif operation == "cp":
            if self.fail_rollback and args[2].startswith("jarvis-core-backup-"):
                return SimpleNamespace(returncode=1)
            self.aliases[args[3]] = self.aliases[args[2]]
        elif operation == "rm":
            self.aliases.pop(args[2], None)
        return SimpleNamespace(returncode=0)

    def assert_loopback(self, kwargs):
        if kwargs.get("env", {}).get("OLLAMA_HOST") != "127.0.0.1:11435":
            raise AssertionError("CLI endpoint not bound")

    def get(self, url, **kwargs):
        if url.endswith("/api/tags"):
            return response({"models": [{"name": name + ":latest", "digest": digest} for name, digest in self.aliases.items()]})
        if url.endswith("/api/ps"):
            return response({"models": self.resident})
        raise AssertionError(url)

    def post(self, url, *, json, **kwargs):
        model = json["model"]
        if url.endswith("/api/show"):
            return response({"model_info": {"general.parameter_count": receipt.SCRATCH_PARAMETERS,
                "general.architecture": "qwen2", "qwen2.context_length": 4096},
                "details": {"parameter_size": "1.5B"}, "capabilities": ["completion"],
                "system": self.system, "parameters": "temperature 0.2\ntop_p 0.85\nnum_ctx 4096\n",
                "template": "<|im_start|>{{ .Prompt }}<|im_end|>"})
        if url.endswith("/api/chat"):
            content = "" if self.fail_canonical_probe and model == receipt.CORE_ALIAS else "test response"
            return response({"done": True, "message": {"role": "assistant", "content": content}})
        if url.endswith("/api/generate"):
            return response({"done": True})
        raise AssertionError(url)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_endpoint_keys_distinguish_ports_and_preserve_literal_ipv6(self):
        self.assertEqual(receipt.activation_endpoint("http://[::1]:11435/"), "http://[::1]:11435")
        self.assertEqual(receipt.activation_endpoint("http://127.0.0.1"), "http://127.0.0.1:11434")
        self.assertNotEqual(receipt.receipt_path(URL, directory=self.directory),
                            receipt.receipt_path("http://127.0.0.1:11434", directory=self.directory))

    def test_external_origins_credentials_and_invalid_ports_are_rejected(self):
        for url in ("http://localhost:11435", "http://example.org", "https://127.0.0.1", "http://u:p@127.0.0.1",
                    URL + "/api", URL + "?x=1", URL + "#x", "http://127.0.0.1:0", "http://127.0.0.1:99999"):
            with self.subTest(url=url), self.assertRaises(receipt.ScratchActivationError):
                receipt.activation_endpoint(url)

    def test_missing_receipt_and_atomic_round_trip(self):
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))
        path = receipt.write_receipt(valid_receipt(), URL, directory=self.directory)
        self.assertEqual(receipt.read_receipt(URL, directory=self.directory), valid_receipt())
        self.assertEqual(list(self.directory.glob(".activation-*")), [])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_present_invalid_evidence_fails_closed(self):
        changes = [("development_only", True), ("parameters", 1_000_000_000), ("trainable_parameters", True),
            ("endpoint", "http://127.0.0.1:11434"), ("context_length", 8192), ("quality_certified", True),
            ("pretrained_language_weights_loaded", True), ("runtime_smoke_passed", False), ("gguf_sha256", "x" * 64)]
        for key, value in changes:
            item = dict(valid_receipt(), **{key: value})
            write_json(receipt.receipt_path(URL, directory=self.directory), item)
            with self.subTest(key=key), self.assertRaises(receipt.ScratchActivationError):
                receipt.read_receipt(URL, directory=self.directory)

    def test_malformed_oversized_symlink_receipts_are_rejected(self):
        path = receipt.receipt_path(URL, directory=self.directory)
        for value in ("{", "[]", "x" * 65_537):
            path.write_text(value)
            with self.assertRaises(receipt.ScratchActivationError):
                receipt.read_receipt(URL, directory=self.directory)
        path.unlink()
        target = self.directory / "target.json"
        write_json(target, valid_receipt())
        path.symlink_to(target)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_cooperative_lock_blocks_reentry_and_releases_on_failure(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with receipt.activation_lock(URL, directory=self.directory):
                with self.assertRaises(receipt.ScratchActivationError):
                    receipt.assert_activation_unlocked(URL, directory=self.directory)
                with self.assertRaises(receipt.ScratchActivationError):
                    with receipt.activation_lock(URL, directory=self.directory):
                        pass
                raise RuntimeError("test failure")
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_lock_never_removes_another_owner_and_can_preserve_recovery_state(self):
        with receipt.activation_lock(URL, directory=self.directory):
            path = receipt.activation_lock_path(URL, directory=self.directory)
            path.write_text("another owner")
        self.assertEqual(path.read_text(), "another owner")
        path.unlink()
        with receipt.activation_lock(URL, directory=self.directory) as lease:
            lease["release"] = False
        self.assertTrue(path.exists())

    def test_runtime_binding_rejects_changed_digest_small_count_missing_capability(self):
        daemon = FakeOllama(existing=False)
        daemon.aliases[receipt.CORE_ALIAS] = "a" * 64
        receipt.verify_receipt_runtime(valid_receipt(), URL, request_get=daemon.get, request_post=daemon.post)
        daemon.aliases[receipt.CORE_ALIAS] = "c" * 64
        with self.assertRaisesRegex(receipt.ScratchActivationError, "differs"):
            receipt.verify_receipt_runtime(valid_receipt(), URL, request_get=daemon.get, request_post=daemon.post)
        for body in ({"model_info": {"general.parameter_count": 1_000_000_000}, "capabilities": ["completion"]},
                     {"model_info": {"general.parameter_count": receipt.SCRATCH_PARAMETERS}, "capabilities": ["embedding"]}):
            with self.assertRaises(receipt.ScratchActivationError):
                receipt.runtime_model_identity(receipt.CORE_ALIAS, URL, request_get=daemon.get, request_post=lambda *a, **k: response(body))

    def test_recovery_marker_is_persistent_nonready_evidence(self):
        receipt.write_recovery_marker(URL, "jarvis-core-backup-test", directory=self.directory)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_receipt_snapshot_restore_changes_only_the_attempted_transaction(self):
        previous = dict(valid_receipt(), ollama_digest="b" * 64)
        attempted = valid_receipt()
        receipt.write_receipt(attempted, URL, directory=self.directory)
        receipt.restore_receipt_snapshot(previous, attempted, URL, directory=self.directory)
        self.assertEqual(receipt.read_receipt(URL, directory=self.directory), previous)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.restore_receipt_snapshot(None, attempted, URL, directory=self.directory)
        receipt.restore_receipt_snapshot(None, previous, URL, directory=self.directory)
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))


class BootstrapActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch("core.scratch_activation.user_data_dir", return_value=self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_no_receipt_keeps_baseline_ipv4_and_ipv6_matching(self):
        post = Mock(return_value=response({"details": {"parameter_size": "1.0B"}}))
        for url in (URL, "http://[::1]:11435"):
            with self.subTest(url=url):
                self.assertTrue(model_matches_manifest(receipt.CORE_ALIAS, 1.0, base_url=url, request_post=post))
        with patch("core.ollama_bootstrap.installed_model_names", return_value={receipt.CORE_ALIAS}), \
                patch("core.ollama_bootstrap.model_matches_manifest", side_effect=lambda model, size, base_url:
                      model_matches_manifest(model, size, base_url=base_url, request_post=post)):
            ready, _ = provision_brain_models("ollama", base_url="http://[::1]:11435")
        self.assertEqual(ready, (receipt.CORE_ALIAS,))

    def test_verified_scratch_receipt_accepts_real_count_and_prevents_baseline_pull_even_if_list_empty(self):
        receipt.write_receipt(valid_receipt(), URL)
        daemon = FakeOllama(existing=False)
        daemon.aliases[receipt.CORE_ALIAS] = "a" * 64
        self.assertTrue(model_matches_manifest(receipt.CORE_ALIAS, 1.0, base_url=URL,
            request_get=daemon.get, request_post=daemon.post))
        runner = Mock()
        with patch("core.ollama_bootstrap.installed_model_names", return_value=set()), \
                patch("core.ollama_bootstrap.requests.get", daemon.get), patch("core.ollama_bootstrap.requests.post", daemon.post):
            ready, _ = provision_brain_models("ollama", base_url=URL, runner=runner)
        self.assertEqual(ready, (receipt.CORE_ALIAS,))
        runner.assert_not_called()

    def test_invalid_receipt_lock_and_changed_runtime_all_refuse_provisioning(self):
        path = receipt.receipt_path(URL)
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        runner = Mock()
        self.assertFalse(model_matches_manifest(receipt.CORE_ALIAS, 1.0, base_url=URL))
        with self.assertRaises(OllamaBootstrapError):
            provision_brain_models("ollama", base_url=URL, runner=runner)
        path.unlink()
        with receipt.activation_lock(URL):
            self.assertFalse(model_matches_manifest(receipt.CORE_ALIAS, 1.0, base_url=URL))
            with self.assertRaises(OllamaBootstrapError):
                provision_brain_models("ollama", base_url=URL, runner=runner)
        receipt.write_receipt(valid_receipt(), URL)
        daemon = FakeOllama()  # Existing alias digest differs from receipt.
        with patch("core.ollama_bootstrap.requests.get", daemon.get), patch("core.ollama_bootstrap.requests.post", daemon.post):
            with self.assertRaises(OllamaBootstrapError):
                provision_brain_models("ollama", base_url=URL, runner=runner)
        runner.assert_not_called()

    def test_transient_runtime_error_after_initial_receipt_check_never_recreates_core(self):
        receipt.write_receipt(valid_receipt(), URL)
        runner = Mock()
        with patch("core.ollama_bootstrap.verify_receipt_runtime", side_effect=[None, receipt.ScratchActivationError("unreachable")]):
            with self.assertRaises(OllamaBootstrapError):
                provision_brain_models("ollama", base_url=URL, runner=runner)
        self.assertFalse(any(call.args[0][1] in {"pull", "create"} for call in runner.call_args_list))

    def test_receipt_disappearing_after_initial_verification_never_recreates_core(self):
        runner = Mock()
        with patch("core.ollama_bootstrap.read_receipt", side_effect=[valid_receipt(), None]), \
                patch("core.ollama_bootstrap.verify_receipt_runtime"), \
                patch("core.ollama_bootstrap.installed_model_names", return_value=set()):
            with self.assertRaises(OllamaBootstrapError):
                provision_brain_models("ollama", base_url=URL, runner=runner)
        runner.assert_not_called()


class ProductionExportTests(unittest.TestCase):
    """A six-parameter mock lock validates actual tiny checkpoint/hash bytes."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run, self.export, self.corpus = [self.root / name for name in ("run", "export", "corpus")]
        for folder in (self.run / "checkpoint", self.export, self.corpus):
            folder.mkdir(parents=True)
        self.lock = {"expected_parameters": 6, "planning_tokens_per_parameter": 20,
                     "architecture": {"model_type": "fixture", "max_position_embeddings": 4}, "tokenizer": {"fixture": True}}
        tensor = {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        header = json.dumps(tensor).encode()
        weights = self.run / "checkpoint" / "model.safetensors"
        weights.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
        write_json(self.run / "checkpoint" / "config.json", self.lock["architecture"])
        counts = {"train": 120, "validation": 8, "test": 8}
        for split, count in counts.items():
            (self.corpus / f"{split}.tokens").write_bytes(bytes(count * 4))
        self.data = {"architecture_lock_sha256": sha256_file(ROOT / "architecture.lock.json"), "tokenizer": self.lock["tokenizer"],
            "token_dtype": "uint32-little-endian", "token_counts": counts, "unreviewed_records": 0,
            "token_files": {split: sha256_file(self.corpus / f"{split}.tokens") for split in counts}}
        write_json(self.corpus / "corpus-manifest.json", self.data)
        self.manifest = {"status": "trained", "global_steps": 15, "initialization": "random_from_config",
            "pretrained_language_weights_loaded": False, "parameters": 6, "trainable_parameters": 6,
            "sampled_update_nonzero": True, "development_only": False,
            "checkpoint_files": {"model.safetensors": sha256_file(weights)}, "tokenizer": self.lock["tokenizer"],
            "architecture_lock_sha256": sha256_file(ROOT / "architecture.lock.json"), "corpus": str(self.corpus),
            "corpus_manifest_sha256": sha256_file(self.corpus / "corpus-manifest.json"), "tokens_seen": 120,
            "sequence_length": 4, "batch_size": 1, "gradient_accumulation": 2}
        write_json(self.run / "training-manifest.json", self.manifest)
        self.evaluation = {"status": "completed", "checkpoint_files": self.manifest["checkpoint_files"],
            "training_manifest_sha256": sha256_file(self.run / "training-manifest.json"), "held_out_loss": 2.0,
            "evaluated_prediction_tokens": 12, "development_only": False, "split": "test",
            "behavioral_probes": [{**probe, "response": probe["expected"], "passed": True} for probe in PROBES]}
        write_json(self.run / "evaluation.json", self.evaluation)
        self.gguf = self.export / "tiny-fixture.gguf"
        self.gguf.write_bytes(b"GGUF" + bytes(24))
        self.report = {"status": "exported", "development_only": False,
            "training_manifest_sha256": sha256_file(self.run / "training-manifest.json"),
            "evaluation_sha256": sha256_file(self.run / "evaluation.json"), "checkpoint_files": self.manifest["checkpoint_files"],
            "behavioral_passes": 4, "gguf_file": self.gguf.name,
            "gguf": {"parameters": 6, "file_bytes": self.gguf.stat().st_size, "sha256": sha256_file(self.gguf)}}
        self.flush()
        for target in ("training.common.architecture_lock", "training.train.architecture_lock", "training.activate.architecture_lock"):
            mocked = patch(target, return_value=self.lock)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch("training.activate.SCRATCH_PARAMETERS", 6)
        mocked.start()
        self.addCleanup(mocked.stop)

    def flush(self):
        write_json(self.export / "export-manifest.json", self.report)

    def check(self):
        return activate.verify_production_export(self.run, self.export)

    def test_verified_tiny_files_bind_all_production_evidence(self):
        evidence = self.check()
        self.assertEqual(evidence["source"], self.gguf)
        self.assertEqual(evidence["gguf_sha256"], sha256_file(self.gguf))

    def test_development_incomplete_wrong_counts_and_changed_hashes_are_rejected(self):
        for key, value in (("development_only", True), ("status", "incomplete"), ("evaluation_sha256", "b" * 64),
                           ("training_manifest_sha256", "b" * 64), ("behavioral_passes", 3), ("gguf_file", "../other.gguf")):
            original = deepcopy(self.report)
            self.report[key] = value
            self.flush()
            with self.subTest(key=key), self.assertRaises(PipelineError):
                self.check()
            self.report = original
        self.report["gguf"]["parameters"] = 5
        self.flush()
        with self.assertRaises(PipelineError):
            self.check()

    def test_changed_real_gguf_checkpoint_or_corpus_bytes_are_rejected(self):
        for file in (self.gguf, self.run / "checkpoint" / "model.safetensors", self.corpus / "test.tokens"):
            original = file.read_bytes()
            file.write_bytes(original + b"changed")
            with self.subTest(file=file.name), self.assertRaises(PipelineError):
                self.check()
            file.write_bytes(original)

    def test_unreviewed_or_insufficient_training_cannot_be_promoted_by_export_labels(self):
        for key, value in (("development_only", True), ("tokens_seen", 119), ("trainable_parameters", 5),
                           ("pretrained_language_weights_loaded", True)):
            original = deepcopy(self.manifest)
            write_json(self.run / "training-manifest.json", dict(self.manifest, **{key: value}))
            with self.subTest(key=key), self.assertRaises(PipelineError):
                self.check()
            write_json(self.run / "training-manifest.json", original)

    def test_symlink_artifact_and_relative_root_are_rejected(self):
        original = self.gguf.read_bytes()
        target = self.root / "elsewhere.gguf"
        target.write_bytes(original)
        self.gguf.unlink()
        self.gguf.symlink_to(target)
        with self.assertRaises(PipelineError):
            self.check()
        with self.assertRaises(PipelineError):
            activate.verify_production_export(Path("relative"), self.export)


class ActivationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "fixture.gguf"
        self.source.write_bytes(b"GGUF" + bytes(24))
        self.directory = self.root / "receipts"
        self.daemon = FakeOllama()
        self.closed_check = Mock()
        self.evidence = {"source": self.source, "gguf_sha256": sha256_file(self.source),
            **{key: "a" * 64 for key in ("training_manifest_sha256", "evaluation_sha256", "export_manifest_sha256", "architecture_lock_sha256")}}
        self.verify = patch("training.activate.verify_production_export", return_value=self.evidence)
        self.verify.start()
        self.addCleanup(self.verify.stop)

    def call(self, **changes):
        return activate.activate(self.root / "run", self.root / "export",
            **dict({"acknowledge_quality_review": True, "acknowledge_jarvis_closed": True,
                "executable": "ollama", "runner": self.daemon.run, "request_get": self.daemon.get,
                "request_post": self.daemon.post, "receipt_directory": self.directory,
                "closed_check": self.closed_check}, **changes))

    def test_stages_verifies_backs_up_promotes_and_commits_bound_receipt(self):
        result = self.call()
        self.assertTrue(result["activated"])
        self.assertFalse(result["quality_certified"])
        saved = receipt.read_receipt(URL, directory=self.directory)
        self.assertEqual(saved["ollama_digest"], self.daemon.aliases[receipt.CORE_ALIAS])
        self.assertEqual(self.daemon.aliases[result["backup_alias"]], "b" * 64)
        self.assertIn("PARAMETER num_ctx 4096", self.daemon.modelfile)
        self.assertIn("PARAMETER temperature 0.2", self.daemon.modelfile)
        self.assertIn("PARAMETER top_p 0.85", self.daemon.modelfile)
        self.assertNotIn("FROM llama", self.daemon.modelfile)
        self.assertNotIn("TEMPLATE", self.daemon.modelfile)
        self.assertEqual(self.closed_check.call_count, 3)
        self.assertFalse(any(call[0] == "pull" for call in self.daemon.calls))
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_missing_acknowledgments_remote_url_ipv6_cli_and_active_jarvis_never_mutate(self):
        for kwargs in ({"acknowledge_quality_review": False}, {"acknowledge_jarvis_closed": False},
                       {"ollama_url": "http://example.org"}, {"ollama_url": "http://[::1]:11435"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((PipelineError, receipt.ScratchActivationError)):
                self.call(**kwargs)
        self.closed_check.side_effect = PipelineError("still running")
        with self.assertRaisesRegex(PipelineError, "still running"):
            self.call()
        self.assertEqual(self.daemon.calls, [])

    def test_failed_canonical_inference_restores_prior_core_and_does_not_write_receipt(self):
        self.daemon.fail_canonical_probe = True
        with self.assertRaisesRegex(PipelineError, "preserved/restored"):
            self.call()
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_failed_initial_activation_removes_canonical_when_no_prior_core(self):
        self.daemon.aliases.clear()
        self.daemon.fail_canonical_probe = True
        with self.assertRaises(PipelineError):
            self.call()
        self.assertNotIn(receipt.CORE_ALIAS, self.daemon.aliases)
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))

    def test_failed_rollback_records_fail_closed_marker(self):
        self.daemon.fail_canonical_probe = self.daemon.fail_rollback = True
        with self.assertRaisesRegex(PipelineError, "manual inspection"):
            self.call()
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_failed_rollback_and_marker_persistence_preserve_lock(self):
        self.daemon.fail_canonical_probe = self.daemon.fail_rollback = True
        with patch("training.activate.write_recovery_marker", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(PipelineError, "manual inspection"):
                self.call()
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_failed_receipt_commit_restores_previous_core(self):
        with patch("training.activate.write_receipt", side_effect=OSError("disk full")):
            with self.assertRaises(PipelineError):
                self.call()
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)

    def test_keyboard_interrupt_after_promotion_restores_core_and_propagates(self):
        original = self.daemon.run
        def interrupted(args, **kwargs):
            result = original(args, **kwargs)
            if args[1] == "cp" and args[2].startswith("jarvis-scratch-stage-"):
                raise KeyboardInterrupt()
            return result
        with self.assertRaises(KeyboardInterrupt):
            self.call(runner=interrupted)
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_keyboard_interrupt_with_failed_rollback_keeps_fail_closed_evidence(self):
        original = self.daemon.run
        self.daemon.fail_rollback = True
        def interrupted(args, **kwargs):
            result = original(args, **kwargs)
            if args[1] == "cp" and args[2].startswith("jarvis-scratch-stage-"):
                raise KeyboardInterrupt()
            return result
        with self.assertRaises(KeyboardInterrupt):
            self.call(runner=interrupted)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_keyboard_interrupt_after_receipt_persistence_keeps_new_model_and_receipt_consistent(self):
        def persisted_then_interrupted(value, url, **kwargs):
            receipt.write_receipt(value, url, **kwargs)
            raise KeyboardInterrupt()
        with patch("training.activate.write_receipt", side_effect=persisted_then_interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.call()
        saved = receipt.read_receipt(URL, directory=self.directory)
        self.assertEqual(saved["ollama_digest"], self.daemon.aliases[receipt.CORE_ALIAS])
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "a" * 64)
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_exception_after_receipt_persistence_does_not_restore_only_the_old_alias(self):
        def persisted_then_failed(value, url, **kwargs):
            receipt.write_receipt(value, url, **kwargs)
            raise OSError("interrupted finalization")
        with patch("training.activate.write_receipt", side_effect=persisted_then_failed):
            with self.assertRaisesRegex(PipelineError, "were committed"):
                self.call()
        saved = receipt.read_receipt(URL, directory=self.directory)
        self.assertEqual(saved["ollama_digest"], self.daemon.aliases[receipt.CORE_ALIAS])
        receipt.assert_activation_unlocked(URL, directory=self.directory)

    def test_existing_receipt_is_verified_under_lock_and_preserved_on_failed_promotion(self):
        previous = valid_receipt()
        self.daemon.aliases[receipt.CORE_ALIAS] = previous["ollama_digest"]
        receipt.write_receipt(previous, URL, directory=self.directory)
        self.daemon.fail_canonical_probe = True
        def verify_under_lock(*args, **kwargs):
            self.assertTrue(receipt.activation_lock_path(URL, directory=self.directory).exists())
            return receipt.verify_receipt_runtime(*args, **kwargs)
        with patch("training.activate.verify_receipt_runtime", side_effect=verify_under_lock):
            with self.assertRaises(PipelineError):
                self.call()
        self.assertEqual(receipt.read_receipt(URL, directory=self.directory), previous)
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], previous["ollama_digest"])

    def test_success_exit_code_without_actual_backup_or_rollback_is_rejected(self):
        original = self.daemon.run
        def false_backup(args, **kwargs):
            if args[1] == "cp" and args[3].startswith("jarvis-core-backup-"):
                return SimpleNamespace(returncode=0)
            return original(args, **kwargs)
        with self.assertRaises(PipelineError):
            self.call(runner=false_backup)
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)
        self.assertIsNone(receipt.read_receipt(URL, directory=self.directory))
        self.daemon.fail_canonical_probe = True
        def false_restore(args, **kwargs):
            if args[1] == "cp" and args[2].startswith("jarvis-core-backup-"):
                return SimpleNamespace(returncode=0)
            return original(args, **kwargs)
        with self.assertRaisesRegex(PipelineError, "manual inspection"):
            self.call(runner=false_restore)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_false_success_removing_new_core_without_prior_alias_is_rejected(self):
        self.daemon.aliases.clear()
        self.daemon.fail_canonical_probe = True
        original = self.daemon.run
        def false_remove(args, **kwargs):
            if args[1:3] == ["rm", receipt.CORE_ALIAS]:
                return SimpleNamespace(returncode=0)
            return original(args, **kwargs)
        with self.assertRaisesRegex(PipelineError, "manual inspection"):
            self.call(runner=false_remove)
        with self.assertRaises(receipt.ScratchActivationError):
            receipt.read_receipt(URL, directory=self.directory)

    def test_unload_inference_and_policy_context_must_be_verified_before_promoting(self):
        original = self.daemon.post
        for field, value in (("system", "different policy"), ("parameters", "num_ctx 8192"),
                             ("template", "<|start_header_id|>{{ .Prompt }}")):
            def changed(url, **kwargs):
                result = original(url, **kwargs)
                if url.endswith("/api/show"):
                    return response(dict(result.json(), **{field: value}))
                return result
            with self.subTest(field=field), self.assertRaises(PipelineError):
                self.call(request_post=changed)
            self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)
        self.assertFalse(any(call[0] == "cp" for call in self.daemon.calls))

    def test_changed_source_during_staging_and_stale_cache_prevent_promotion(self):
        with patch("training.activate.sha256_file", return_value="c" * 64):
            with self.assertRaises(PipelineError):
                self.call()
        self.assertFalse(any(call[0] == "cp" for call in self.daemon.calls))
        self.daemon.resident = [{"name": next(name for name in self.daemon.aliases) + ":latest"}]
        # Core resident failure occurs after backup but still before promotion.
        with self.assertRaises(PipelineError):
            self.call()
        self.assertEqual(self.daemon.aliases[receipt.CORE_ALIAS], "b" * 64)

    def test_active_process_detection_does_not_print_command_line_secrets(self):
        process = SimpleNamespace(info={"pid": -1, "name": "jarvis-brain.exe"})
        with self.assertRaisesRegex(PipelineError, "still running"):
            activate.assert_jarvis_closed(process_iter=lambda attrs: [process])
        process = SimpleNamespace(info={"pid": -1, "name": "python"}, cmdline=lambda: ["python", "brain_sidecar.py", "--ui-token", "secret"])
        with self.assertRaisesRegex(PipelineError, "still running") as error:
            activate.assert_jarvis_closed(process_iter=lambda attrs: [process])
        self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
