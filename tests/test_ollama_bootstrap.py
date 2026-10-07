"""Offline tests for bundled Ollama/JARVIS Brain provisioning."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.ollama_bootstrap import (
    OllamaBootstrapError,
    _OWNED_SERVICES,
    _modelfile_expectations,
    _ollama_environment,
    bootstrap_local_brain,
    ensure_ollama_service,
    import_gguf_model,
    load_brain_manifest,
    model_matches_manifest,
    model_matches_pinned_spec,
    parameter_size_b,
    provision_brain_models,
    select_brain_models,
    stop_owned_ollama_service,
    verify_model_inference,
)


def snapshot(total, available):
    return HardwareSnapshot(
        device_id="node",
        total_ram_gb=total,
        available_ram_gb=available,
        cpu_percent=10,
        cpu_count=8,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=100,
        system_pressure=0.2,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, tzinfo=timezone.utc),
    )


class BootstrapTests(unittest.TestCase):
    def test_manifest_has_dedicated_jarvis_core_1b_and_hidden_experts(self):
        specs = load_brain_manifest()
        self.assertEqual(
            {spec.alias for spec in specs},
            {
                "jarvis-core-1b",
                "jarvis-brain-fast",
                "jarvis-brain-engineering",
                "jarvis-brain-lite",
            },
        )
        self.assertTrue(all(spec.modelfile.is_file() for spec in specs))
        core = next(spec for spec in specs if spec.alias == "jarvis-core-1b")
        self.assertEqual(core.role, "coordinator")
        self.assertEqual(core.parameters_b, 1.0)
        for spec in specs:
            self.assertRegex(spec.base_digest, r"^[0-9a-f]{64}$")
            self.assertRegex(spec.weights_digest, r"^[0-9a-f]{64}$")
            self.assertRegex(spec.modelfile_sha256, r"^[0-9a-f]{64}$")

    def test_modelfile_provenance_treats_windows_crlf_as_canonical_lf(self):
        canonical = 'FROM llama3.2:1b\nSYSTEM """\nJARVIS\n"""\n'
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            (models / "test.Modelfile").write_bytes(canonical.replace("\n", "\r\n").encode("utf-8"))
            (models / "manifest.json").write_text(json.dumps({
                "schema_version": 2,
                "models": [{
                    "alias": "test",
                    "base": "llama3.2:1b",
                    "base_digest": "a" * 64,
                    "weights_digest": "b" * 64,
                    "modelfile_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "role": "test",
                    "min_ram_gb": 0,
                    "parameters_b": 1,
                }],
            }), encoding="utf-8")
            with patch("core.ollama_bootstrap.resource_path", side_effect=lambda *parts: root.joinpath(*parts)):
                specs = load_brain_manifest()
        self.assertEqual(specs[0].alias, "test")

    def test_low_memory_machine_keeps_core_and_skips_engineering_lane(self):
        selected = select_brain_models(load_brain_manifest(), snapshot(8, 4.5))
        aliases = {item.alias for item in selected}
        self.assertIn("jarvis-core-1b", aliases)
        self.assertIn("jarvis-brain-lite", aliases)
        self.assertIn("jarvis-brain-fast", aliases)
        self.assertNotIn("jarvis-brain-engineering", aliases)

    def test_balanced_machine_can_prepare_core_and_all_experts(self):
        selected = select_brain_models(load_brain_manifest(), snapshot(16, 9))
        self.assertEqual(
            {item.alias for item in selected},
            {
                "jarvis-core-1b",
                "jarvis-brain-fast",
                "jarvis-brain-engineering",
                "jarvis-brain-lite",
            },
        )

    def test_unknown_hardware_bootstraps_core_only(self):
        selected = select_brain_models(load_brain_manifest(), None)
        self.assertEqual(tuple(item.alias for item in selected), ("jarvis-core-1b",))

    def test_owner_selected_model_store_is_bound_to_private_loopback_runtime(self):
        with TemporaryDirectory() as directory:
            env = _ollama_environment("http://127.0.0.1:11435", directory)
            self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11435")
            self.assertEqual(Path(env["OLLAMA_MODELS"]), Path(directory).resolve())

    def test_literal_ipv6_loopback_is_preserved_for_ollama_cli(self):
        env = _ollama_environment("http://[::1]:11435/")
        self.assertEqual(env["OLLAMA_HOST"], "[::1]:11435")

    def test_ollama_parameter_size_parser_supports_billions_and_millions(self):
        self.assertEqual(parameter_size_b("1B"), 1.0)
        self.assertAlmostEqual(parameter_size_b("1.7B"), 1.7)
        self.assertAlmostEqual(parameter_size_b("950M"), 0.95)

    def test_existing_alias_is_trusted_only_when_parameter_size_matches_manifest(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"details": {"parameter_size": "1.0B"}}
        post = Mock(return_value=response)
        self.assertTrue(
            model_matches_manifest(
                "jarvis-core-1b",
                1.0,
                base_url="http://127.0.0.1:11435",
                request_post=post,
            )
        )
        response.json.return_value = {"details": {"parameter_size": "7B"}}
        self.assertFalse(
            model_matches_manifest(
                "jarvis-core-1b",
                1.0,
                base_url="http://127.0.0.1:11435",
                request_post=post,
            )
        )

    def test_non_loopback_ollama_endpoint_is_rejected(self):
        with self.assertRaises(OllamaBootstrapError):
            _ollama_environment("http://example.com:11434")

    def test_sub_billion_alias_cannot_pass_as_core_1b(self):
        response = Mock()
        response.json.return_value = {"details": {"parameter_size": "950M"}}
        self.assertFalse(model_matches_manifest(
            "jarvis-core-1b", 1.0, request_post=Mock(return_value=response),
        ))

    def test_inference_probe_requires_complete_nonempty_assistant_text(self):
        response = Mock()
        post = Mock(return_value=response)
        for body in (
            {"done": False, "message": {"role": "assistant", "content": "ok"}},
            {"done": True, "message": {"role": "assistant", "content": " "}},
            {"done": True, "message": {"role": "user", "content": "ok"}},
            {"done": True, "message": {"role": "assistant", "content": "ok", "tool_calls": [{}]}},
        ):
            with self.subTest(body=body):
                response.json.return_value = body
                self.assertFalse(verify_model_inference("jarvis-core-1b", request_post=post))
        response.json.return_value = {"done": True, "message": {"role": "assistant", "content": "Ready"}}
        self.assertTrue(verify_model_inference("jarvis-core-1b", request_post=post))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "jarvis-core-1b")
        self.assertFalse(payload["stream"])

    @patch("core.ollama_bootstrap._installed_model_digest")
    @patch("core.ollama_bootstrap.model_matches_pinned_spec", return_value=False)
    @patch("core.ollama_bootstrap.installed_model_names", return_value=set())
    def test_new_alias_must_pass_pinned_provenance(self, listed, checked, digest):
        core = next(spec for spec in load_brain_manifest() if spec.alias == "jarvis-core-1b")
        digest.return_value = core.base_digest
        runner = Mock(return_value=Mock(returncode=0))
        with self.assertRaisesRegex(OllamaBootstrapError, "pinned provenance"):
            provision_brain_models("ollama", runner=runner)
        self.assertEqual([call.args[0][1] for call in runner.call_args_list], ["pull", "create"])
        for call in runner.call_args_list:
            self.assertEqual(call.kwargs["encoding"], "utf-8")
            self.assertEqual(call.kwargs["errors"], "replace")
        checked.assert_called_once()

    @patch("core.ollama_bootstrap._installed_model_digest", return_value="0" * 64)
    @patch("core.ollama_bootstrap.installed_model_names", return_value=set())
    def test_changed_registry_tag_fails_before_alias_creation(self, listed, digest):
        runner = Mock(return_value=Mock(returncode=0))
        with self.assertRaisesRegex(OllamaBootstrapError, "mutable registry tag changed"):
            provision_brain_models("ollama", runner=runner)
        self.assertEqual([call.args[0][1] for call in runner.call_args_list], ["pull"])

    def test_pinned_model_verification_binds_tag_weights_prompt_and_parameters(self):
        spec = next(spec for spec in load_brain_manifest() if spec.alias == "jarvis-brain-lite")
        system, parameters = _modelfile_expectations(spec.modelfile)
        tags = Mock()
        tags.raise_for_status.return_value = None
        tags.json.return_value = {
            "models": [
                {"name": spec.base, "digest": spec.base_digest},
                {"name": spec.alias + ":latest", "digest": "f" * 64},
            ]
        }
        alias = {
            "details": {"parameter_size": "2.0B", "parent_model": spec.base},
            "capabilities": ["completion", "tools"],
            "template": "pinned-template",
            "license": "pinned-license",
            "system": system,
            "parameters": "\n".join(f"{key} {value}" for key, value in parameters.items()),
            "modelfile": "FROM C:\\\\models\\\\sha256-" + spec.weights_digest,
        }
        base = {
            "template": "pinned-template",
            "license": "pinned-license",
            "modelfile": "FROM C:\\\\models\\\\sha256-" + spec.weights_digest,
        }

        def response(value):
            result = Mock()
            result.raise_for_status.return_value = None
            result.json.return_value = value
            return result

        self.assertTrue(model_matches_pinned_spec(
            spec,
            request_get=Mock(return_value=tags),
            request_post=Mock(side_effect=[response(alias), response(base)]),
        ))

        mutations = (
            ("weights", {**alias, "modelfile": "FROM sha256-" + "0" * 64}, base),
            ("system", {**alias, "system": system + "tampered"}, base),
            ("template", {**alias, "template": "tampered"}, base),
            ("parameters", {**alias, "parameters": "temperature 9"}, base),
        )
        for name, changed_alias, changed_base in mutations:
            with self.subTest(name=name):
                self.assertFalse(model_matches_pinned_spec(
                    spec,
                    request_get=Mock(return_value=tags),
                    request_post=Mock(side_effect=[response(changed_alias), response(changed_base)]),
                ))

    @patch("core.ollama_bootstrap.verify_model_inference", return_value=False)
    @patch("core.ollama_bootstrap.provision_brain_models", return_value=(("jarvis-core-1b",), ()))
    @patch("core.ollama_bootstrap.ensure_ollama_service")
    @patch("core.ollama_bootstrap.find_ollama_executable", return_value="ollama")
    def test_setup_never_emits_ready_when_real_generation_probe_fails(self, found, service, provisioned, probe):
        progress = Mock()
        with self.assertRaisesRegex(OllamaBootstrapError, "text-generation check failed"):
            bootstrap_local_brain(progress=progress)
        self.assertNotIn("complete", [call.args[0] for call in progress.call_args_list])
        probe.assert_called_once_with("jarvis-core-1b", base_url="http://127.0.0.1:11434")

    @patch("core.ollama_bootstrap.wait_for_ollama", side_effect=[False, True, True, True])
    def test_switching_model_store_restarts_only_our_owned_server(self, wait):
        first, second = Mock(), Mock()
        first.poll.return_value = second.poll.return_value = None
        popen = Mock(side_effect=[first, second])
        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            with patch.dict(_OWNED_SERVICES, {}, clear=True):
                ensure_ollama_service("ollama", base_url="http://127.0.0.1:11435", model_store=first_dir, popen=popen)
                ensure_ollama_service("ollama", base_url="http://127.0.0.1:11435", model_store=second_dir, popen=popen)
                first.terminate.assert_called_once()
                first.wait.assert_called_once_with(timeout=5)
                second.terminate.assert_not_called()
                self.assertEqual(Path(popen.call_args.kwargs["env"]["OLLAMA_MODELS"]), Path(second_dir).resolve())

    @patch("core.ollama_bootstrap.wait_for_ollama", return_value=True)
    def test_unknown_running_server_cannot_silently_ignore_selected_model_store(self, wait):
        popen = Mock()
        with TemporaryDirectory() as directory, patch.dict(_OWNED_SERVICES, {}, clear=True):
            with self.assertRaisesRegex(OllamaBootstrapError, "unrelated Ollama service"):
                ensure_ollama_service("ollama", base_url="http://127.0.0.1:11435", model_store=directory, popen=popen)
            popen.assert_not_called()
            self.assertFalse(stop_owned_ollama_service(base_url="http://127.0.0.1:11435"))

    @patch("core.ollama_bootstrap.wait_for_ollama", return_value=True)
    def test_default_model_store_can_reuse_owners_existing_server(self, wait):
        popen = Mock()
        with patch.dict(_OWNED_SERVICES, {}, clear=True):
            ensure_ollama_service("ollama", popen=popen)
            popen.assert_not_called()

    @patch("core.ollama_bootstrap.wait_for_ollama")
    def test_endpoint_is_validated_before_any_service_probe(self, wait):
        with self.assertRaises(OllamaBootstrapError):
            ensure_ollama_service("ollama", base_url="http://example.com:11434")
        wait.assert_not_called()


class GGUFImportTests(unittest.TestCase):
    @staticmethod
    def write_gguf(directory, name="Owner model Q4.gguf"):
        path = Path(directory) / name
        # A bounded header fixture: the mocked Ollama process validates weights.
        path.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + bytes(16))
        return path

    @staticmethod
    def response(metadata=None):
        response = Mock()
        response.json.return_value = metadata if metadata is not None else {
            "details": {"parameter_size": "135M"}, "capabilities": ["completion"],
        }
        return response

    def test_selected_file_import_uses_quoted_path_safe_argv_and_active_store(self):
        commands, modelfiles = [], []

        def runner(command, **kwargs):
            commands.append((command, kwargs))
            modelfiles.append((Path(command[4]), Path(command[4]).read_text("utf-8")))
            return Mock(returncode=0)

        with TemporaryDirectory() as directory:
            source = self.write_gguf(directory)
            store = Path(directory) / "ollama-store"
            post = Mock(return_value=self.response())
            alias = import_gguf_model(
                "ollama", source, base_url="http://127.0.0.1:11435", model_store=store,
                runner=runner, request_post=post,
            )
            self.assertRegex(alias, r"^jarvis-import-owner-model-q4-[a-f0-9]{12}$")
            self.assertEqual(commands[0][0][:4], ["ollama", "create", alias, "-f"])
            self.assertEqual(commands[0][1]["env"]["OLLAMA_HOST"], "127.0.0.1:11435")
            self.assertEqual(Path(commands[0][1]["env"]["OLLAMA_MODELS"]), store.resolve())
            self.assertNotIn("shell", commands[0][1])
            self.assertIn(f'FROM "{source.as_posix()}"\n', modelfiles[0][1])
            self.assertFalse(modelfiles[0][0].exists())
            post.assert_called_once_with(
                "http://127.0.0.1:11435/api/show", json={"model": alias}, timeout=10,
            )

    def test_sub_billion_text_expert_is_allowed_and_inference_is_not_called(self):
        runner = Mock(return_value=Mock(returncode=0))
        post = Mock(return_value=self.response())
        with TemporaryDirectory() as directory, patch("core.ollama_bootstrap.verify_model_inference") as inference:
            alias = import_gguf_model("ollama", self.write_gguf(directory), alias="jarvis-import-my-expert", runner=runner, request_post=post)
        self.assertEqual(alias, "jarvis-import-my-expert")
        self.assertEqual(runner.call_args.args[0][1], "create")
        runner.assert_called_once()
        inference.assert_not_called()

    def test_numeric_parameter_metadata_is_supported(self):
        post = Mock(return_value=self.response({"model_info": {"general.parameter_count": 135_000_000}}))
        with TemporaryDirectory() as directory:
            alias = import_gguf_model("ollama", self.write_gguf(directory), runner=Mock(return_value=Mock(returncode=0)), request_post=post)
        self.assertTrue(alias.startswith("jarvis-import-"))

    def test_relative_missing_directory_and_non_gguf_inputs_never_start_ollama(self):
        runner = Mock()
        with TemporaryDirectory() as directory:
            wrong_suffix = Path(directory) / "model.bin"
            wrong_suffix.write_bytes(b"GGUF" + bytes(20))
            inputs = ("model.gguf", Path(directory) / "missing.gguf", Path(directory), wrong_suffix, 123)
            for path in inputs:
                with self.subTest(path=path), self.assertRaises(OllamaBootstrapError):
                    import_gguf_model("ollama", path, runner=runner)
        runner.assert_not_called()

    def test_fake_header_and_truncated_gguf_are_rejected(self):
        runner = Mock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            for content in (b"not GGUF" + bytes(24), b"GGUF"):
                path.write_bytes(content)
                with self.subTest(content=content), self.assertRaisesRegex(OllamaBootstrapError, "GGUF header"):
                    import_gguf_model("ollama", path, runner=runner)
        runner.assert_not_called()

    def test_symbolic_link_is_rejected(self):
        runner = Mock()
        with TemporaryDirectory() as directory:
            source = self.write_gguf(directory)
            link = Path(directory) / "linked.gguf"
            try:
                link.symlink_to(source)
            except OSError:
                self.skipTest("Symbolic links are unavailable on this platform")
            with self.assertRaisesRegex(OllamaBootstrapError, "symbolic link"):
                import_gguf_model("ollama", link, runner=runner)
        runner.assert_not_called()

    def test_import_cannot_overwrite_core_or_inject_a_modelfile_command(self):
        runner = Mock()
        with TemporaryDirectory() as directory:
            source = self.write_gguf(directory)
            for alias in ("jarvis-core-1b", "jarvis-brain-fast", "jarvis-import-x\nSYSTEM no", "../jarvis-import-x", "jarvis-import-X", 123):
                with self.subTest(alias=alias), self.assertRaises(OllamaBootstrapError):
                    import_gguf_model("ollama", source, alias=alias, runner=runner)
        runner.assert_not_called()

    def test_glob_path_is_rejected_before_ollama_can_expand_it(self):
        runner = Mock()
        with TemporaryDirectory() as directory:
            source = self.write_gguf(directory, "model[1].gguf")
            with self.assertRaisesRegex(OllamaBootstrapError, "unsupported characters"):
                import_gguf_model("ollama", source, runner=runner)
        runner.assert_not_called()

    def test_embedding_only_or_missing_parameter_metadata_does_not_report_import_success(self):
        with TemporaryDirectory() as directory:
            source = self.write_gguf(directory)
            for metadata in (
                {"details": {"parameter_size": "135M"}, "capabilities": ["embedding"]},
                {"details": {"parameter_size": "0B"}},
                {"model_info": {"general.parameter_count": True}},
                {},
            ):
                with self.subTest(metadata=metadata), self.assertRaisesRegex(OllamaBootstrapError, "could not be verified"):
                    import_gguf_model("ollama", source, runner=Mock(return_value=Mock(returncode=0)), request_post=Mock(return_value=self.response(metadata)))

    def test_failed_import_cleans_up_temporary_modelfile(self):
        modelfiles = []

        def runner(command, **kwargs):
            modelfiles.append(Path(command[4]))
            return Mock(returncode=1)

        with TemporaryDirectory() as directory:
            with self.assertRaises(OllamaBootstrapError):
                import_gguf_model("ollama", self.write_gguf(directory), runner=runner)
        self.assertFalse(modelfiles[0].exists())


if __name__ == "__main__":
    unittest.main()
