"""Offline tests for bundled Ollama/JARVIS Brain provisioning."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.ollama_bootstrap import (
    OllamaBootstrapError,
    _OWNED_SERVICES,
    _ollama_environment,
    bootstrap_local_brain,
    ensure_ollama_service,
    load_brain_manifest,
    model_matches_manifest,
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

    @patch("core.ollama_bootstrap.model_matches_manifest", return_value=False)
    @patch("core.ollama_bootstrap.installed_model_names", return_value=set())
    def test_new_alias_must_pass_metadata_verification(self, listed, checked):
        runner = Mock(return_value=Mock(returncode=0))
        with self.assertRaisesRegex(OllamaBootstrapError, "parameter size"):
            provision_brain_models("ollama", runner=runner)
        self.assertEqual([call.args[0][1] for call in runner.call_args_list], ["pull", "create"])
        checked.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
