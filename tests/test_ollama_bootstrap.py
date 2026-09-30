"""Offline tests for bundled Ollama/JARVIS Brain provisioning."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.ollama_bootstrap import (
    OllamaBootstrapError,
    _ollama_environment,
    load_brain_manifest,
    model_matches_manifest,
    parameter_size_b,
    select_brain_models,
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


if __name__ == "__main__":
    unittest.main()
