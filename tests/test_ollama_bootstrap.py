"""Offline tests for bundled Ollama/JARVIS Brain provisioning."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.ollama_bootstrap import (
    BrainModelSpec,
    OllamaBootstrapError,
    _ollama_environment,
    load_brain_manifest,
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
    def test_manifest_defines_all_jarvis_brain_aliases(self):
        specs = load_brain_manifest()
        self.assertEqual(
            {spec.alias for spec in specs},
            {
                "jarvis-brain-fast",
                "jarvis-brain-engineering",
                "jarvis-brain-lite",
            },
        )
        self.assertTrue(all(spec.modelfile.is_file() for spec in specs))
        self.assertTrue(all(spec.parameters_b >= 1.0 for spec in specs))
        self.assertEqual(min(spec.parameters_b for spec in specs), 1.7)

    def test_low_memory_machine_does_not_preload_engineering_lane(self):
        selected = select_brain_models(load_brain_manifest(), snapshot(8, 4.5))
        aliases = {item.alias for item in selected}
        self.assertIn("jarvis-brain-lite", aliases)
        self.assertIn("jarvis-brain-fast", aliases)
        self.assertNotIn("jarvis-brain-engineering", aliases)

    def test_balanced_machine_can_prepare_all_lanes(self):
        selected = select_brain_models(load_brain_manifest(), snapshot(16, 9))
        aliases = {item.alias for item in selected}
        self.assertEqual(
            aliases,
            {
                "jarvis-brain-fast",
                "jarvis-brain-engineering",
                "jarvis-brain-lite",
            },
        )

    def test_unknown_hardware_bootstraps_only_lite_lane(self):
        selected = select_brain_models(load_brain_manifest(), None)
        self.assertEqual(tuple(item.alias for item in selected), ("jarvis-brain-lite",))

    def test_owner_selected_model_store_is_bound_to_private_loopback_runtime(self):
        with TemporaryDirectory() as directory:
            env = _ollama_environment("http://127.0.0.1:11435", directory)
            self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11435")
            self.assertEqual(Path(env["OLLAMA_MODELS"]), Path(directory).resolve())

    def test_non_loopback_ollama_endpoint_is_rejected(self):
        with self.assertRaises(OllamaBootstrapError):
            _ollama_environment("http://example.com:11434")


if __name__ == "__main__":
    unittest.main()
