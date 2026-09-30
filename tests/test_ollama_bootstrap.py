"""Offline tests for bundled Ollama/JARVIS Brain provisioning."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.ollama_bootstrap import (
    BrainModelSpec,
    OllamaBootstrapError,
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


if __name__ == "__main__":
    unittest.main()
