"""Tests for typed NEXUS persona configuration."""

import unittest

from core.model_provider import ModelTier
from core.model_router import ModelRouter, ProviderKind, TaskKind
from core.model_runtime import RuntimeStatus
from core.hardware_profile import (
    GPUMemoryKind,
    HardwareSnapshot,
    PowerSource,
)
from core.personas import BUILTIN_PERSONAS, FRIDAY, JARVIS, TABY
from datetime import datetime, timezone


def hw():
    return HardwareSnapshot(
        device_id="node-test",
        total_ram_gb=16,
        available_ram_gb=8,
        cpu_percent=20,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=0.30,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


class PersonaSpecTests(unittest.TestCase):
    def test_builtin_registry(self):
        self.assertEqual(BUILTIN_PERSONAS.names(), ("friday", "jarvis", "taby"))
        self.assertIs(BUILTIN_PERSONAS.get("taby"), TABY)

    def test_default_and_allowed_tasks(self):
        self.assertEqual(FRIDAY.resolve_task(), TaskKind.REALTIME)
        self.assertEqual(JARVIS.resolve_task(), TaskKind.CODING)
        self.assertEqual(TABY.resolve_task(TaskKind.CODING), TaskKind.CODING)
        with self.assertRaises(ValueError):
            FRIDAY.resolve_task(TaskKind.CODING)

    def test_persona_allowlist_never_grants_unknown_capability(self):
        self.assertTrue(TABY.allows_tool("weather_report"))
        self.assertFalse(JARVIS.allows_tool("weather_report"))
        self.assertFalse(TABY.allows_tool("shell"))

    def test_same_standard_request_routes_differently_by_persona(self):
        from core.model_provider import ModelRequest

        router = ModelRouter()
        status = RuntimeStatus(models={}, warnings=())
        request = ModelRequest("help", tier=ModelTier.STANDARD)

        friday = router.route(
            request, hw(), status, FRIDAY.routing, task=FRIDAY.resolve_task()
        )
        jarvis = router.route(
            request, hw(), status, JARVIS.routing, task=JARVIS.resolve_task()
        )

        self.assertEqual(friday.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(friday.primary.model, "qwen3.5:4b")
        self.assertEqual(jarvis.primary.provider, ProviderKind.OLLAMA)
        self.assertEqual(jarvis.primary.model, "qwen2.5-coder:7b")


if __name__ == "__main__":
    unittest.main()
