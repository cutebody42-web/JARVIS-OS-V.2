"""Hidden JARVIS Core 1B + expert council contracts."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.jarvis_council import CORE_MODEL, JarvisCouncil
from core.model_provider import ModelResponse
from core.model_router import TaskKind


def hardware(available=9.0, pressure=0.2):
    return HardwareSnapshot(
        device_id="node",
        total_ram_gb=16.0,
        available_ram_gb=available,
        cpu_percent=10,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=pressure,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, tzinfo=timezone.utc),
    )


class Profiler:
    def __init__(self, snap):
        self.snap = snap
    def capture(self):
        return self.snap


class Runtime:
    def __init__(self):
        self.ensure_calls = []
    def ensure(self, model, priority, *, keep_alive=None):
        self.ensure_calls.append((model, priority, keep_alive))


class Provider:
    def __init__(self, model, **kwargs):
        self.model = model
    def generate(self, request):
        if self.model == CORE_MODEL:
            return ModelResponse("route for owner goal", "ollama", self.model)
        return ModelResponse(f"expert note from {self.model}", "ollama", self.model)


class CouncilTests(unittest.TestCase):
    @patch("core.jarvis_council.OllamaProvider", Provider)
    def test_core_1b_coordinates_parallel_hidden_experts(self):
        runtime = Runtime()
        council = JarvisCouncil(
            ollama_base_url="http://127.0.0.1:11435",
            profiler=Profiler(hardware()),
            runtime=runtime,
        )
        result = council.consult("debug this code", TaskKind.CODING)
        self.assertIsNotNone(result)
        self.assertEqual(result.models[0], CORE_MODEL)
        self.assertIn("jarvis-brain-engineering", result.models)
        self.assertIn(CORE_MODEL, [call[0] for call in runtime.ensure_calls])
        context = council.context(result)
        self.assertIn("route for owner goal", context)
        self.assertNotIn("expert note from jarvis-core-1b", context)

    @patch("core.jarvis_council.OllamaProvider", Provider)
    def test_manual_installed_model_can_join_without_replacing_automatic_core(self):
        council = JarvisCouncil(
            ollama_base_url="http://127.0.0.1:11435",
            profiler=Profiler(hardware()),
            runtime=Runtime(),
            manual_model="my-local-model:latest",
        )
        result = council.consult("plan this", TaskKind.GENERAL)
        self.assertEqual(result.models[0], CORE_MODEL)
        self.assertIn("my-local-model:latest", result.models)

    @patch("core.jarvis_council.OllamaProvider", Provider)
    def test_high_pressure_falls_back_to_core_only(self):
        council = JarvisCouncil(
            ollama_base_url="http://127.0.0.1:11435",
            profiler=Profiler(hardware(available=2.5, pressure=0.92)),
            runtime=Runtime(),
        )
        result = council.consult("hello", TaskKind.GENERAL)
        self.assertEqual(result.models, (CORE_MODEL,))


if __name__ == "__main__":
    unittest.main()
