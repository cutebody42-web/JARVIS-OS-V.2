"""Offline lifecycle tests for NEXUS ModelRuntime."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock

from core.hardware_profile import GPUMemoryKind, HardwareSnapshot, PowerSource
from core.model_runtime import (
    ModelRuntime,
    ModelRuntimeError,
    ModelState,
)


GIB = 1024 ** 3


def response(body):
    result = Mock()
    result.raise_for_status.return_value = None
    result.json.return_value = body
    return result


def snapshot(pressure):
    return HardwareSnapshot(
        device_id="hp-660",
        total_ram_gb=16.0,
        available_ram_gb=max(0.0, 16.0 * (1.0 - pressure)),
        cpu_percent=pressure * 100,
        cpu_count=12,
        gpu_vram_gb=None,
        gpu_memory_kind=GPUMemoryKind.SHARED,
        power_source=PowerSource.AC,
        battery_pct=90,
        system_pressure=pressure,
        loaded_models={},
        timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )


class FakeOllama:
    def __init__(self):
        self.loaded = {}
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout):
        self.get_calls.append((url, timeout))
        return response(
            {
                "models": [
                    {
                        "name": name,
                        "model": name,
                        "size": info["size"],
                        "size_vram": info.get("size_vram", 0),
                        "expires_at": "2026-09-30T12:10:00Z",
                    }
                    for name, info in sorted(self.loaded.items())
                ]
            }
        )

    def post(self, url, json, timeout):
        self.post_calls.append((url, dict(json), timeout))
        model = json["model"]
        keep_alive = json.get("keep_alive")
        if keep_alive == 0:
            self.loaded.pop(model, None)
            reason = "unload"
        else:
            self.loaded.setdefault(
                model,
                {"size": 4 * GIB, "size_vram": 0},
            )
            reason = "load"
        return response(
            {
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": reason,
            }
        )


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama()
        self.now = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
        self.runtime = ModelRuntime(
            http_get=self.fake.get,
            http_post=self.fake.post,
            clock=lambda: self.now,
        )

    def test_ensure_loads_model_marks_it_managed_and_returns_handle(self):
        handle = self.runtime.ensure("qwen3.5:4b", priority=10)

        self.assertEqual(handle.model, "qwen3.5:4b")
        self.assertEqual(handle.state, ModelState.WARM)
        self.assertEqual(handle.priority, 10)
        self.assertEqual(handle.keep_alive, "5m")

        loaded = self.runtime.get_loaded()
        self.assertTrue(loaded["qwen3.5:4b"].managed)
        self.assertEqual(loaded["qwen3.5:4b"].priority, 10)
        self.assertEqual(loaded["qwen3.5:4b"].reported_size_gb, 4.0)

        post = self.fake.post_calls[0][1]
        self.assertEqual(post["messages"], [])
        self.assertFalse(post["stream"])
        self.assertEqual(post["keep_alive"], "5m")

    def test_release_with_ttl_refreshes_loaded_model_but_does_not_make_it_cold(self):
        self.runtime.ensure("jarvis:7b", priority=5)
        handle = self.runtime.release("jarvis:7b", ttl=300)

        self.assertEqual(handle.state, ModelState.WARM)
        self.assertEqual(handle.keep_alive, 300)
        self.assertIn("jarvis:7b", self.fake.loaded)

    def test_immediate_release_unloads_model(self):
        self.runtime.ensure("jarvis:7b", priority=5)
        handle = self.runtime.release("jarvis:7b", ttl=0)

        self.assertEqual(handle.state, ModelState.COLD)
        self.assertNotIn("jarvis:7b", self.fake.loaded)
        self.assertEqual(self.runtime.get_status().models["jarvis:7b"].state, ModelState.COLD)

    def test_release_of_cold_model_never_loads_it(self):
        before_posts = len(self.fake.post_calls)
        handle = self.runtime.release("never-loaded:1b", ttl=300)

        self.assertEqual(handle.state, ModelState.COLD)
        self.assertEqual(len(self.fake.post_calls), before_posts)

    def test_evict_lowest_priority_only_targets_nexus_managed_models(self):
        # External model is visible but must never be evicted automatically.
        self.fake.loaded["external-user-model"] = {"size": 2 * GIB, "size_vram": 0}
        self.runtime.ensure("friday:4b", priority=10)
        self.runtime.ensure("jarvis:7b", priority=5)

        victim = self.runtime.evict_lowest_priority()

        self.assertEqual(victim.model, "jarvis:7b")
        self.assertNotIn("jarvis:7b", self.fake.loaded)
        self.assertIn("friday:4b", self.fake.loaded)
        self.assertIn("external-user-model", self.fake.loaded)

    def test_protected_model_is_not_evicted(self):
        self.runtime.ensure("friday:4b", priority=10)
        self.runtime.ensure("jarvis:7b", priority=5)

        victim = self.runtime.evict_lowest_priority(protected={"jarvis:7b"})

        self.assertEqual(victim.model, "friday:4b")

    def test_emergency_pressure_evicts_one_model_and_records_warning(self):
        self.runtime.ensure("friday:4b", priority=10)
        self.runtime.ensure("jarvis:7b", priority=5)

        victim = self.runtime.relieve_pressure(snapshot(0.95), threshold=0.90)

        self.assertEqual(victim.model, "jarvis:7b")
        warnings = self.runtime.get_status().warnings
        self.assertTrue(any(item.startswith("emergency_eviction:jarvis:7b") for item in warnings))

    def test_pressure_below_threshold_does_nothing(self):
        self.runtime.ensure("friday:4b", priority=10)
        self.assertIsNone(self.runtime.relieve_pressure(snapshot(0.70)))
        self.assertIn("friday:4b", self.fake.loaded)

    def test_status_exposes_known_cold_and_external_warm_models(self):
        self.fake.loaded["external"] = {"size": GIB, "size_vram": GIB}
        self.runtime.release("known-cold", ttl=0)

        status = self.runtime.get_status()

        self.assertEqual(status.models["known-cold"].state, ModelState.COLD)
        self.assertFalse(status.models["known-cold"].managed)
        self.assertEqual(status.models["external"].state, ModelState.WARM)
        self.assertFalse(status.models["external"].managed)
        self.assertEqual(status.models["external"].vram_gb, 1.0)

    def test_status_query_failure_is_fail_closed_for_eviction(self):
        def broken_get(*_args, **_kwargs):
            raise OSError("offline")

        runtime = ModelRuntime(
            http_get=broken_get,
            http_post=self.fake.post,
            clock=lambda: self.now,
        )

        with self.assertRaises(ModelRuntimeError):
            runtime.evict_lowest_priority()

    def test_scheme_less_ollama_host_is_supported(self):
        runtime = ModelRuntime(
            ollama_base_url="127.0.0.1:1234",
            http_get=self.fake.get,
            http_post=self.fake.post,
            clock=lambda: self.now,
        )
        runtime.get_loaded()
        self.assertEqual(
            self.fake.get_calls[-1][0],
            "http://127.0.0.1:1234/api/ps",
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            ModelRuntime(timeout_seconds=0)
        with self.assertRaises(ValueError):
            self.runtime.ensure("", priority=1)
        with self.assertRaises(TypeError):
            self.runtime.ensure("model", priority=True)
        with self.assertRaises(ValueError):
            self.runtime.release("model", ttl=-1)
        with self.assertRaises(ValueError):
            self.runtime.relieve_pressure(snapshot(0.5), threshold=1.1)


if __name__ == "__main__":
    unittest.main()
