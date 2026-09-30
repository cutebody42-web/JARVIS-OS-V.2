"""Offline tests for NEXUS hardware snapshots and OS probes."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import (
    GPUMemoryKind,
    HardwareProfiler,
    HardwareSnapshot,
    PowerSource,
)


GIB = 1024 ** 3


class HardwareSnapshotTests(unittest.TestCase):
    def test_snapshot_can_be_constructed_directly_for_router_tests(self):
        snapshot = HardwareSnapshot(
            device_id="dell-3420",
            total_ram_gb=8.0,
            available_ram_gb=2.0,
            cpu_percent=25.0,
            cpu_count=8,
            gpu_vram_gb=None,
            gpu_memory_kind=GPUMemoryKind.SHARED,
            power_source=PowerSource.BATTERY,
            battery_pct=52,
            system_pressure=0.75,
            loaded_models={"qwen3:1.7b": 1.4},
            timestamp=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.memory_pressure, 0.75)
        self.assertEqual(snapshot.loaded_models["qwen3:1.7b"], 1.4)
        with self.assertRaises(TypeError):
            snapshot.loaded_models["other"] = 1.0

    def test_validation_rejects_impossible_values(self):
        base = dict(
            device_id="hp-660",
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            cpu_percent=20.0,
            cpu_count=12,
            gpu_vram_gb=None,
            gpu_memory_kind=GPUMemoryKind.UNKNOWN,
            power_source=PowerSource.AC,
            battery_pct=80,
            system_pressure=0.5,
            loaded_models={},
            timestamp=datetime.now(timezone.utc),
        )
        for field, bad in (
            ("available_ram_gb", 20.0),
            ("cpu_percent", 101.0),
            ("cpu_count", 0),
            ("battery_pct", 101),
            ("system_pressure", 1.1),
        ):
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                HardwareSnapshot(**{**base, field: bad})


class HardwareProfilerTests(unittest.TestCase):
    @staticmethod
    def _psutil(
        *,
        total=16 * GIB,
        available=8 * GIB,
        cpu=30.0,
        count=12,
        battery=None,
    ):
        fake = Mock()
        fake.virtual_memory.return_value = SimpleNamespace(total=total, available=available)
        fake.cpu_percent.return_value = cpu
        fake.cpu_count.return_value = count
        fake.sensors_battery.return_value = battery
        return fake

    def test_capture_uses_current_ram_cpu_power_and_loaded_models(self):
        psutil = self._psutil(
            available=4 * GIB,
            cpu=55.0,
            battery=SimpleNamespace(percent=73.2, power_plugged=True),
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "models": [
                {"name": "qwen3.5:4b", "size": 4 * GIB, "size_vram": 0},
                {"model": "qwen2.5-coder:7b", "size": 5 * GIB},
            ]
        }
        http_get = Mock(return_value=response)
        profiler = HardwareProfiler(
            device_id="hp-660",
            psutil_module=psutil,
            http_get=http_get,
            gpu_probe=lambda: (None, GPUMemoryKind.SHARED),
            clock=lambda: datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        )

        snapshot = profiler.capture()

        self.assertEqual(snapshot.device_id, "hp-660")
        self.assertEqual(snapshot.total_ram_gb, 16.0)
        self.assertEqual(snapshot.available_ram_gb, 4.0)
        self.assertEqual(snapshot.cpu_percent, 55.0)
        self.assertEqual(snapshot.power_source, PowerSource.AC)
        self.assertEqual(snapshot.battery_pct, 73)
        self.assertEqual(snapshot.gpu_memory_kind, GPUMemoryKind.SHARED)
        self.assertEqual(snapshot.loaded_models["qwen3.5:4b"], 4.0)
        self.assertEqual(snapshot.loaded_models["qwen2.5-coder:7b"], 5.0)
        self.assertAlmostEqual(snapshot.system_pressure, 0.75)
        self.assertEqual(snapshot.warnings, ())
        http_get.assert_called_once_with("http://127.0.0.1:11434/api/ps", timeout=1.5)

    def test_battery_mode_and_high_cpu_drive_pressure(self):
        psutil = self._psutil(
            available=12 * GIB,
            cpu=92.0,
            battery=SimpleNamespace(percent=40, power_plugged=False),
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": []}
        snapshot = HardwareProfiler(
            device_id="dell-3420",
            psutil_module=psutil,
            http_get=Mock(return_value=response),
            gpu_probe=lambda: (None, GPUMemoryKind.SHARED),
        ).capture()

        self.assertEqual(snapshot.power_source, PowerSource.BATTERY)
        self.assertEqual(snapshot.battery_pct, 40)
        self.assertAlmostEqual(snapshot.system_pressure, 0.92)

    def test_missing_battery_and_ollama_do_not_fail_snapshot(self):
        psutil = self._psutil(battery=None)
        http_get = Mock(side_effect=OSError("offline"))
        snapshot = HardwareProfiler(
            device_id="ci-node",
            psutil_module=psutil,
            http_get=http_get,
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
        ).capture()

        self.assertEqual(snapshot.power_source, PowerSource.UNKNOWN)
        self.assertIsNone(snapshot.battery_pct)
        self.assertEqual(snapshot.loaded_models, {})
        self.assertIn("ollama_ps_unavailable", snapshot.warnings)

    def test_gpu_probe_failure_is_nonfatal_and_visible(self):
        psutil = self._psutil()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": []}

        def fail_gpu():
            raise RuntimeError("driver")

        snapshot = HardwareProfiler(
            device_id="hp-660",
            psutil_module=psutil,
            http_get=Mock(return_value=response),
            gpu_probe=fail_gpu,
        ).capture()
        self.assertIsNone(snapshot.gpu_vram_gb)
        self.assertEqual(snapshot.gpu_memory_kind, GPUMemoryKind.UNKNOWN)
        self.assertIn("gpu_probe_failed", snapshot.warnings)

    def test_scheme_less_and_bind_ollama_hosts_are_connectable(self):
        psutil = self._psutil()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": []}

        get_one = Mock(return_value=response)
        HardwareProfiler(
            device_id="node-a",
            psutil_module=psutil,
            http_get=get_one,
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
            ollama_base_url="127.0.0.1:1234",
        ).capture()
        get_one.assert_called_once_with("http://127.0.0.1:1234/api/ps", timeout=1.5)

        get_two = Mock(return_value=response)
        HardwareProfiler(
            device_id="node-b",
            psutil_module=psutil,
            http_get=get_two,
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
            ollama_base_url="0.0.0.0:11434",
        ).capture()
        get_two.assert_called_once_with("http://127.0.0.1:11434/api/ps", timeout=1.5)

    def test_configured_device_id_wins(self):
        psutil = self._psutil()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": []}
        with patch.dict("os.environ", {"JARVIS_DEVICE_ID": "hp-660"}, clear=False):
            snapshot = HardwareProfiler(
                psutil_module=psutil,
                http_get=Mock(return_value=response),
                gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
            ).capture()
        self.assertEqual(snapshot.device_id, "hp-660")

    def test_automatic_device_id_is_hashed_not_raw_machine_identifier(self):
        psutil = self._psutil()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": []}
        with patch.dict("os.environ", {}, clear=True), patch(
            "core.hardware_profile._stable_machine_identifier",
            return_value="windows:SUPER-SECRET-MACHINE-GUID",
        ):
            snapshot = HardwareProfiler(
                psutil_module=psutil,
                http_get=Mock(return_value=response),
                gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
            ).capture()
        self.assertTrue(snapshot.device_id.startswith("node-"))
        self.assertNotIn("SUPER-SECRET", snapshot.device_id)


if __name__ == "__main__":
    unittest.main()
