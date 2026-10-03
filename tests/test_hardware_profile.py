"""Offline tests for NEXUS hardware snapshots and OS probes."""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.hardware_profile import (
    GPUMemoryKind,
    CgroupMemoryBudget,
    HardwareProfiler,
    HardwareSnapshot,
    PowerSource,
    _probe_cgroup_memory,
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

    def test_unavailable_battery_interface_preserves_real_ram_and_cpu_snapshot(self):
        for error in (
            FileNotFoundError("/sys/class/power_supply is absent"),
            PermissionError("power telemetry denied"),
            NotImplementedError("battery telemetry unsupported"),
        ):
            with self.subTest(error=type(error).__name__):
                psutil = self._psutil(available=6 * GIB, cpu=20.0)
                psutil.sensors_battery.side_effect = error
                response = Mock()
                response.json.return_value = {"models": []}
                snapshot = HardwareProfiler(
                    device_id="headless-node",
                    psutil_module=psutil,
                    http_get=Mock(return_value=response),
                    gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN),
                ).capture()
                self.assertEqual(snapshot.total_ram_gb, 16.0)
                self.assertEqual(snapshot.available_ram_gb, 6.0)
                self.assertEqual(snapshot.cpu_percent, 20.0)
                self.assertIs(snapshot.power_source, PowerSource.UNKNOWN)
                self.assertIsNone(snapshot.battery_pct)
                self.assertIn("battery_probe_failed", snapshot.warnings)

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

    def test_container_limit_and_current_usage_clamp_routing_budget(self):
        response = Mock()
        response.json.return_value = {"models": []}
        snapshot = HardwareProfiler(
            device_id="container", psutil_module=self._psutil(total=10 * GIB, available=7 * GIB, cpu=20),
            memory_probe=lambda: CgroupMemoryBudget(8 * GIB, int(1.5 * GIB)),
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN), http_get=Mock(return_value=response),
        ).capture()
        self.assertEqual(snapshot.total_ram_gb, 8.0)
        self.assertEqual(snapshot.available_ram_gb, 1.5)
        self.assertEqual(snapshot.memory_pressure, 0.8125)
        self.assertEqual(snapshot.system_pressure, 0.8125)
        self.assertIn("cgroup_memory_limit_applied", snapshot.warnings)

    def test_host_available_memory_remains_the_upper_bound(self):
        response = Mock()
        response.json.return_value = {"models": []}
        snapshot = HardwareProfiler(
            device_id="container", psutil_module=self._psutil(total=10 * GIB, available=2 * GIB),
            memory_probe=lambda: CgroupMemoryBudget(8 * GIB, 6 * GIB),
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN), http_get=Mock(return_value=response),
        ).capture()
        self.assertEqual(snapshot.total_ram_gb, 8.0)
        self.assertEqual(snapshot.available_ram_gb, 2.0)

    def test_injected_psutil_does_not_implicitly_read_the_real_host_cgroup(self):
        response = Mock()
        response.json.return_value = {"models": []}
        with patch("core.hardware_profile._probe_cgroup_memory", side_effect=AssertionError("ambient probe")) as probe:
            snapshot = HardwareProfiler(
                device_id="fixture", psutil_module=self._psutil(),
                gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN), http_get=Mock(return_value=response),
            ).capture()
        probe.assert_not_called()
        self.assertEqual(snapshot.total_ram_gb, 16.0)
        self.assertEqual(snapshot.available_ram_gb, 8.0)

    def test_resource_probe_failure_preserves_snapshot_with_zero_allocation_headroom(self):
        response = Mock()
        response.json.return_value = {"models": []}
        snapshot = HardwareProfiler(
            device_id="container", psutil_module=self._psutil(),
            memory_probe=Mock(side_effect=PermissionError("denied")),
            gpu_probe=lambda: (None, GPUMemoryKind.UNKNOWN), http_get=Mock(return_value=response),
        ).capture()
        self.assertEqual(snapshot.available_ram_gb, 0.0)
        self.assertEqual(snapshot.system_pressure, 1.0)
        self.assertIn("cgroup_memory_probe_failed", snapshot.warnings)


class CgroupMemoryTests(unittest.TestCase):
    @staticmethod
    def probe(files, *, platform="Linux"):
        def read(path):
            value = files.get(str(path), FileNotFoundError(str(path)))
            if isinstance(value, Exception):
                raise value
            return value
        return _probe_cgroup_memory(read_text=read, platform_name=platform)

    def test_v2_clamps_to_hard_limit_minus_all_current_usage_including_cache(self):
        budget = self.probe({
            "/proc/self/cgroup": "0::/\n",
            "/sys/fs/cgroup/memory.max": str(8 * GIB),
            "/sys/fs/cgroup/memory.current": str(int(6.5 * GIB)),
            # Cache is deliberately not assumed reclaimable for allocations.
            "/sys/fs/cgroup/memory.stat": f"file {5 * GIB}\n",
        })
        self.assertEqual(budget.limit_bytes, 8 * GIB)
        self.assertEqual(budget.available_bytes, int(1.5 * GIB))
        self.assertEqual(budget.warnings, ())

    def test_v1_memory_controller_uses_current_usage(self):
        budget = self.probe({
            "/proc/self/cgroup": "4:cpu,cpuacct:/\n5:memory:/\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(4 * GIB),
            "/sys/fs/cgroup/memory/memory.usage_in_bytes": str(3 * GIB),
        })
        self.assertEqual(budget.limit_bytes, 4 * GIB)
        self.assertEqual(budget.available_bytes, GIB)

    def test_parent_sibling_usage_can_constrain_child_with_more_local_headroom(self):
        budget = self.probe({
            "/proc/self/cgroup": "0::/sessions/job\n",
            "/sys/fs/cgroup/sessions/job/memory.max": str(8 * GIB),
            "/sys/fs/cgroup/sessions/job/memory.current": str(2 * GIB),
            "/sys/fs/cgroup/sessions/memory.max": str(10 * GIB),
            "/sys/fs/cgroup/sessions/memory.current": str(7 * GIB),
            "/sys/fs/cgroup/memory.max": "max",
        })
        self.assertEqual(budget.limit_bytes, 8 * GIB)
        self.assertEqual(budget.available_bytes, 3 * GIB)

    def test_nested_v1_limit_is_not_hidden_by_unlimited_mount_root(self):
        budget = self.probe({
            "/proc/self/cgroup": "5:memory:/docker/job\n",
            "/sys/fs/cgroup/memory/docker/job/memory.limit_in_bytes": str(2 * GIB),
            "/sys/fs/cgroup/memory/docker/job/memory.usage_in_bytes": str(GIB),
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
        })
        self.assertEqual(budget.limit_bytes, 2 * GIB)
        self.assertEqual(budget.available_bytes, GIB)

    def test_unlimited_or_missing_cgroup_does_not_invent_a_limit(self):
        for files in ({}, {"/sys/fs/cgroup/memory.max": "max"},
                      {"/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712"}):
            with self.subTest(files=files):
                self.assertEqual(self.probe(files), CgroupMemoryBudget())

    def test_known_limit_with_unreadable_or_invalid_usage_has_zero_safe_headroom(self):
        for usage in (PermissionError("denied"), "-1", "bad", FileNotFoundError("gone")):
            with self.subTest(usage=usage):
                budget = self.probe({"/sys/fs/cgroup/memory.max": str(8 * GIB),
                                     "/sys/fs/cgroup/memory.current": usage})
                self.assertEqual(budget.limit_bytes, 8 * GIB)
                self.assertEqual(budget.available_bytes, 0)
                self.assertIn("cgroup_memory_usage_unavailable", budget.warnings)

    def test_unreadable_or_invalid_limit_is_not_assumed_unlimited(self):
        for value in (PermissionError("denied"), "0", "bad"):
            with self.subTest(value=value):
                budget = self.probe({"/sys/fs/cgroup/memory.max": value})
                self.assertIsNone(budget.limit_bytes)
                self.assertEqual(budget.available_bytes, 0)
                self.assertIn("cgroup_memory_limit_unavailable", budget.warnings)

    def test_usage_above_limit_never_reports_negative_memory(self):
        budget = self.probe({"/sys/fs/cgroup/memory.max": str(8 * GIB),
                             "/sys/fs/cgroup/memory.current": str(9 * GIB)})
        self.assertEqual(budget.available_bytes, 0)

    def test_non_linux_platform_never_reads_container_files(self):
        read = Mock(side_effect=AssertionError("unexpected file read"))
        self.assertEqual(_probe_cgroup_memory(read_text=read, platform_name="Windows"), CgroupMemoryBudget())
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
