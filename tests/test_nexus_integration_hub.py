import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.local_stt import LocalSTTError
from core.mcp_discovery import MCPDiscoveryError
from core.nexus.integration_hub import NEXUSIntegrationHub
from core.temporal_memory import TemporalMemoryError


def _states(hub):
    return {item.name: item for item in hub.status()}


class NEXUSIntegrationHubTests(unittest.TestCase):
    def test_hub_exposes_temporal_memory_as_builtin(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={}, platform_name="Linux")
            states = _states(hub)
            self.assertTrue(states["temporal_memory"].available)
            self.assertEqual(states["temporal_memory"].provider, "sqlite-temporal-claims")
            store = hub.temporal_memory()
            claim = store.add_claim(subject="jarvis", predicate="integration", value="temporal-memory", source="test:hub", verified=True)
            self.assertEqual(store.get(claim.claim_id).value, "temporal-memory")
            self.assertTrue(_states(hub)["temporal_memory"].active)

    def test_hub_requires_explicit_local_stt_model_path(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={}, platform_name="Linux")
            with self.assertRaises(LocalSTTError):
                hub.local_speech()

    def test_hub_reports_configured_stt_path_without_starting_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "whisper"
            model.mkdir()
            hub = NEXUSIntegrationHub(data_root=root, environment={"JARVIS_STT_MODEL_PATH": str(model)}, platform_name="Linux")
            state = _states(hub)["local_stt"]
            self.assertTrue(state.configured)
            self.assertFalse(state.active)

    def test_hub_validates_mcp_configuration_only_on_explicit_use(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={"JARVIS_MCP_DISCOVERY_JSON": "not-json"}, platform_name="Linux")
            state = _states(hub)["mcp_discovery"]
            self.assertFalse(state.configured)
            self.assertFalse(state.available)
            with self.assertRaises(MCPDiscoveryError):
                hub.mcp_discovery()

    def test_hub_builds_local_mcp_registry_without_connecting(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(
                data_root=Path(td),
                environment={"JARVIS_MCP_DISCOVERY_JSON": '[{"server_id":"local","url":"http://127.0.0.1:9000/mcp"}]'},
                platform_name="Linux",
            )
            runtime = hub.mcp_discovery()
            self.assertEqual(runtime.registry.snapshot()[0].server_id, "local")
            self.assertTrue(_states(hub)["mcp_discovery"].active)

    def test_mcp_status_requires_valid_bounded_nonempty_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = '[{"server_id":"local","url":"http://127.0.0.1:9000/mcp"}]'
            hub = NEXUSIntegrationHub(
                data_root=root,
                environment={"JARVIS_MCP_DISCOVERY_JSON": valid},
                platform_name="Linux",
            )
            with patch.object(hub, "_module_available", return_value=True):
                state = _states(hub)["mcp_discovery"]
            self.assertTrue(state.configured)
            self.assertTrue(state.available)

            for raw in ("[]", "[" + (" " * 131072) + "]"):
                with self.subTest(length=len(raw)):
                    invalid = NEXUSIntegrationHub(
                        data_root=root,
                        environment={"JARVIS_MCP_DISCOVERY_JSON": raw},
                        platform_name="Linux",
                    )
                    state = _states(invalid)["mcp_discovery"]
                    self.assertFalse(state.configured)
                    self.assertFalse(state.available)
                    with self.assertRaises(MCPDiscoveryError):
                        invalid.mcp_discovery()

    def test_injected_environment_controls_data_root_without_creating_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            injected = root / "injected"
            process = root / "process"
            with patch.dict(os.environ, {"JARVIS_DATA_ROOT": str(process)}, clear=False):
                hub = NEXUSIntegrationHub(
                    environment={"JARVIS_DATA_ROOT": str(injected)},
                    platform_name="Linux",
                )
            self.assertEqual(hub.data_root, injected)
            self.assertEqual(hub.temporal_memory_path, injected / "memory" / "temporal.db")
            self.assertFalse(injected.exists())

    def test_unsafe_writable_paths_are_rejected_before_state_creation(self):
        with self.assertRaisesRegex(TemporalMemoryError, "absolute"):
            NEXUSIntegrationHub(
                environment={"JARVIS_DATA_ROOT": "relative-state"},
                platform_name="Linux",
            )
        with tempfile.TemporaryDirectory() as td:
            source_path = Path(__file__).resolve().parents[1] / "unsafe-temporal.db"
            with self.assertRaisesRegex(TemporalMemoryError, "outside application source"):
                NEXUSIntegrationHub(
                    data_root=Path(td),
                    environment={"JARVIS_TEMPORAL_MEMORY_PATH": str(source_path)},
                    platform_name="Linux",
                )
            self.assertFalse(source_path.exists())

    def test_non_windows_uia_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            hub = NEXUSIntegrationHub(data_root=Path(td), environment={}, platform_name="Linux")
            with self.assertRaises(RuntimeError):
                hub.windows_uia()


if __name__ == "__main__":
    unittest.main()
