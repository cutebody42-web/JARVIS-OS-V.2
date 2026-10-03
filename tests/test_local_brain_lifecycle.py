"""Readiness and shutdown checks for the installed desktop Brain host."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from api.jarvis_local_server import LocalBrainHost, SetupState, create_local_brain_app


class LocalBrainLifecycleTests(unittest.TestCase):
    def host(self):
        host = LocalBrainHost.__new__(LocalBrainHost)
        host.ui_token = 'lifecycle-test-owner-token-' * 2
        host.ollama_base_url = 'http://127.0.0.1:11435'
        host.model_store = None
        host.setup = SetupState('setup_required')
        return host

    def test_alias_alone_does_not_make_core_ready(self):
        host = self.host()
        with patch('api.jarvis_local_server.find_ollama_executable', return_value='ollama'), patch('api.jarvis_local_server.ensure_ollama_service'), patch('api.jarvis_local_server.installed_model_names', return_value={'jarvis-core-1b'}), patch('api.jarvis_local_server.model_matches_manifest', return_value=False) as metadata:
            self.assertFalse(host.local_brain_ready())
            metadata.assert_called_once_with('jarvis-core-1b', 1.0, base_url=host.ollama_base_url)

    def test_setup_cannot_report_ready_before_generation_verification(self):
        host = self.host()
        for phase in ('starting', 'ollama', 'models', 'verification', 'failed'):
            host.setup.phase = phase
            with patch('api.jarvis_local_server.find_ollama_executable') as find:
                self.assertFalse(host.local_brain_ready())
                find.assert_not_called()

    def test_invalid_store_records_failed_state_and_remains_retryable(self):
        host = self.host()
        host._guard = threading.RLock()
        host._setup_guard = threading.Lock()
        with self.assertRaisesRegex(ValueError, 'absolute path'):
            host.setup_local_brain(approved=True, model_store='relative/path')
        self.assertEqual(host.setup.phase, 'failed')
        self.assertIn('absolute path', host.setup.error)
        self.assertTrue(host._setup_guard.acquire(blocking=False))
        host._setup_guard.release()

    def test_application_shutdown_releases_owned_host(self):
        host = self.host()
        host.close = Mock()
        with TestClient(create_local_brain_app(host)):
            pass
        host.close.assert_called_once()

    def test_stale_pairing_approval_is_a_recoverable_http_conflict(self):
        host = self.host()
        host.node = SimpleNamespace(pairing=Mock())
        host.node.pairing.approve.side_effect = PermissionError('Pairing offer is missing or expired')
        client = TestClient(create_local_brain_app(host))
        response = client.post('/v1/pair/approve',
            headers={'Authorization': 'Bearer ' + host.ui_token},
            json={'pairing_id': 'expired-offer'})
        self.assertEqual(response.status_code, 409)
        self.assertIn('expired', response.json()['detail'])

    def test_gguf_import_requires_owner_token_and_reports_import_errors(self):
        host = self.host()
        host.import_local_model = Mock(side_effect=ValueError('Choose an absolute GGUF file'))
        client = TestClient(create_local_brain_app(host))
        request = {'path': 'relative.gguf'}
        self.assertEqual(client.post('/v1/models/import', json=request).status_code, 401)
        host.import_local_model.assert_not_called()
        response = client.post('/v1/models/import',
            headers={'Authorization': 'Bearer ' + host.ui_token}, json=request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('absolute GGUF', response.json()['detail'])

    def test_gguf_import_cannot_race_model_setup(self):
        host = self.host()
        host._setup_guard = threading.Lock()
        host._setup_guard.acquire()
        with self.assertRaisesRegex(RuntimeError, 'already running'):
            host.import_local_model('/models/expert.gguf')
        self.assertTrue(host._setup_guard.locked())
        host._setup_guard.release()

    def test_shutdown_stops_voice_approval_loop_and_only_owned_ollama(self):
        host = self.host()
        host._approval_stop = threading.Event()
        host._approval_thread = Mock()
        host.voice = Mock()
        with patch('api.jarvis_local_server.stop_owned_ollama_service') as stop:
            host.close()
            stop.assert_called_once_with(base_url=host.ollama_base_url)
        self.assertTrue(host._approval_stop.is_set())
        host.voice.stop.assert_called_once()
