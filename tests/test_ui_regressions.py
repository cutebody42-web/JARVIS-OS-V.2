"""Offscreen behavior of the shipped desktop interface.

Historical tests for never-shipped tour/TTS-cache/Auto-card APIs are mapped in
`docs/nexus/qa-contract-migration.md`; they are not hardware skips or passes.
"""
import json
import inspect
import os
import tempfile
import unittest
import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import ui


class _NoSecrets:
    def get(self, _key):
        return None


_METRICS = {"cpu": 1.0, "mem": 1.0, "net": 0.0, "gpu": -1.0, "tmp": -1.0}


class UIRegressionTests(unittest.TestCase):
    def setUp(self):
        settings = ui.UI_SETTINGS_FILE.read_bytes()
        api = ui.API_FILE.read_bytes() if ui.API_FILE.exists() else None
        voice = self.window._get_selected_voice()
        self.addCleanup(ui.UI_SETTINGS_FILE.write_bytes, settings)
        self.addCleanup(lambda: ui.API_FILE.write_bytes(api) if api is not None else ui.API_FILE.unlink(missing_ok=True))
        self.addCleanup(self.window._sync_voice_combo, voice)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def _setup(self):
        overlay = ui.SetupOverlay()
        self.addCleanup(overlay.deleteLater)
        return overlay

    def test_unverified_setup_callback_cannot_unlock_runtime(self):
        overlay = self._setup()
        with patch.object(self.window, "_overlay", overlay), patch.object(self.window, "_ready", False):
            self.window._on_setup_done("unverified-key", "linux", False)
            self.assertFalse(self.window.operational_ready)
            self.assertIs(self.window._overlay, overlay)

    def test_approval_for_a_different_key_cannot_unlock_runtime(self):
        overlay = self._setup()
        overlay._verified_key = "original-key"
        with patch.object(self.window, "_overlay", overlay), patch.object(self.window, "_ready", False):
            self.window._on_setup_done("changed-key", "linux", False)
            self.assertFalse(self.window.operational_ready)

    def test_successful_setup_keeps_key_in_memory_and_preserves_config(self):
        overlay = self._setup()
        key = "test-verified-key"
        overlay._verified_key = key
        ui.API_FILE.write_text(json.dumps({"voice_name": "kore", "gemini_api_key": "legacy"}))
        with patch.object(self.window, "_overlay", overlay), patch.object(self.window, "_ready", False), \
             patch.object(self.window, "_show_voice_select_then_name"), \
             patch.object(ui, "get_secret_store") as secrets:
            self.window._on_setup_done(key, "linux", False)
            self.assertTrue(self.window.operational_ready)
            self.assertIsNone(self.window._overlay)
            self.assertEqual(os.environ["GEMINI_API_KEY"], key)
            self.assertEqual(json.loads(ui.API_FILE.read_text()), {"voice_name": "kore", "os_system": "linux"})
            secrets.assert_not_called()

    def test_saved_key_is_written_only_after_verification_and_opt_in(self):
        overlay = self._setup()
        overlay._verified_key = "test-verified-key"
        store = MagicMock()
        with patch.object(self.window, "_overlay", overlay), patch.object(self.window, "_ready", False), \
             patch.object(self.window, "_show_voice_select_then_name"), \
             patch.object(ui, "get_secret_store", return_value=store):
            self.window._on_setup_done(overlay._verified_key, "linux", True)
        store.set.assert_called_once_with("gemini_api_key", "test-verified-key")
        self.assertNotIn("test-verified-key", ui.API_FILE.read_text())

    def test_validation_freezes_key_and_deduplicates_requests(self):
        overlay = self._setup()
        overlay._verified_key = "previous"
        with patch.object(ui.threading, "Thread") as worker:
            overlay.validate_candidate("test-key")
            overlay.validate_candidate("second-key")
        self.assertFalse(overlay._key_input.isEnabled())
        self.assertFalse(overlay._init_btn.isEnabled())
        self.assertEqual(overlay._verified_key, "")
        worker.assert_called_once()

    def test_failed_validation_is_recoverable_and_never_emits_done(self):
        overlay = self._setup()
        done = MagicMock()
        overlay.done.connect(done)
        with patch.object(ui.threading, "Thread"):
            overlay.validate_candidate("bad-key")
        overlay._on_validation_finished(False, "Key rejected", "bad-key", False)
        self.assertTrue(overlay._key_input.isEnabled())
        self.assertTrue(overlay._init_btn.isEnabled())
        self.assertFalse(overlay._validation_pending)
        self.assertIn("Key rejected", overlay._validation_lbl.text())
        self.assertEqual(overlay._verified_key, "")
        done.assert_not_called()

    def test_validation_worker_exception_returns_error_without_leaking_key(self):
        overlay = self._setup()
        key = "sensitive-test-key"
        with patch.object(ui.threading, "Thread") as worker, \
             patch("core.api_key_validator.validate_gemini_api_key", side_effect=RuntimeError(key)):
            overlay.validate_candidate(key)
            worker.call_args.kwargs["target"]()
        self.app.processEvents()
        self.assertFalse(overlay._validation_pending)
        self.assertNotIn(key, overlay._validation_lbl.text())
        self.assertEqual(overlay._verified_key, "")

    def test_successful_validation_emits_exact_candidate_and_remember_choice(self):
        overlay = self._setup()
        done = MagicMock()
        overlay.done.connect(done)
        with patch.object(ui.threading, "Thread"):
            overlay.validate_candidate("verified-key", remember_key=True)
        overlay._on_validation_finished(True, "Verified", "verified-key", True)
        done.assert_called_once_with("verified-key", overlay._sel_os, True)
        self.assertEqual(overlay._verified_key, "verified-key")

    def test_invalid_saved_key_is_revoked_without_deleting_replacement(self):
        for current, removed in (("bad-key", True), ("replacement", False)):
            with self.subTest(current=current):
                overlay = self._setup()
                store = MagicMock()
                store.get.return_value = current
                with patch.object(ui.threading, "Thread"), patch.object(ui, "get_secret_store", return_value=store):
                    overlay.validate_candidate("bad-key", purge_saved_on_failure=True)
                    overlay._on_validation_finished(False, "Rejected", "bad-key", True)
                self.assertEqual(store.delete.called, removed)

    def test_failed_candidate_does_not_remove_an_unrelated_session_key(self):
        overlay = self._setup()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "different-key"}):
            overlay._on_validation_finished(False, "Rejected", "bad-key", False)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "different-key")

    def test_api_key_input_masks_secrets(self):
        self.assertEqual(self._setup()._key_input.echoMode(), ui.QLineEdit.EchoMode.Password)

    def test_setup_validation_does_not_change_remember_preference(self):
        overlay = self._setup()
        overlay._toggle_remember_key()
        with patch.object(overlay, "validate_candidate") as validate:
            overlay._key_input.setText(" candidate ")
            overlay._submit()
        validate.assert_called_once_with("candidate", remember_key=True)

    def test_startup_blocks_text_and_file_callbacks_even_when_setup_is_hidden(self):
        overlay = self._setup()
        overlay.hide()
        with patch.object(self.window, "_overlay", overlay), patch.object(self.window, "_ready", False), \
             patch.object(self.window, "on_text_command") as callback, patch.object(ui.threading, "Thread") as worker:
            self.window._send("run a task")
            self.window._on_file_selected("/does/not/exist")
            callback.assert_not_called()
            worker.assert_not_called()

    def test_operational_ready_requires_key_gate_and_dismissed_setup(self):
        facade = ui.JarvisUI.__new__(ui.JarvisUI)
        facade._win = self.window
        for ready, overlay, expected in ((False, None, False), (True, self._setup(), False), (True, None, True)):
            with self.subTest(ready=ready, expected=expected), \
                 patch.object(self.window, "_ready", ready), patch.object(self.window, "_overlay", overlay):
                self.assertEqual(facade.operational_ready, expected)
                self.assertEqual(facade.interaction_gated, not expected)

    def test_ready_text_dispatches_once_and_callback_failure_stays_in_ui(self):
        with patch.object(self.window, "_ready", True), patch.object(self.window, "_overlay", None), \
             patch.object(self.window, "on_text_command", side_effect=RuntimeError("test failure")) as callback, \
             patch.object(ui.threading, "Thread") as worker:
            self.window._send("  test command  ")
            worker.call_args.kwargs["target"]()
        callback.assert_called_once_with("test command")

    def test_known_ui_command_is_deterministic_and_does_not_call_model(self):
        with patch.object(self.window, "_ready", True), patch.object(self.window, "_overlay", None), \
             patch.object(self.window, "on_text_command") as callback, \
             patch.object(self.window, "_set_command_center") as center:
            self.window._send("open command center")
        center.assert_called_once_with(True)
        callback.assert_not_called()

    def test_unknown_ui_action_cannot_become_system_shutdown(self):
        with patch.object(self.window, "_request_quit") as quit_ui:
            self.assertFalse(self.window._handle_ui_command("shut down my computer"))
            self.assertFalse(self.window._handle_ui_command("elevate privileges"))
        quit_ui.assert_not_called()

    def test_empty_submission_never_starts_worker(self):
        with patch.object(self.window, "_ready", True), patch.object(self.window, "_overlay", None), \
             patch.object(ui.threading, "Thread") as worker:
            self.window._send("   ")
        worker.assert_not_called()

    def test_saved_voice_survives_restart_without_persisting_keys(self):
        ui.API_FILE.write_text(json.dumps({"voice_name": "puck", "gemini_api_key": "legacy", "tts_api_key": "legacy"}))
        with patch.object(self.window, "on_tts_provider_change", None):
            self.window._on_tts_select_done("gemini", "kore", "")
        saved = json.loads(ui.API_FILE.read_text())
        self.assertEqual(saved["voice_name"], "kore")
        self.assertNotIn("gemini_api_key", saved)
        self.assertNotIn("tts_api_key", saved)
        self.window._sync_voice_combo("puck")
        self.window._load_saved_voice()
        self.assertEqual(self.window._get_selected_voice(), "kore")
        self.assertEqual(os.environ["GEMINI_VOICE_NAME"], "kore")

    def test_voice_selection_notifies_engine_once(self):
        with patch.object(self.window, "on_tts_provider_change") as callback:
            self.window._on_tts_select_done("gemini", "kore", "")
        callback.assert_called_once_with("gemini", "", "kore")

    def test_external_tts_does_not_replace_gemini_voice(self):
        ui.API_FILE.write_text(json.dumps({"voice_name": "kore"}))
        with patch.object(self.window, "on_tts_provider_change", None):
            self.window._on_tts_select_done("elevenlabs", "external-voice", "")
        self.assertEqual(json.loads(ui.API_FILE.read_text())["voice_name"], "kore")

    def test_invalid_saved_voice_falls_back_to_existing_puck_default(self):
        ui.API_FILE.write_text(json.dumps({"voice_name": "missing-voice"}))
        self.window._load_saved_voice()
        self.assertEqual(self.window._get_selected_voice(), "puck")

    def test_corrupt_settings_use_safe_graphics_default(self):
        for content in ("invalid json", "[]", '{"graphics_quality":"missing"}'):
            with self.subTest(content=content):
                ui.UI_SETTINGS_FILE.write_text(content)
                self.assertEqual(ui.get_graphics_quality(), "medium")

    def test_invalid_graphics_choice_preserves_existing_settings(self):
        ui.set_graphics_quality("low")
        before = ui.UI_SETTINGS_FILE.read_bytes()
        with self.assertRaises(ValueError):
            ui.set_graphics_quality("unknown")
        self.assertEqual(ui.UI_SETTINGS_FILE.read_bytes(), before)

    def test_settings_existing_graphics_choices_emit_owner_selection(self):
        overlay = ui.SettingsOverlay(current_graphics="medium")
        self.addCleanup(overlay.deleteLater)
        self.assertEqual(set(overlay._graphics_btns), {"low", "medium", "high"})
        selected = []
        overlay.graphics_changed.connect(selected.append)
        for quality, card in overlay._graphics_btns.items():
            self.assertTrue(card.accessibleName())
            card.click()
            self.assertEqual(selected[-1], quality)

    def test_graphics_choice_updates_renderer_cadence_and_persistence(self):
        original = self.window._graphics_quality
        try:
            with patch.object(self.window, "_show_toast"):
                self.window._apply_graphics_quality_live("low")
            self.assertEqual(self.window._graphics_quality, "low")
            self.assertEqual(self.window._metric_tmr.interval(), ui.GRAPHICS_PROFILES["low"]["metrics_ms"])
            self.assertEqual(ui.get_graphics_quality(), "low")
        finally:
            with patch.object(self.window, "_show_toast"):
                self.window._apply_graphics_quality_live(original)

    def test_task_and_tool_panels_start_empty_and_bound_real_activity(self):
        tasks, logs = ui.TaskQueueWidget(), ui.ToolLogWidget()
        self.addCleanup(tasks.deleteLater)
        self.addCleanup(logs.deleteLater)
        self.assertEqual(tasks._tasks, [])
        self.assertEqual(logs._entries, [])
        for index in range(40):
            tasks._on_task(f"task-{index}", "calling")
            logs._on_entry(f"tool-{index}")
        self.assertEqual(len(tasks._tasks), 20)
        self.assertEqual(len(logs._entries), 30)
        tasks._on_task("task-39", "done")
        self.assertEqual(tasks._tasks[-1]["status"], "done")

    def test_popup_cleanup_does_not_swallow_unrelated_runtime_error(self):
        manager = ui.PopupManager(self.window._ai_core_wrap)
        manager.active_popups = [SimpleNamespace(_dismiss=MagicMock(side_effect=RuntimeError("unexpected")))]
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            manager.dismiss_all_popups()

    def test_startup_does_not_force_fullscreen(self):
        self.assertFalse(self.window.isFullScreen())

    def test_subtitle_chunks_remain_visible_until_turn_completion(self):
        subtitle = ui._SubtitleWidget()
        self.addCleanup(subtitle.deleteLater)
        subtitle.set_text("First chunk")
        subtitle.set_text("Second chunk")
        self.assertEqual(subtitle._chunks, [["First", "chunk"], ["Second", "chunk"]])
        self.assertFalse(subtitle._hold_timer.isActive())
        subtitle.start_hold_timer()
        self.assertTrue(subtitle._hold_timer.isActive())
        subtitle.clear_subtitle()
        self.assertEqual(subtitle._chunks, [])
        self.assertFalse(subtitle._hold_timer.isActive())

    def test_muted_facade_never_displays_speech_subtitles(self):
        facade = ui.JarvisUI.__new__(ui.JarvisUI)
        facade._win = self.window
        self.window._subtitle.clear_subtitle()
        with patch.object(self.window, "_muted", True):
            facade.show_subtitle("Private speech")
        self.app.processEvents()
        self.assertEqual(self.window._subtitle._chunks, [])

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(cls.temp_dir.name)

        cls.original_update_metrics = staticmethod(ui.MainWindow._update_metrics)
        cls.patchers = [
            patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_VOICE_NAME": ""}),
            patch.object(ui, "UI_SETTINGS_FILE", temp_path / "ui_settings.json"),
            patch.object(ui, "LAYOUT_SETTINGS_FILE", temp_path / "layout_settings.json"),
            patch.object(ui, "API_FILE", temp_path / "api_keys.json"),
            patch.object(ui, "get_secret_store", return_value=_NoSecrets()),
            patch.object(ui.MainWindow, "_setup_system_tray", lambda self: None),
            patch.object(ui.MainWindow, "_update_metrics", lambda self: None),
        ]
        for patcher in cls.patchers:
            patcher.start()

        ui.UI_SETTINGS_FILE.write_text(
            json.dumps({}),
            encoding="utf-8",
        )
        cls.window = ui.MainWindow("face.png")
        cls.window.show()
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.window.hide()
        cls.window.deleteLater()
        cls.app.processEvents()
        for patcher in reversed(cls.patchers):
            patcher.stop()
        cls.temp_dir.cleanup()

    def test_every_theme_has_a_complete_valid_palette(self):
        required = set(ui.ThemeManager._COLOR_KEYS)
        for name, theme in ui.ThemeManager._THEMES.items():
            self.assertFalse(required - set(theme), name)
            for key in required:
                self.assertRegex(theme[key], r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")


    def test_theme_change_updates_existing_widget_styles(self):
        ui.ThemeManager.set_theme("arc_reactor")
        ui.ThemeManager.set_theme("stealth_red")
        self.app.processEvents()
        self.assertEqual(ui.C.BG, "#080203")
        self.assertIn("#080203", self.window.centralWidget().styleSheet().lower())


    def test_graphics_setting_preserves_saved_theme(self):
        ui.UI_SETTINGS_FILE.write_text(
            json.dumps({"graphics_quality": "low", "theme": "nanotech_gold"}),
            encoding="utf-8",
        )
        ui.set_graphics_quality("high")
        saved = json.loads(ui.UI_SETTINGS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(saved, {
            "graphics_quality": "high",
            "theme": "nanotech_gold",
        })


    def test_quit_button_is_visible_and_accessible(self):
        button = self.window._quit_btn
        self.assertEqual(button.objectName(), "JarvisQuitButton")
        self.assertEqual(button.accessibleName(), "Quit JARVIS")
        self.assertEqual(button.toolTip(), "Quit JARVIS")
        self.assertFalse(button.isHidden())


    def test_dock_uses_crisp_command_rail_visual_language(self):
        rail = self.window._dock_frame
        style = rail.styleSheet().lower()
        self.assertEqual(rail.objectName(), "JarvisCommandRail")
        self.assertEqual(rail.accessibleName(), "JARVIS command rail")
        self.assertIsNone(rail.graphicsEffect())
        self.assertNotIn("qlineargradient", style)
        self.assertNotIn("border-radius: 24px", style)
        self.assertIn("border-radius: 6px", style)
        self.assertEqual(self.window._rail_control_track.objectName(), "CommandControlTrack")
        self.assertEqual(self.window._rail_control_track.accessibleName(), "Command controls")
        self.assertEqual(self.window._command_title_lbl.text(), "COMMAND RAIL")
        self.assertIn("LOCAL", self.window._rail_mode_lbl.text())

        buttons = rail.findChildren(ui.QPushButton)
        self.assertEqual(len(buttons), 8)
        for button in buttons:
            self.assertTrue(button.accessibleName().strip(), button.text())
            self.assertTrue(button.toolTip().strip(), button.text())
            self.assertNotIn("🎙", button.text())

        primary_buttons = (
            self.window._mute_btn,
            self.window._tts_btn,
            self.window._name_btn,
            self.window._theme_btn,
            self.window._quit_btn,
        )
        visible_labels = {button.text().split("·", 1)[0].strip() for button in primary_buttons}
        self.assertEqual(visible_labels, {"MIC", "VOICE", "NAME", "THEME", "QUIT"})


    def test_dock_mute_state_reuses_command_rail_style(self):
        original = self.window._muted
        try:
            self.window._muted = True
            self.window._style_mute_btn()
            self.assertIn("MUTED", self.window._mute_btn.text())
            self.assertEqual(self.window._mute_btn.accessibleName(), "Microphone muted")
            self.assertIn("border-radius: 5px", self.window._mute_btn.styleSheet())
            self.assertNotIn("border-radius: 17px", self.window._mute_btn.styleSheet())
        finally:
            self.window._muted = original
            self.window._style_mute_btn()


    def test_quit_command_uses_shared_shutdown_path(self):
        with patch.object(self.window, "_request_quit") as request_quit, \
             patch.object(ui.QTimer, "singleShot", side_effect=lambda _delay, callback: callback()):
            self.assertTrue(self.window._handle_ui_command("quit JARVIS"))
        request_quit.assert_called_once_with()


    def test_setup_overlay_is_centered_after_first_show(self):
        overlay = self.window._overlay
        self.assertIsNotNone(overlay)
        expected_x = (self.window.centralWidget().width() - overlay.width()) // 2
        expected_y = (self.window.centralWidget().height() - overlay.height()) // 2
        self.assertEqual((overlay.x(), overlay.y()), (expected_x, expected_y))


    def test_tools_panel_ignores_generic_system_lifecycle_messages(self):
        entries = self.window._mission.tool_widget._entries
        original_entries = list(entries)
        entries.clear()
        try:
            self.window._parse_log_for_context("SYS: JARVIS online.")
            self.app.processEvents()
            self.assertEqual(entries, [])

            self.window._parse_log_for_context("🔧 web_search calling")
            self.app.processEvents()
            self.assertEqual(len(entries), 1)
            self.assertIn("web_search", entries[0]["text"])
        finally:
            entries[:] = original_entries
            self.window._mission.tool_widget._rebuild()


    def test_popup_cleanup_ignores_deleted_qt_wrappers(self):
        class DeletedPopup:
            def _dismiss(self):
                raise RuntimeError(
                    "wrapped C/C++ object of type BasePopup has been deleted"
                )

        manager = ui.PopupManager(self.window._ai_core_wrap)
        manager.active_popups = [DeletedPopup()]
        manager.dismiss_all_popups()
        self.assertEqual(manager.active_popups, [])


    def test_splitter_uses_evaluated_stylesheet(self):
        sheet = self.window._splitter.styleSheet()
        self.assertNotIn('f"""', sheet)
        self.assertNotIn("{C.", sheet)
        self.assertIn("QSplitter::handle", sheet)


    def test_qss_alpha_colors_are_explicit_rgba(self):
        self.assertEqual(ui.qss_rgba("#ff2244", 0x22), "rgba(255, 34, 68, 34)")
        source = Path(ui.__file__).read_text(encoding="utf-8")
        self.assertNotRegex(source, r"\{C\.[A-Z_]+\}[0-9A-Fa-f]{2}")


    def test_unavailable_hardware_metrics_are_not_fabricated(self):
        self.window._gpu_info_cached = True
        self.window._gpu_chip = "GPU unavailable"
        self.window._gpu_vram_str = "N/A"
        with patch.object(ui, "_get_metrics", return_value=dict(_METRICS)):
            self.original_update_metrics(self.window)
        self.assertEqual(self.window._gpu_pct_lbl.text(), "N/A")
        self.assertEqual(self.window._spark_tmp._value, "N/A")
        self.assertEqual(self.window._gpu_load_bar.value(), 0)
