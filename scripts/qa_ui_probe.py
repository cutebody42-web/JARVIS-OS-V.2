#!/usr/bin/env python3
"""Render representative PyQt surfaces and collect accessibility metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QAbstractButton, QComboBox, QLineEdit

import ui


class _NoSecrets:
    def get(self, _key):
        return None


def _metrics(widget) -> dict:
    controls = []
    seen = set()
    for kind in (QAbstractButton, QComboBox, QLineEdit):
        for control in widget.findChildren(kind):
            if id(control) not in seen:
                controls.append(control)
                seen.add(id(control))
    missing_names = []
    small_controls = []
    for control in controls:
        label = control.accessibleName().strip()
        visible_text = getattr(control, "text", lambda: "")()
        placeholder = getattr(control, "placeholderText", lambda: "")()
        if not label and not str(visible_text or placeholder).strip():
            missing_names.append(control.metaObject().className())
        if control.width() < 24 or control.height() < 24:
            small_controls.append({
                "type": control.metaObject().className(),
                "width": control.width(),
                "height": control.height(),
            })
    return {
        "controls": len(controls),
        "missing_accessible_names": missing_names,
        "controls_below_24px": small_controls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    temp_settings = args.output / "ui-settings.json"
    temp_layout = args.output / "layout-settings.json"
    temp_api = args.output / "api-keys.json"

    patchers = [
        patch.object(ui, "UI_SETTINGS_FILE", temp_settings),
        patch.object(ui, "LAYOUT_SETTINGS_FILE", temp_layout),
        patch.object(ui, "API_FILE", temp_api),
        patch.object(ui, "get_secret_store", return_value=_NoSecrets()),
        patch.object(ui.MainWindow, "_setup_system_tray", lambda self: None),
        patch.object(ui.MainWindow, "_update_metrics", lambda self: None),
        patch.object(ui.MainWindow, "_restore_detached_panels", lambda self: None),
        patch.object(ui.MainWindow, "_start_layout_autosave", lambda self: None),
        patch.object(ui.MainWindow, "_start_auto_graphics_detection", lambda self: None),
    ]
    for patcher in patchers:
        patcher.start()
    try:
        temp_settings.write_text("{}", encoding="utf-8")
        window = ui.MainWindow("face.png")
        results = {}
        for width, height in ((980, 680), (1280, 820)):
            window.resize(width, height)
            window.show()
            app.processEvents()
            name = f"main-{width}x{height}"
            window.grab().save(str(args.output / f"{name}.png"))
            results[name] = _metrics(window)

        settings = ui.SettingsOverlay(current_graphics="medium")
        settings.resize(520, 400)
        settings._s_stack.setCurrentIndex(2)
        settings.show()
        app.processEvents()
        settings.grab().save(str(args.output / "settings-graphics.png"))
        results["settings-graphics"] = _metrics(settings)
        settings.hide()
        settings.deleteLater()
        window.hide()
        window.deleteLater()
        app.processEvents()
        (args.output / "ui-probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
    print(args.output / "ui-probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
