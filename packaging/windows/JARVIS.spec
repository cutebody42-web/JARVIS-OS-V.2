# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller product build for local-first JARVIS.

The generated onedir application embeds CPython and Python dependencies.
The owner does not need a separate Python installation.
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
for package in (
    "actions",
    "agent",
    "api",
    "awareness",
    "config",
    "core",
    "memory",
    "google.genai",
    "keyring.backends",
):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

datas = [("models", "models")]

import os
for source, target in (
    ("assets", "assets"),
    ("core/prompt.txt", "core"),
):
    if os.path.exists(source):
        datas.append((source, target))

a = Analysis(
    ["desktop_app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="JARVIS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="JARVIS",
)
