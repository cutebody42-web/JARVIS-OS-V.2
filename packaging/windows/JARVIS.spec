# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller product build for local-first JARVIS.

The generated onedir application embeds CPython and Python dependencies.
The owner does not need a separate Python installation.
"""

from PyInstaller.utils.hooks import collect_submodules
import os

PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

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

datas = [(os.path.join(PROJECT_ROOT, "models"), "models")]

for source, target in (
    ("assets", "assets"),
    ("core/prompt.txt", "core"),
):
    absolute = os.path.join(PROJECT_ROOT, source)
    if os.path.exists(absolute):
        datas.append((absolute, target))

a = Analysis(
    [os.path.join(PROJECT_ROOT, "desktop_app.py")],
    pathex=[PROJECT_ROOT],
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
