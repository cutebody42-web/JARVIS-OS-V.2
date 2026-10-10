# -*- mode: python ; coding: utf-8 -*-
"""Single-file headless JARVIS Brain sidecar for the Tauri desktop shell."""

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
    "uvicorn",
    "fastapi",
):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

build_info = os.path.join(PROJECT_ROOT, "build-info.json")
if not os.path.isfile(build_info):
    raise RuntimeError("Run scripts/generate_build_info.py before packaging")
datas = [(os.path.join(PROJECT_ROOT, "models"), "models"), (build_info, ".")]
for source, target in (
    ("assets", "assets"),
    ("core/prompt.txt", "core"),
):
    absolute = os.path.join(PROJECT_ROOT, source)
    if os.path.exists(absolute):
        datas.append((absolute, target))

a = Analysis(
    [os.path.join(PROJECT_ROOT, "brain_sidecar.py")],
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
    a.binaries,
    a.datas,
    [],
    name="jarvis-brain",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep the inherited authentication pipe available as sys.stdin. A windowed
    # PyInstaller bootloader replaces standard streams with None on Windows.
    # tauri-plugin-shell 2.4 launches sidecars with CREATE_NO_WINDOW, so this
    # console-subsystem executable still runs without showing a console window.
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
