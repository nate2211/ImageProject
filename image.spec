# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# --- CONFIGURATION ---
ENTRY_POINT = 'gui.py'
APP_NAME = 'NatesImageGenGUI'
FFMPEG_DIR = 'ffmpeg-8.0-essentials_build'
# ---------------------

# Ensure the ffmpeg directory exists
if not os.path.exists(FFMPEG_DIR):
    print(f"WARNING: FFmpeg directory '{FFMPEG_DIR}' not found.")

# Define binaries: (Source, Destination)
binaries = [
    (os.path.join(FFMPEG_DIR, 'bin', 'ffmpeg.exe'), '.'),
    (os.path.join(FFMPEG_DIR, 'bin', 'ffprobe.exe'), '.'),
]

# If your ffmpeg folder structure is flat (no bin folder), use this instead:
# binaries = [
#     (os.path.join(FFMPEG_DIR, 'ffmpeg.exe'), '.'),
#     (os.path.join(FFMPEG_DIR, 'ffprobe.exe'), '.'),
# ]

a = Analysis(
    [ENTRY_POINT],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=[
        'PIL',
        'numpy',
        'requests',
        'palettes',
        'content',
        'filters',
        'learning'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- ONEFILE BUILD CONFIGURATION ---
exe = EXE(
    pyz,
    a.scripts,
    # In OneFile mode, we include binaries/datas here instead of in COLLECT
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,

    # 1. Hide the Console
    console=False,

    # 2. Disable windowed traceback (cleaner exit if it crashes)
    disable_windowed_traceback=False,

    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Note: The COLLECT block is removed for --onefile mode