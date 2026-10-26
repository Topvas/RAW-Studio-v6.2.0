# -*- mode: python ; coding: utf-8 -*-
# raw_studio.spec — PyInstaller-специфікація RAW Studio v6.1.0
#
# Збірка автономного виконуваного файла (у теці з raw_studio.py і raw_studio.spec):
#   Windows:  pip install numpy matplotlib pystray pyinstaller
#             pyinstaller raw_studio.spec
#   Linux:    python3 -m pip install numpy matplotlib pystray pyinstaller
#             python3 -m PyInstaller raw_studio.spec
# Результат: dist\RAW_Studio.exe (Windows) / dist/RAW_Studio (Linux, macOS).
# Один файл (onefile), без консолі (GUI), Python на цільовій машині не потрібен.
import os
import sys
_hi = ['matplotlib.backends.backend_tkagg', 'mpl_toolkits.mplot3d']
_hi += ['pystray._win32'] if sys.platform == 'win32' else ['pystray._xorg']
_ico = 'RAW_Studio.ico' if os.path.exists('RAW_Studio.ico') else None
a = Analysis(
    ['raw_studio.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=_hi,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib.tests', 'numpy.tests', 'tkinter.test'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RAW_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=_ico,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
