# -*- mode: python ; coding: utf-8 -*-
"""
app.spec
--------
PyInstaller build spec for the GUI app (app.py), one-folder mode.

Build, on WINDOWS, from inside this folder (the parenthesized name needs
quoting — PyInstaller itself doesn't care about it, only the shell does),
in an activated venv with requirements.txt installed plus PyInstaller:

    cd "path\to\profilometry_fft_analyzer(nonlocal)"
    pip install -r requirements.txt pyinstaller
    pyinstaller app.spec

Output: dist\ProfilometryFFTAnalyzer\ProfilometryFFTAnalyzer.exe plus its
support files. Run the .exe from inside that folder — it creates
input_files\, data_files\, image_fits\, error_logs\, failed_images\,
calibration_profiles.json, and the two parameter files
(parameter_defaults.json / parameter_settings.json) next to itself on first
run (app_settings.py's get_app_dir()), so keep the whole
dist\ProfilometryFFTAnalyzer\ folder together; copying just the .exe
elsewhere won't work.

The parameter files are deliberately written next to the .exe rather than
bundled inside it, which is what lets the Preferences menu edit tuning
values in a frozen build. Don't add them to `datas` — a bundled copy would
be read-only and unwritable at runtime.

PyInstaller is not a cross-compiler: build ON Windows, FOR Windows.

One-folder (not one-file): easyOCR pulls in PyTorch, which is large — a
one-file build re-extracts a GB-scale payload to a temp dir on every
launch. One-folder starts fast and is easy to zip/copy as a unit instead.

If the built .exe errors on startup with "ModuleNotFoundError" for
something not listed below, that's PyInstaller's static import scanner
missing a dynamically-loaded submodule (common with scipy/sklearn's
compiled extensions) — add it to hiddenimports and rebuild; this is a
normal part of freezing an app with this dependency stack, not a sign the
spec is wrong. See the "Known gotchas" section in README.txt.
"""
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# The five dependencies actually known to need explicit help under
# PyInstaller with this stack: easyocr/torch are huge and data-file-heavy,
# cv2/scipy/sklearn ship compiled extensions their own hooks don't always
# fully enumerate. numpy/pandas/Pillow/matplotlib have solid upstream
# PyInstaller hooks and don't need this.
for pkg in ('easyocr', 'torch', 'cv2', 'scipy', 'sklearn'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ProfilometryFFTAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True is friendlier while you're first getting a build to
    # work: startup errors (e.g. a missing hidden import) print to a
    # console window instead of the app silently failing to appear. Once
    # it's running cleanly, flip this to False for a normal windowed app
    # (no console window) and rebuild.
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ProfilometryFFTAnalyzer',
)
