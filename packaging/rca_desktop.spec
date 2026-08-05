# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
project_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))
src_dir = os.path.join(project_root, "src")

# See excludes.txt for why this list exists and how to regenerate it.
with open(os.path.join(project_root, "packaging", "excludes.txt")) as f:
    excludes = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

a = Analysis(
    [os.path.join(src_dir, "desktop", "main.py")],
    pathex=[src_dir],
    binaries=[],
    datas=[],
    hiddenimports=[
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "statsmodels.tsa.stattools",
    ],
    hookspath=[os.path.join(project_root, "packaging", "hooks")],
    runtime_hooks=[os.path.join(project_root, "packaging", "runtime_hook.py")],
    excludes=excludes,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RCA-Desktop",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="RCA-Desktop",
)
