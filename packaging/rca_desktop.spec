# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
project_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))
src_dir = os.path.join(project_root, "src")

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
    runtime_hooks=[],
    # The app's real import closure is 172 top-level modules; none of these are in
    # it. They are installed in this global (non-venv) Python and were observed
    # being walked by a previous Analysis run, costing ~50 minutes.
    excludes=[
        "torchvision", "torchaudio", "torchao", "pytorch_lightning", "geopandas",
        "causallearn", "matplotlib", "pydot", "streamlit",
        "googleapiclient", "psycopg2", "psycopg", "lightgbm", "mako", "ormsgpack",
        "boto3", "botocore", "prometheus_api_client",
        "IPython", "notebook", "jupyter",
    ],
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
