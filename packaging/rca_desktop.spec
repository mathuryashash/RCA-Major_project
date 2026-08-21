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
    # The .ico below stamps the executable; this copy is what Qt loads at
    # runtime for the window icon, which is a separate mechanism.
    datas=[(os.path.join(project_root, "assets", "logo.ico"), "assets")],
    hiddenimports=[
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "statsmodels.tsa.stattools",
        # torch.utils._pytree imports optree lazily, inside a function, so no
        # static analysis reaches it -- while its .dist-info ships anyway and
        # tells torch the package is there. See packaging/excludes.txt.
        "optree",
        # Belt and braces rather than a fix for a proven defect. Matplotlib
        # backends are often selected at runtime through matplotlib.use(),
        # which static analysis cannot follow; this application imports the Qt
        # canvas directly, so PyInstaller does find it, and the build manifest
        # confirms it. Stated explicitly so a future refactor that moves the
        # import behind matplotlib.use() does not silently drop the backend.
        #
        # (An earlier comment here claimed the backend was missing from the
        # bundle. That was wrong: pure-Python modules live in the PYZ archive,
        # not as loose files under _internal, so looking for them on disk finds
        # nothing regardless.)
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
    ],
    hookspath=[os.path.join(project_root, "packaging", "hooks")],
    runtime_hooks=[os.path.join(project_root, "packaging", "runtime_hook.py")],
    excludes=excludes,
    cipher=block_cipher,
)

# QtWebEngine ships a debug build of its developer-tools resources, 72MB, for
# a browser view the user cannot open devtools on. The non-debug .pak beside
# it is the one actually loaded.
a.datas = [entry for entry in a.datas if "devtools_resources.debug" not in entry[0]]

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
    icon=os.path.join(project_root, "assets", "logo.ico"),
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
