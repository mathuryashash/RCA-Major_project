"""Local override of the pyinstaller-hooks-contrib hook-torch.

The contrib hook calls collect_submodules("torch"), which imports every torch
submodule in an isolated child process. On torch 2.12 + Windows that child dies
importing torch.autograd (abort in c10/util/AbortHandler.h, exit 0xC0000409),
taking the whole build with it.

We keep what the contrib hook collects that static analysis cannot find — torch's
data files and its DLLs — and let normal import analysis pick up the modules the
app actually imports. The contrib hook's MKL branch is omitted: this torch
distribution does not require mkl, so that branch collects nothing.
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    PY_DYLIB_PATTERNS,
)

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

# PyTorch lazily loads Dynamo polyfills. Static analysis cannot see those
# imports, so the frozen application omitted ``polyfills.copy`` and failed at
# runtime. Copy Dynamo's Python sources next to Torch without declaring it as
# a hidden import: a hidden import makes PyInstaller walk Dynamo and all of
# its optional compiler backends.
datas = collect_data_files("torch._dynamo", include_py_files=True)

datas += collect_data_files(
    "torch",
    excludes=["**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib", "**/*.cpp", "**/*.pyi", "**/*.cmake"],
)
binaries = collect_dynamic_libs("torch", search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"])
