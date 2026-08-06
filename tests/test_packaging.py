"""Packaging invariants that only fail once the app is frozen.

These cost a 20-minute build to discover the hard way, so they are pinned
here instead.
"""

import re
from pathlib import Path

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"


def _excluded_modules():
    lines = (PACKAGING / "excludes.txt").read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_optree_is_not_excluded():
    """Excluding optree breaks training in the packaged app only.

    PyInstaller bundles optree's .dist-info regardless, so torch's
    `importlib.metadata.version("optree")` probe reports the package present
    and then imports it. Excluding the module turns a supported fallback into
    "No module named 'optree'" partway through training -- and torch imports
    it lazily, so no static analysis catches this at build time.
    """
    assert "optree" not in _excluded_modules()


def test_optree_is_a_declared_hidden_import():
    """Un-excluding is not enough: nothing static imports optree."""
    spec = (PACKAGING / "rca_desktop.spec").read_text(encoding="utf-8")
    hidden = re.search(r"hiddenimports=\[(.*?)\]", spec, re.S)
    assert hidden and '"optree"' in hidden.group(1)


def test_torch_modules_torch_imports_itself_are_not_excluded():
    """torch/__init__.py imports torch.export at module scope, and
    torch._dynamo.guards imports torch._inductor at module scope. Excluding
    either breaks `import torch` outright rather than trimming the build.
    """
    excluded = _excluded_modules()
    for module in ("torch.export", "torch._export", "torch._inductor"):
        assert module not in excluded
