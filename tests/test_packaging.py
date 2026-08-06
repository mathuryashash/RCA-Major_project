"""Packaging invariants that only fail once the app is frozen.

These cost a 20-minute build to discover the hard way, so they are pinned
here instead.
"""

import ast
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


def _hidden_imports():
    """Read the spec as source, not as text.

    Matching the list with a regex captured the comment inside it, so
    commenting the entry out left this test passing while the packaged app
    was broken again. Comments do not survive into an AST.
    """
    tree = ast.parse((PACKAGING / "rca_desktop.spec").read_text(encoding="utf-8"))
    return [
        ast.literal_eval(element)
        for node in ast.walk(tree) if isinstance(node, ast.Call)
        for keyword in node.keywords if keyword.arg == "hiddenimports"
        for element in keyword.value.elts
    ]


def test_optree_is_a_declared_hidden_import():
    """Un-excluding is not enough: nothing static imports optree."""
    assert "optree" in _hidden_imports()


def test_torch_modules_torch_imports_itself_are_not_excluded():
    """torch/__init__.py imports torch.export at module scope, and
    torch._dynamo.guards imports torch._inductor at module scope. Excluding
    either breaks `import torch` outright rather than trimming the build.
    """
    excluded = _excluded_modules()
    for module in ("torch.export", "torch._export", "torch._inductor"):
        # Excluding a parent excludes everything under it, and a regenerated
        # list is far likelier to gain a bare "torch" than the dotted name --
        # its torchvision/torchaudio neighbours are already there.
        parents = {module.rsplit(".", 1)[0], module}
        assert not (parents & excluded), f"{parents & excluded} would break `import torch`"
