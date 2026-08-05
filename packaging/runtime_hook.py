"""Runtime hook — runs before app code in the frozen executable.

Patches `inspect.findsource` to return a dummy when the real source
is unavailable (PyInstaller has no .py files), which prevents the
`OSError: could not get source code` crash inside
`torch.utils._config_module.install_config_module` at import time.

The excludes in `excludes.txt` target the same crash. Both landed in one
build, so neither has been shown to be sufficient alone. Drop this hook and
rebuild once to find out; keep it until then.
"""

import inspect

_orig_findsource = inspect.findsource


def _findsource(obj):
    try:
        return _orig_findsource(obj)
    except OSError:
        return (["_ = None\n"], 0)


inspect.findsource = _findsource
