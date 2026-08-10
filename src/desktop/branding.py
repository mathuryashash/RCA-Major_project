"""Locating the application icon, from source and from a frozen build.

PyInstaller stamps the .ico onto the executable, which covers Explorer and
the taskbar, but the Qt window icon is set at runtime and needs the file
itself -- and the path differs between a source checkout and a bundle.
"""

import sys
from pathlib import Path

from PySide6.QtGui import QIcon


def icon_path() -> Path:
    """Where logo.ico lives for the way this process was started."""
    if getattr(sys, "frozen", False):
        # PyInstaller extracts bundled data beside the executable.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "assets" / "logo.ico"
    return Path(__file__).resolve().parents[2] / "assets" / "logo.ico"


def app_icon() -> QIcon:
    """The window icon, or an empty one if the file is missing.

    A missing icon is a cosmetic problem and must never stop the app opening.
    """
    path = icon_path()
    return QIcon(str(path)) if path.exists() else QIcon()
