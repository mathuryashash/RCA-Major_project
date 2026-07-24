"""Desktop app entry point."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

from PySide6.QtWidgets import QApplication

from desktop.theme import apply_theme
from desktop.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RCA Desktop")
    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
