"""Main window — tab shell wiring Stage 1 and Stage 2 views together."""

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QLabel, QHBoxLayout, QVBoxLayout, QWidget,
)

from desktop.branding import app_icon
from desktop.state import AppState
from desktop.views.data_view import DataView
from desktop.views.stage1_view import Stage1View
from desktop.views.stage2_view import Stage2View
from version import __version__


#: Below this the three tabs cannot lay out without clipping their controls.
MINIMUM_SIZE = (1024, 640)

#: Comfortable on a large display without swallowing the whole desktop.
PREFERRED_SIZE = (1500, 950)


class MainWindow(QMainWindow):
    def _size_to_screen(self) -> None:
        """Fit the window to the display it opens on.

        A fixed 1400x900 is larger than a 1366x768 laptop panel, so the window
        opened with its lower edge -- the export buttons and the status line --
        off the bottom of the screen, and on a small display there was no way
        to reach them. Take the preferred size where it fits, most of the
        available area where it does not, and maximise when even the minimum
        would overflow.
        """
        screen = QApplication.primaryScreen()
        if screen is None:                          # offscreen or headless
            self.resize(*PREFERRED_SIZE)
            return

        available = screen.availableGeometry()      # excludes the taskbar
        self.setMinimumSize(*MINIMUM_SIZE)

        if available.width() < MINIMUM_SIZE[0] or available.height() < MINIMUM_SIZE[1]:
            self.showMaximized()
            return

        width = min(PREFERRED_SIZE[0], int(available.width() * 0.9))
        height = min(PREFERRED_SIZE[1], int(available.height() * 0.9))
        self.resize(width, height)
        # Centre on the screen the user is actually looking at, rather than
        # wherever Qt would place it on a multi-monitor desktop.
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def __init__(self):
        super().__init__()
        # The version belongs where a user reporting a problem will see it
        # without being asked to go looking for it.
        self.setWindowTitle(f"AI-Powered Root Cause Analysis  v{__version__}")
        self.setWindowIcon(app_icon())
        self._size_to_screen()

        self.state = AppState()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)

        # One compact row rather than two stacked blocks. The old header spent
        # ~80px of every tab largely restating the window title, on the most
        # valuable vertical space on the screen.
        header = QHBoxLayout()
        title = QLabel("Local Root Cause Analysis")
        title.setObjectName("heroTitle")
        subtitle = QLabel("— slowdowns, stalls and crashes on this machine")
        subtitle.setObjectName("heroSubtitle")
        header.addWidget(title)
        header.addSpacing(10)
        header.addWidget(subtitle)
        header.addStretch(1)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.stage1 = Stage1View(self.state)
        self.stage2 = Stage2View(self.state)
        self.data_view = DataView()
        self.tabs.addTab(self.data_view, "Captured Data")
        self.tabs.addTab(self.stage1, "1 — Baseline && Training")
        self.tabs.addTab(self.stage2, "2 — Run RCA Inference")
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.stage1.model_trained.connect(self._on_model_trained)

        # A model trained in an earlier session is still a usable model. Without
        # this, reopening the app left Stage 2 locked until the user retrained,
        # discarding a perfectly good artifact for no reason.
        from desktop.workers import model_path
        from pipeline import engine

        if engine.model_status(model_path()).exists:
            self.state.model_trained = True
            self.stage2.set_enabled(True)
            self.statusBar().showMessage("Existing model loaded — Stage 2 ready", 5000)

    def _on_model_trained(self):
        self.state.model_trained = True
        self.stage2.set_enabled(True)
        self.statusBar().showMessage("Model trained — Stage 2 unlocked", 5000)
