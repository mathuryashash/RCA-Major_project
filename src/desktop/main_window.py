"""Main window — tab shell wiring Stage 1 and Stage 2 views together."""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QLabel, QHBoxLayout, QVBoxLayout, QWidget,
)

from desktop.state import AppState
from desktop.views.data_view import DataView
from desktop.views.stage1_view import Stage1View
from desktop.views.stage2_view import Stage2View


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Root Cause Analysis")
        self.resize(1400, 900)

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
