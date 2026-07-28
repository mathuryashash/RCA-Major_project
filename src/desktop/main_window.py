"""Main window — tab shell wiring Stage 1 and Stage 2 views together."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QLabel, QVBoxLayout, QWidget

from desktop.state import AppState
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

        title = QLabel("🔍 AI-Powered Root Cause Analysis")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "Diagnose slowdowns, stalls and crashes on this machine using "
            "LSTM Autoencoders, Granger Causality, and Multi-factor Root Cause Scoring"
        )
        subtitle.setObjectName("heroSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.stage1 = Stage1View(self.state)
        self.stage2 = Stage2View(self.state)
        self.tabs.addTab(self.stage1, "1 — Baseline && Training")
        self.tabs.addTab(self.stage2, "2 — Run RCA Inference")
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.stage1.model_trained.connect(self._on_model_trained)

    def _on_model_trained(self):
        self.state.model_trained = True
        self.stage2.set_enabled(True)
        self.statusBar().showMessage("Model trained — Stage 2 unlocked", 5000)
