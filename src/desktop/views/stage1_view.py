"""Stage 1 tab: generate synthetic baseline data and train the LSTM Autoencoder."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSlider, QSpinBox, QPushButton, QProgressBar, QPlainTextEdit, QLabel,
)

from desktop.workers import TrainWorker


def _slider_with_spinbox(minimum, maximum, default, parent_layout, label):
    row = QHBoxLayout()
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(default)
    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    row.addWidget(slider, stretch=3)
    row.addWidget(spin, stretch=1)
    parent_layout.addRow(label, row)
    return spin


class Stage1View(QWidget):
    model_trained = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None

        layout = QVBoxLayout(self)

        params_box = QGroupBox("Training Parameters")
        form = QFormLayout()
        self.baseline_days_spin = _slider_with_spinbox(10, 60, 30, form, "Baseline Training Days")
        self.epochs_spin = _slider_with_spinbox(1, 30, 5, form, "LSTM Training Epochs")
        self.window_size_spin = _slider_with_spinbox(6, 60, 12, form, "LSTM Window Size (timesteps)")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        form.addRow("Random Seed", self.seed_spin)
        params_box.setLayout(form)
        layout.addWidget(params_box)

        self.train_button = QPushButton("Generate Data && Train Model")
        self.train_button.setObjectName("primaryAction")
        self.train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self.train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        log_box.setLayout(log_layout)
        layout.addWidget(log_box, stretch=1)

    def _on_train_clicked(self):
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_console.appendPlainText("Starting Stage 1 pipeline …")

        self.state.baseline_days = self.baseline_days_spin.value()
        self.state.training_epochs = self.epochs_spin.value()
        self.state.window_size = self.window_size_spin.value()
        self.state.seed = self.seed_spin.value()

        self.worker = TrainWorker(
            baseline_days=self.state.baseline_days,
            epochs=self.state.training_epochs,
            window_size=self.state.window_size,
            seed=self.state.seed,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct: int, message: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)
        self.log_console.appendPlainText(f"[{pct:3d}%] {message}")

    def _on_finished(self, payload):
        normal_df, feat_cols, detector, elapsed = payload
        self.state.normal_df = normal_df
        self.state.feat_cols = feat_cols
        self.state.detector = detector
        self.state.model_trained = True

        self.log_console.appendPlainText(
            f"Model trained in {elapsed:.1f}s | {len(normal_df):,} samples | "
            f"{len(feat_cols)} features"
        )
        self.train_button.setEnabled(True)
        self.model_trained.emit()

    def _on_failed(self, message: str):
        self.log_console.appendPlainText(f"ERROR: {message}")
        self.status_label.setText(f"Failed: {message}")
        self.train_button.setEnabled(True)
