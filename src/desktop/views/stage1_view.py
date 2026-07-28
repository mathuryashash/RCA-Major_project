"""Train a baseline model from collected local telemetry."""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSlider,
    QSpinBox, QPushButton, QProgressBar, QPlainTextEdit, QLabel,
)

from desktop.workers import TrainWorker, model_path
from pipeline import engine
from telemetry import config


def _slider_with_spinbox(minimum, maximum, default, layout, label):
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
    layout.addRow(label, row)
    return spin


class Stage1View(QWidget):
    model_trained = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        layout = QVBoxLayout(self)

        info = QLabel("Uses only clean samples from the local telemetry collector. At least 3 clean days are required.")
        info.setWordWrap(True)
        layout.addWidget(info)

        status_box = QGroupBox("Collection Status")
        status_form = QFormLayout()
        self.clean_days_label = QLabel("—")
        self.uninterrupted_label = QLabel("—")
        self.remaining_label = QLabel("—")
        self.model_label = QLabel("—")
        status_form.addRow("Clean days collected", self.clean_days_label)
        status_form.addRow("Longest uninterrupted run", self.uninterrupted_label)
        status_form.addRow("Remaining until trainable", self.remaining_label)
        status_form.addRow("Current model", self.model_label)
        status_box.setLayout(status_form)
        layout.addWidget(status_box)

        params = QGroupBox("Training Parameters")
        form = QFormLayout()
        self.epochs_spin = _slider_with_spinbox(1, 30, 5, form, "LSTM Training Epochs")
        self.window_size_spin = _slider_with_spinbox(6, 60, 12, form, "LSTM Window Size (samples)")
        params.setLayout(form)
        layout.addWidget(params)

        self.train_button = QPushButton("Train from Clean Collected Telemetry")
        self.train_button.setObjectName("primaryAction")
        self.train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self.train_button)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console, stretch=1)

        self.refresh_status()
        # ponytail: reads the samples table on the UI thread every 30s. Fine at
        # ~1M rows/year; move to a QThread if the read ever becomes visible.
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self.refresh_status)
        self._status_timer.start(30_000)

    def refresh_status(self):
        """Show real collection progress and gate training on it."""
        try:
            readiness = engine.baseline_readiness(config.db_path())
        except Exception as exc:  # noqa: BLE001 - no collector yet is a normal state
            self.clean_days_label.setText("no telemetry collected yet")
            self.uninterrupted_label.setText("—")
            self.remaining_label.setText("start the collector: python -m telemetry install")
            self.train_button.setEnabled(False)
            self.status_label.setText(str(exc))
            return

        self.clean_days_label.setText(
            f"{readiness.clean_samples:,} samples ({readiness.clean_days:.2f} days)"
        )
        self.uninterrupted_label.setText(
            f"{readiness.uninterrupted_samples:,} / {readiness.required_samples:,} samples needed"
        )
        if readiness.ready:
            self.remaining_label.setText("ready to train")
        elif readiness.hours_remaining < 24:
            self.remaining_label.setText(f"{readiness.hours_remaining:.1f} hours remaining")
        else:
            self.remaining_label.setText(f"{readiness.days_remaining:.2f} days remaining")

        status = engine.model_status(model_path())
        if not status.exists:
            self.model_label.setText(status.reason or "none")
        elif status.age_days is not None:
            self.model_label.setText(f"trained {status.age_days:.1f} days ago")
        else:
            self.model_label.setText("trained")

        # Training rejects fragmented history, so the button follows the
        # uninterrupted figure rather than the total.
        self.train_button.setEnabled(readiness.ready and self.worker is None)

    def _on_train_clicked(self):
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.state.training_epochs = self.epochs_spin.value()
        self.state.window_size = self.window_size_spin.value()
        self.worker = TrainWorker(self.state.training_epochs, self.state.window_size)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct, message):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)
        self.log_console.appendPlainText(f"[{pct:3d}%] {message}")

    def _on_finished(self, payload):
        baseline, features, detector, scaler, elapsed = payload
        self.state.normal_df = baseline
        self.state.feat_cols = features
        self.state.detector = detector
        self.state.scaler = scaler
        self.state.model_trained = True
        self.log_console.appendPlainText(f"Trained in {elapsed:.1f}s from {len(baseline):,} clean samples and {len(features)} features.")
        self.worker = None
        self.refresh_status()
        self.model_trained.emit()

    def _on_failed(self, message):
        self.status_label.setText(f"Failed: {message}")
        self.log_console.appendPlainText(f"ERROR: {message}")
        self.worker = None
        self.refresh_status()
