"""Train a baseline model from collected local telemetry."""

import sys

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
    # addRow() is given a QLayout, not a widget, so Qt cannot buddy the label
    # to either control -- both are anonymous to a screen reader without this.
    slider.setAccessibleName(label)
    spin.setAccessibleName(label)
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

        info = QLabel(
            "Trains only on clean samples from the local telemetry collector. "
            "Windows either side of a crash, a disk fault or a collector gap are excluded, "
            "and the run must be uninterrupted — a model window cannot span a gap."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        status_box = QGroupBox("Collection Status")
        status_form = QFormLayout()
        self.clean_days_label = QLabel("—")
        self.uninterrupted_label = QLabel("—")
        self.current_run_label = QLabel("—")
        self.remaining_label = QLabel("—")
        self.model_label = QLabel("—")
        status_form.addRow("Clean samples collected", self.clean_days_label)
        status_form.addRow("Longest uninterrupted run", self.uninterrupted_label)
        status_form.addRow("Current unbroken run", self.current_run_label)
        status_form.addRow("Remaining until trainable", self.remaining_label)
        status_form.addRow("Current model", self.model_label)
        status_box.setLayout(status_form)
        layout.addWidget(status_box)

        params = QGroupBox("Training Parameters")
        form = QFormLayout()
        self.epochs_spin = _slider_with_spinbox(1, 30, 5, form, "LSTM Training Epochs")
        self.window_size_spin = _slider_with_spinbox(6, 60, 12, form, "LSTM Window Size (samples)")
        # Both settings change how long training takes, so the cost of a choice
        # should be visible before making it rather than discovered afterwards.
        self.estimate_label = QLabel("—")
        form.addRow("Estimated training time", self.estimate_label)
        # Window size changes how many training windows the history yields, so
        # re-evaluate the gate immediately rather than at the next 30s tick.
        self.window_size_spin.valueChanged.connect(self.refresh_status)
        self.epochs_spin.valueChanged.connect(self._refresh_estimate)
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

    def _refresh_estimate(self):
        """Quote the cost of the current settings.

        Calibrated by the last real run on this machine, so the first quote
        uses built-in constants and later ones use measured ones.
        """
        windows = getattr(self, "_total_windows", 0)
        if not windows:
            self.estimate_label.setText("—")
            return
        seconds = engine.estimate_training_seconds(
            windows, self.window_size_spin.value(), self.epochs_spin.value(),
        )
        self.estimate_label.setText(
            f"{engine.format_duration(seconds)}  ({windows:,} windows × "
            f"{self.epochs_spin.value()} epochs)"
        )

    def refresh_status(self):
        """Show real collection progress and gate training on it."""
        try:
            # Readiness must be judged at the window size training will use.
            # Evaluating the default while training used the slider let the
            # gate open on a configuration that then failed.
            readiness = engine.baseline_readiness(
                config.db_path(), window_size=self.window_size_spin.value()
            )
        except Exception as exc:  # noqa: BLE001 - no collector yet is a normal state
            # This is the first thing a new user sees, and it used to be the
            # worst: an instruction to run `python -m telemetry install`, which
            # someone who downloaded a packaged ZIP has no Python to run, sat
            # beside a raw sqlite error string. Two ways of saying "broken" to
            # a person whose installation is working exactly as intended.
            self.clean_days_label.setText("nothing collected yet")
            self.uninterrupted_label.setText("—")
            self.current_run_label.setText("—")
            if getattr(sys, "frozen", False):
                self.remaining_label.setText(
                    "Collection starts on its own — check back in about a day."
                )
                self.status_label.setText(
                    "Waiting for the first samples. The collector runs in the "
                    "background and needs roughly 21 hours of quiet history "
                    "before there is enough to learn from."
                )
            else:
                self.remaining_label.setText(
                    "start the collector:  python -m telemetry install"
                )
                self.status_label.setText(f"No database yet ({exc})")
            self.train_button.setEnabled(False)
            return

        self.clean_days_label.setText(
            f"{readiness.clean_samples:,} samples ({readiness.clean_days:.2f} days)"
        )
        self.uninterrupted_label.setText(
            f"{readiness.uninterrupted_samples:,} / {readiness.required_samples:,} samples needed"
        )
        # The countdown tracks this run, not the longest: an earlier, longer
        # segment is closed and can never reach the threshold.
        self.current_run_label.setText(
            f"{readiness.current_run_samples:,} / {readiness.required_samples:,} samples"
        )
        # A percentage, not just a countdown. The first day of this application
        # is otherwise a screen that says "wait" and never visibly moves, which
        # is indistinguishable from one that has stopped working.
        collected = min(readiness.current_run_samples, readiness.required_samples)
        share = 100.0 * collected / max(readiness.required_samples, 1)

        if readiness.ready:
            self.remaining_label.setText("ready to train")
            self.progress_bar.setValue(100)
        elif readiness.hours_remaining < 24:
            self.remaining_label.setText(
                f"{share:.0f}% collected — about {readiness.hours_remaining:.1f} "
                f"hours to go"
            )
            self.progress_bar.setValue(int(share))
        else:
            self.remaining_label.setText(
                f"{share:.0f}% collected — about {readiness.days_remaining:.1f} "
                f"days to go"
            )
            self.progress_bar.setValue(int(share))

        self._total_windows = readiness.total_windows
        self._refresh_estimate()

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
