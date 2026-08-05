"""Background workers for training and RCA from the local telemetry store."""

from PySide6.QtCore import QThread, Signal

from pipeline import engine
from telemetry import config


def model_path():
    """Single definition of where the trained artifact lives."""
    return config.app_dir() / "telemetry_model.pt"


class TrainWorker(QThread):
    """Train only from clean samples collected on this device."""

    progress = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, epochs: int, window_size: int, parent=None):
        super().__init__(parent)
        self.epochs = epochs
        self.window_size = window_size

    def run(self):
        try:
            import time
            self.progress.emit(10, "Checking clean telemetry baseline …")
            started = time.time()
            baseline, features, detector, scaler = engine.train_from_real_telemetry(
                config.db_path(), model_path(),
                epochs=self.epochs, window_size=self.window_size,
            )
            self.progress.emit(100, "Model trained from collected telemetry")
            self.finished_ok.emit((baseline, features, detector, scaler, time.time() - started))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DetectIncidentsWorker(QThread):
    """Find incidents in collected history.

    Scores the model across up to a week of samples, so it must not run on the
    UI thread.
    """

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, lookback_hours: int = 168, parent=None):
        super().__init__(parent)
        self.lookback_hours = lookback_hours

    def run(self):
        try:
            incidents = engine.detect_incidents(
                config.db_path(), model_path(), lookback_hours=self.lookback_hours,
            )
            self.finished_ok.emit(incidents)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Run RCA over a chosen observed telemetry window."""

    progress = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, hours: int, max_granger_lag: int, start=None, end=None,
                 trigger: str = "manual", parent=None):
        super().__init__(parent)
        self.hours = hours
        self.max_granger_lag = max_granger_lag
        self.start = start
        self.end = end
        self.trigger = trigger

    def run(self):
        try:
            payload = engine.run_real_rca(
                config.db_path(), model_path(),
                hours=self.hours, max_lag=self.max_granger_lag,
                start=self.start, end=self.end, trigger=self.trigger,
                progress=self.progress.emit,
            )
            if not payload["active_anomalies"]:
                self.failed.emit("No anomalies were detected in this observed window.")
                return
            self.progress.emit(95, "Generating the report …")
            payload["causal_results"]["process_attribution"] = payload["process_attribution"]
            report = engine.generate_reports(
                payload["causal_results"], payload["root_causes"], payload["anomaly_times"], output_dir=str(config.app_dir() / "reports"),
            )
            payload["report"] = report
            self.progress.emit(100, "RCA complete")
            self.finished_ok.emit(payload)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
