"""Background workers for training and RCA from the local telemetry store."""

from PySide6.QtCore import QThread, Signal

from pipeline import engine
from telemetry import config


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
                config.db_path(), config.app_dir() / "telemetry_model.pt",
                epochs=self.epochs, window_size=self.window_size,
            )
            self.progress.emit(100, "Model trained from collected telemetry")
            self.finished_ok.emit((baseline, features, detector, scaler, time.time() - started))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Run RCA over the latest observed telemetry window."""

    progress = Signal(int, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, hours: int, max_granger_lag: int, parent=None):
        super().__init__(parent)
        self.hours = hours
        self.max_granger_lag = max_granger_lag

    def run(self):
        try:
            self.progress.emit(10, "Loading recent collected telemetry …")
            payload = engine.run_real_rca(
                config.db_path(), config.app_dir() / "telemetry_model.pt",
                hours=self.hours, max_lag=self.max_granger_lag,
            )
            if not payload["active_anomalies"]:
                self.failed.emit("No anomalies were detected in this observed window.")
                return
            self.progress.emit(75, "Building constrained causal graph …")
            payload["causal_results"]["process_attribution"] = payload["process_attribution"]
            report = engine.generate_reports(
                payload["causal_results"], payload["root_causes"], payload["anomaly_times"],
                metadata=None, output_dir=str(config.app_dir() / "reports"),
            )
            payload["report"] = report
            self.progress.emit(100, "RCA complete")
            self.finished_ok.emit(payload)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
