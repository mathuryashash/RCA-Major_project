"""QThread workers wrapping pipeline.engine calls so the UI never blocks."""

from PySide6.QtCore import QThread, Signal

from pipeline import engine


class TrainWorker(QThread):
    """Stage 1: generate baseline data + train the LSTM Autoencoder."""

    progress = Signal(int, str)
    finished_ok = Signal(object)  # (normal_df, feat_cols, detector, elapsed_seconds)
    failed = Signal(str)

    def __init__(self, baseline_days: int, epochs: int, window_size: int, seed: int, parent=None):
        super().__init__(parent)
        self.baseline_days = baseline_days
        self.epochs = epochs
        self.window_size = window_size
        self.seed = seed

    def run(self):
        try:
            import time
            self.progress.emit(10, "Generating baseline data …")
            normal_df, _incident_df, _meta, feat_cols = engine.generate_data(
                seed=self.seed, baseline_days=self.baseline_days,
            )

            self.progress.emit(40, "Preprocessing …")
            normal_scaled, _incident_scaled, _scaler = engine.preprocess(
                normal_df, normal_df, feat_cols
            )

            self.progress.emit(55, f"Training LSTM ({self.epochs} epoch(s)) …")
            t0 = time.time()
            detector = engine.train_model(
                normal_scaled=normal_scaled, n_features=len(feat_cols),
                epochs=self.epochs, window_size=self.window_size,
                model_path="outputs/lstm_autoencoder_best.pt", skip_train=False,
            )
            elapsed = time.time() - t0

            self.progress.emit(100, "Model trained")
            self.finished_ok.emit((normal_df, feat_cols, detector, elapsed))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Stage 2: inject a failure scenario and run the full RCA pipeline."""

    progress = Signal(int, str)
    finished_ok = Signal(object)  # dict payload, see run()
    failed = Signal(str)

    def __init__(self, normal_df, feat_cols, detector, failure_type: str,
                 severity: float, max_granger_lag: int, seed: int, parent=None):
        super().__init__(parent)
        self.normal_df = normal_df
        self.feat_cols = feat_cols
        self.detector = detector
        self.failure_type = failure_type
        self.severity = severity
        self.max_granger_lag = max_granger_lag
        self.seed = seed

    def run(self):
        try:
            import pandas as pd

            self.progress.emit(10, "Generating incident data …")
            _normal_df, incident_df, metadata, _feat_cols = engine.generate_data(
                seed=self.seed, failure_type=self.failure_type, severity=self.severity,
            )

            self.progress.emit(25, "Preprocessing …")
            _normal_scaled, incident_scaled, _scaler = engine.preprocess(
                self.normal_df, incident_df, self.feat_cols
            )

            self.progress.emit(45, "Detecting anomalies …")
            anomaly_scores, anomaly_times, active_anomalies = engine.detect_anomalies(
                self.detector, incident_scaled, self.feat_cols,
            )

            if len(active_anomalies) == 0:
                self.failed.emit(
                    "No anomalies detected. Try increasing severity or training epochs."
                )
                return

            failure_start_idx = len(incident_df) - 200
            failure_start_time = pd.Timestamp(incident_df.iloc[failure_start_idx]["timestamp"])

            self.progress.emit(70, "Running Granger causality & ranking root causes …")
            causal_results = engine.run_causal_inference(
                incident_scaled=incident_scaled, feat_cols=self.feat_cols,
                anomaly_scores=anomaly_scores, anomaly_times=anomaly_times,
                active_anomalies=active_anomalies, failure_start_time=failure_start_time,
                max_lag=self.max_granger_lag,
            )
            root_causes = engine.rank_root_causes(causal_results)

            self.progress.emit(90, "Generating reports …")
            report = engine.generate_reports(
                results=causal_results, root_causes=root_causes,
                anomaly_times=anomaly_times, metadata=metadata,
                failure_type=self.failure_type, output_dir="outputs",
            )

            self.progress.emit(100, "Pipeline complete")
            self.finished_ok.emit({
                "causal_results": causal_results,
                "root_causes": root_causes,
                "incident_scaled": incident_scaled,
                "anomaly_scores": anomaly_scores,
                "anomaly_times": anomaly_times,
                "metadata": metadata,
                "report": report,
            })
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
