import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_inference.dynamic_graph import DynamicGraphGenerator
from models.lstm_autoencoder import AnomalyDetector
from pipeline import engine


def _telemetry_frame(rows=80):
    timestamps = pd.date_range("2026-07-01", periods=rows, freq="30s", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "cpu_pct": np.linspace(10, 30, rows),
        "mem_pct": np.linspace(40, 50, rows),
        "disk_busy_pct": np.linspace(1, 5, rows),
    })


def test_preprocess_scales_observed_values_to_unit_range():
    baseline = _telemetry_frame()
    incident = _telemetry_frame()
    features = ["cpu_pct", "mem_pct", "disk_busy_pct"]
    normal_scaled, incident_scaled, _scaler = engine.preprocess(baseline, incident, features)
    assert normal_scaled.min() >= 0.0
    assert normal_scaled.max() <= 1.0
    assert list(incident_scaled.columns)[0] == "timestamp"


def test_model_artifact_preserves_feature_order_and_thresholds(tmp_path):
    features = ["cpu_pct", "mem_pct", "disk_busy_pct"]
    scaler = MinMaxScaler().fit(_telemetry_frame()[features])
    detector = AnomalyDetector(n_features=len(features), window_size=6)
    detector.threshold_per_metric = np.array([0.1, 0.2, 0.3])
    path = tmp_path / "telemetry_model.pt"
    engine.save_model_artifact(detector, scaler, features, path)
    loaded, loaded_scaler, loaded_features = engine.load_model_artifact(path)
    assert loaded_features == features
    assert loaded.window_size == 6
    assert np.allclose(loaded.threshold_per_metric, detector.threshold_per_metric)
    assert loaded_scaler.n_features_in_ == len(features)


def test_static_topology_prunes_impossible_edges():
    graph = DynamicGraphGenerator()
    assert graph.is_path_possible("cpu_pct", "disk_busy_pct")
    assert not graph.is_path_possible("disk_busy_pct", "cpu_pct")


def test_training_reports_every_epoch(tmp_path):
    """Training is the longest thing the app does and every epoch looks alike.

    Without a per-epoch callback the progress bar sat at one value for the
    whole fit, which reads as a hang rather than as work.
    """
    import numpy as np

    from models.lstm_autoencoder import AnomalyDetector

    detector = AnomalyDetector(n_features=2, window_size=5)
    seen = []
    detector.train(
        np.random.rand(200, 2).astype(np.float32),
        epochs=3,
        checkpoint_path=tmp_path / "model.pt",
        on_epoch=lambda done, total, train_loss, val_loss: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_training_estimate_scales_with_the_settings_that_drive_it():
    """The quote must move when the inputs that cost time move."""
    from pipeline import engine

    base = engine.estimate_training_seconds(1700, 12, 5, cold_start=False)
    assert engine.estimate_training_seconds(1700, 12, 20, cold_start=False) > base
    assert engine.estimate_training_seconds(1700, 60, 5, cold_start=False) > base
    assert engine.estimate_training_seconds(5000, 12, 5, cold_start=False) > base
    # Torch pulls in Dynamo on first use, which is not free.
    assert engine.estimate_training_seconds(1700, 12, 5, cold_start=True) > base


def test_rca_estimate_grows_faster_than_the_window():
    """Granger tests every pair, and the pair count grows with the window."""
    from pipeline import engine

    small = engine.estimate_rca_seconds(100) - engine.estimate_rca_seconds(0)
    large = engine.estimate_rca_seconds(2000) - engine.estimate_rca_seconds(1900)
    assert large > small * 5


def test_durations_read_naturally():
    from pipeline import engine

    assert engine.format_duration(1) == "~1 second"
    assert engine.format_duration(45) == "~45 seconds"
    assert "minute" in engine.format_duration(600)
    assert "hour" in engine.format_duration(9000)
