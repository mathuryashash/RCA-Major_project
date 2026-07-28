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
