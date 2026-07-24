# tests/test_pipeline_engine.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import engine


def test_generate_data_returns_expected_shapes():
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=1, baseline_days=1, failure_type="cpu_spike", severity=0.9,
    )
    assert "timestamp" not in feat_cols
    assert len(feat_cols) == 10
    assert len(normal_df) > 0
    assert len(incident_df) > 0
    assert metadata["root_cause"] == "cpu_usage_percent" or "cpu" in metadata["root_cause"].lower()


def test_preprocess_scales_to_unit_range():
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=1, baseline_days=1, failure_type="memory_leak", severity=0.8,
    )
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )
    assert normal_scaled.min() >= 0.0 - 1e-6
    assert normal_scaled.max() <= 1.0 + 1e-6
    assert incident_scaled[feat_cols].min().min() >= 0.0 - 1e-6
    assert incident_scaled[feat_cols].max().max() <= 1.0 + 1e-6
    assert list(incident_scaled.columns)[0] == "timestamp"


def test_train_and_detect_roundtrip(tmp_path):
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=2, baseline_days=2, failure_type="database_slow_query", severity=1.0,
    )
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )
    # nested dir deliberately does not exist — train_model must create it
    model_path = str(tmp_path / "outputs" / "test_model.pt")
    detector = engine.train_model(
        normal_scaled=normal_scaled, n_features=len(feat_cols),
        epochs=1, window_size=6, model_path=model_path, skip_train=False,
    )
    anomaly_scores, anomaly_times, active = engine.detect_anomalies(
        detector, incident_scaled, feat_cols,
    )
    assert isinstance(anomaly_scores, dict)
    assert isinstance(active, list)
