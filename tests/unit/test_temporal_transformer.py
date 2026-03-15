"""
Unit tests for the Temporal Transformer anomaly detector.

Covers: PositionalEncoding, TemporalTransformerModel, and
TemporalTransformerDetector (pair creation, training, scoring).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.temporal_transformer import (
    PositionalEncoding,
    TemporalTransformerDetector,
    TemporalTransformerModel,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_healthy_df(
    n_rows: int = 200, n_features: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Create a smooth, low-noise DataFrame that a model can learn easily."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, n_rows)
    cols = {}
    for i in range(n_features):
        cols[f"metric_{i}"] = np.sin(t + i) + rng.normal(0, 0.05, n_rows)
    return pd.DataFrame(cols)


def _make_anomalous_df(
    healthy_df: pd.DataFrame, spike_col: int = 0, spike_magnitude: float = 10.0
) -> pd.DataFrame:
    """Return a copy with a large spike injected into the last forecast_horizon rows."""
    df = healthy_df.copy()
    col = df.columns[spike_col]
    df.iloc[-10:, spike_col] = df.iloc[-10:, spike_col] + spike_magnitude
    return df


# -----------------------------------------------------------------------
# 1. PositionalEncoding shape
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestPositionalEncoding:
    def test_positional_encoding_shape(self):
        d_model = 128
        seq_len = 60
        batch = 4
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)
        x = torch.randn(batch, seq_len, d_model)
        out = pe(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_positional_encoding_adds_signal(self):
        """Output should differ from input (positional signal is non-zero)."""
        d_model = 64
        pe = PositionalEncoding(d_model=d_model, dropout=0.0)
        x = torch.zeros(1, 10, d_model)
        out = pe(x)
        assert not torch.allclose(out, x), "PE should add a non-zero signal"


# -----------------------------------------------------------------------
# 2. Transformer model forward pass
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestTemporalTransformerModel:
    def test_transformer_model_forward(self):
        batch, seq_len, n_features = 8, 60, 5
        horizon = 10
        model = TemporalTransformerModel(
            n_features=n_features,
            d_model=128,
            n_heads=4,
            n_layers=2,
            forecast_horizon=horizon,
        )
        x = torch.randn(batch, seq_len, n_features)
        out = model(x)
        assert out.shape == (batch, horizon, n_features)

    def test_transformer_model_single_feature(self):
        """Ensure it works when there is only one metric."""
        model = TemporalTransformerModel(n_features=1, d_model=64, n_heads=4)
        x = torch.randn(2, 60, 1)
        out = model(x)
        assert out.shape == (2, 10, 1)


# -----------------------------------------------------------------------
# 3. Detector — create_forecast_pairs
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorCreatePairs:
    def test_detector_create_pairs(self):
        n_features = 3
        seq_len = 60
        horizon = 10
        det = TemporalTransformerDetector(
            n_features=n_features,
            sequence_length=seq_len,
            forecast_horizon=horizon,
        )
        length = 100
        data = np.random.randn(length, n_features).astype(np.float32)
        inputs, targets = det.create_forecast_pairs(data)

        expected_n = length - seq_len - horizon + 1
        assert inputs.shape == (expected_n, seq_len, n_features)
        assert targets.shape == (expected_n, horizon, n_features)

        # Verify continuity: target of first pair follows its input
        np.testing.assert_array_equal(inputs[0], data[:seq_len])
        np.testing.assert_array_equal(targets[0], data[seq_len : seq_len + horizon])

    def test_create_pairs_insufficient_data(self):
        det = TemporalTransformerDetector(
            n_features=2, sequence_length=60, forecast_horizon=10
        )
        short_data = np.random.randn(50, 2)
        with pytest.raises(ValueError, match="Data length"):
            det.create_forecast_pairs(short_data)


# -----------------------------------------------------------------------
# 4. Detector — training
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorTrain:
    def test_detector_train(self):
        """Training should complete and loss should decrease over epochs."""
        df = _make_healthy_df(n_rows=200, n_features=2)
        det = TemporalTransformerDetector(
            n_features=2, d_model=32, n_heads=2, n_layers=1
        )
        stats = det.train(df, epochs=10, batch_size=16, learning_rate=1e-3)

        assert "epoch_losses" in stats
        assert len(stats["epoch_losses"]) == 10
        assert stats["epoch_losses"][-1] < stats["epoch_losses"][0], (
            "Loss should decrease over training"
        )
        assert det.is_trained


# -----------------------------------------------------------------------
# 5. Detector — score before training
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorScoreUntrained:
    def test_detector_score_untrained(self):
        det = TemporalTransformerDetector(n_features=2)
        df = _make_healthy_df(n_rows=100, n_features=2)
        with pytest.raises(ValueError, match="trained"):
            det.score(df)


# -----------------------------------------------------------------------
# 6. Detector — normal data scores low
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorScoreNormal:
    def test_detector_score_normal(self):
        df = _make_healthy_df(n_rows=300, n_features=2, seed=42)
        det = TemporalTransformerDetector(
            n_features=2, d_model=32, n_heads=2, n_layers=1
        )
        det.train(df.iloc[:200], epochs=20, batch_size=16)

        # Score a fresh healthy window (unseen but from same distribution)
        scores = det.score(df.iloc[200:])
        assert scores.shape == (2,)
        # All scores should be below 0.7 for normal data
        assert np.all(scores < 0.7), f"Normal scores too high: {scores}"


# -----------------------------------------------------------------------
# 7. Detector — anomalous data scores higher
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorScoreAnomalous:
    def test_detector_score_anomalous(self):
        df_healthy = _make_healthy_df(n_rows=300, n_features=2, seed=42)
        det = TemporalTransformerDetector(
            n_features=2, d_model=32, n_heads=2, n_layers=1
        )
        det.train(df_healthy.iloc[:200], epochs=20, batch_size=16)

        # Normal window
        normal_scores = det.score(df_healthy.iloc[200:])

        # Anomalous window — large spike in metric_0
        df_anom = _make_anomalous_df(
            df_healthy.iloc[200:], spike_col=0, spike_magnitude=10.0
        )
        anom_scores = det.score(df_anom)

        assert anom_scores.shape == (2,)
        # Anomalous metric should score higher than its normal counterpart
        assert anom_scores[0] > normal_scores[0], (
            f"Anomalous score ({anom_scores[0]:.3f}) should exceed "
            f"normal score ({normal_scores[0]:.3f}) for spiked metric"
        )


# -----------------------------------------------------------------------
# 8. Detector — single feature
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestDetectorSingleFeature:
    def test_detector_single_feature(self):
        """The full pipeline should work with just 1 metric."""
        rng = np.random.RandomState(99)
        t = np.linspace(0, 6 * np.pi, 250)
        df = pd.DataFrame({"cpu": np.sin(t) + rng.normal(0, 0.05, 250)})

        det = TemporalTransformerDetector(
            n_features=1,
            d_model=32,
            n_heads=2,
            n_layers=1,
        )
        stats = det.train(df.iloc[:180], epochs=15, batch_size=16)
        assert det.is_trained
        assert stats["final_loss"] < stats["epoch_losses"][0]

        scores = det.score(df.iloc[180:])
        assert scores.shape == (1,)
        assert 0.0 <= scores[0] <= 1.0
