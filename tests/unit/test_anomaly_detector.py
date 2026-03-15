"""
Unit tests for src.models.anomaly_detector
Tests SimpleLogAnomalyDetector and MetricAnomalyDetector.
"""

import numpy as np
import pytest

from src.models.anomaly_detector import MetricAnomalyDetector, SimpleLogAnomalyDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NORMAL_LOGS = [
    "Request handled: GET /api/health 200 5ms",
    "Request handled: POST /api/users 201 12ms",
    "Database query completed in 23ms",
    "Cache hit for key: user_session_123",
    "Background job completed: email_dispatch",
] * 20  # Repeat to build corpus


@pytest.fixture
def trained_log_detector():
    """Return a SimpleLogAnomalyDetector that has already been trained."""
    detector = SimpleLogAnomalyDetector()
    detector.train(NORMAL_LOGS)
    return detector


@pytest.fixture
def metric_detector():
    return MetricAnomalyDetector(z_threshold=3)


# ===================================================================
# SimpleLogAnomalyDetector
# ===================================================================


class TestSimpleLogAnomalyDetector:
    @pytest.mark.unit
    def test_untrained_raises_or_returns_empty(self):
        """score() before train() should handle gracefully (return zeros)."""
        detector = SimpleLogAnomalyDetector()
        messages = ["some log message", "another log message"]
        result = detector.score(messages)
        assert isinstance(result, np.ndarray)
        assert np.all(result == 0)

    @pytest.mark.unit
    def test_train_sets_trained_flag(self, trained_log_detector):
        """After train(), is_trained should be True."""
        assert trained_log_detector.is_trained is True

    @pytest.mark.unit
    def test_train_sets_threshold(self, trained_log_detector):
        """threshold should be a positive float after training."""
        assert isinstance(trained_log_detector.threshold, (float, np.floating))
        assert trained_log_detector.threshold > 0

    @pytest.mark.unit
    def test_score_returns_ndarray(self, trained_log_detector):
        """score() returns a numpy ndarray."""
        result = trained_log_detector.score(["test message"])
        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_score_length_matches_input(self, trained_log_detector):
        """score([msg1, msg2, msg3]) returns array of length 3."""
        messages = ["message one", "message two", "message three"]
        result = trained_log_detector.score(messages)
        assert len(result) == 3

    @pytest.mark.unit
    def test_normal_logs_score_low(self, trained_log_detector):
        """Train on normal logs, score same logs — scores should be <= threshold."""
        scores = trained_log_detector.score(NORMAL_LOGS)
        # All training logs should score at or below the 99th-percentile threshold
        fraction_at_or_below = np.mean(scores <= trained_log_detector.threshold)
        assert fraction_at_or_below >= 0.99, (
            f"Expected >=99% of normal logs at or below threshold, "
            f"got {fraction_at_or_below * 100:.1f}%"
        )

    @pytest.mark.unit
    def test_anomalous_log_scores_higher(self, trained_log_detector):
        """An anomalous message should score higher than the mean of normal scores."""
        normal_scores = trained_log_detector.score(NORMAL_LOGS)
        normal_mean = np.mean(normal_scores)

        anomalous = ["CRITICAL database crash OOM killed segfault core dumped"]
        anomalous_score = trained_log_detector.score(anomalous)[0]

        assert anomalous_score > normal_mean, (
            f"Anomalous score {anomalous_score:.4f} should exceed "
            f"normal mean {normal_mean:.4f}"
        )

    @pytest.mark.unit
    def test_score_values_are_nonnegative(self, trained_log_detector):
        """All scores should be >= 0."""
        messages = [
            "Request handled: GET /api/health 200 5ms",
            "CRITICAL database crash OOM killed",
            "unknown weird token $$$ %%% ^^^",
        ]
        scores = trained_log_detector.score(messages)
        assert np.all(scores >= 0), f"Found negative scores: {scores}"


# ===================================================================
# MetricAnomalyDetector
# ===================================================================


class TestMetricAnomalyDetector:
    @pytest.mark.unit
    def test_update_stats_stores_data(self, metric_detector):
        """After update_stats, the metric key should exist in stats."""
        metric_detector.update_stats("cpu", [50, 55, 60])
        assert "cpu" in metric_detector.stats
        mean, std = metric_detector.stats["cpu"]
        assert isinstance(mean, (float, np.floating))
        assert isinstance(std, (float, np.floating))

    @pytest.mark.unit
    def test_score_normal_value(self, metric_detector):
        """A value at the center of the distribution should score low."""
        values = np.random.default_rng(42).normal(loc=50, scale=5, size=200).tolist()
        metric_detector.update_stats("cpu", values)
        s = metric_detector.score("cpu", 50)
        assert s < 0.5, f"Expected score < 0.5 for normal value, got {s:.4f}"

    @pytest.mark.unit
    def test_score_anomalous_value(self, metric_detector):
        """A far-outlier value should score high."""
        values = np.random.default_rng(42).normal(loc=50, scale=5, size=200).tolist()
        metric_detector.update_stats("cpu", values)
        s = metric_detector.score("cpu", 200)
        assert s > 0.8, f"Expected score > 0.8 for extreme outlier, got {s:.4f}"

    @pytest.mark.unit
    def test_score_clamped_to_01(self, metric_detector):
        """Score should never exceed 1.0 or go below 0.0."""
        values = np.random.default_rng(42).normal(loc=50, scale=5, size=200).tolist()
        metric_detector.update_stats("cpu", values)

        # Test a huge outlier
        assert metric_detector.score("cpu", 99999) <= 1.0
        # Test value at the mean
        assert metric_detector.score("cpu", 50) >= 0.0

    @pytest.mark.unit
    def test_unknown_metric_returns_zero(self, metric_detector):
        """Scoring a metric that was never updated should return 0.0."""
        s = metric_detector.score("nonexistent_metric", 42)
        assert s == 0.0
