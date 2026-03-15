"""
Unit tests for src.models.unified_scorer
Tests the UnifiedAnomalyScorer: log scoring, metric scoring, unified combination,
graceful degradation, edge cases, and threshold behaviour.
"""

import numpy as np
import pytest

from src.models.unified_scorer import UnifiedAnomalyScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer():
    """Default scorer with PRD defaults (alpha=0.80, threshold=0.5)."""
    return UnifiedAnomalyScorer()


@pytest.fixture
def custom_scorer():
    """Scorer with custom alpha."""
    return UnifiedAnomalyScorer(alpha=0.60, anomaly_threshold=0.4)


# ===================================================================
# 1. Default weights
# ===================================================================


class TestDefaultWeights:
    @pytest.mark.unit
    def test_default_alpha(self, scorer):
        assert scorer.alpha == 0.80

    @pytest.mark.unit
    def test_default_threshold(self, scorer):
        assert scorer.anomaly_threshold == 0.5

    @pytest.mark.unit
    def test_default_log_model_weights(self, scorer):
        assert scorer.log_model_weights == {"tfidf": 0.3, "logbert": 0.7}

    @pytest.mark.unit
    def test_default_metric_model_weights(self, scorer):
        assert scorer.metric_model_weights == {"lstm_ae": 0.6, "transformer": 0.4}

    @pytest.mark.unit
    def test_log_level_weights_complete(self, scorer):
        expected_keys = {
            "DEBUG",
            "INFO",
            "WARN",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "FATAL",
        }
        assert set(scorer.LOG_LEVEL_WEIGHTS.keys()) == expected_keys
        assert scorer.LOG_LEVEL_WEIGHTS["DEBUG"] == 0.1
        assert scorer.LOG_LEVEL_WEIGHTS["CRITICAL"] == 1.0


# ===================================================================
# 2. Log scoring — basic (TF-IDF + LogBERT)
# ===================================================================


class TestLogScoringBasic:
    @pytest.mark.unit
    def test_log_scoring_basic(self, scorer):
        tfidf = np.array([0.3, 0.7, 0.1])
        logbert = np.array([0.4, 0.9, 0.2])
        levels = ["INFO", "ERROR", "DEBUG"]

        result = scorer.score_logs(tfidf, logbert, levels)

        assert "unified_scores" in result
        assert "is_anomalous" in result
        assert "model_scores" in result
        assert "rarity_priors" in result
        assert len(result["unified_scores"]) == 3
        assert result["unified_scores"].dtype == np.float64

    @pytest.mark.unit
    def test_log_scoring_model_weights_applied(self, scorer):
        """With both models, model_score = 0.3*tfidf + 0.7*logbert."""
        tfidf = np.array([1.0])
        logbert = np.array([0.0])
        levels = ["INFO"]

        result = scorer.score_logs(tfidf, logbert, levels)
        # model_score = 0.3*1.0 + 0.7*0.0 = 0.3
        expected_model = 0.3
        np.testing.assert_almost_equal(
            result["model_scores"][0], expected_model, decimal=5
        )


# ===================================================================
# 3. Log scoring — TF-IDF only (fallback)
# ===================================================================


class TestLogScoringTfidfOnly:
    @pytest.mark.unit
    def test_log_scoring_tfidf_only(self, scorer):
        tfidf = np.array([0.5, 0.8])
        levels = ["WARN", "ERROR"]

        result = scorer.score_logs(
            tfidf_scores=tfidf, logbert_scores=None, log_levels=levels
        )

        assert len(result["unified_scores"]) == 2
        # model_score should be tfidf with re-normalised weight (1.0)
        np.testing.assert_array_almost_equal(result["model_scores"], tfidf)


# ===================================================================
# 4. Log scoring — LogBERT only (fallback)
# ===================================================================


class TestLogScoringLogbertOnly:
    @pytest.mark.unit
    def test_log_scoring_logbert_only(self, scorer):
        logbert = np.array([0.6, 0.3])
        levels = ["CRITICAL", "DEBUG"]

        result = scorer.score_logs(
            tfidf_scores=None, logbert_scores=logbert, log_levels=levels
        )

        assert len(result["unified_scores"]) == 2
        # model_score should be logbert with re-normalised weight (1.0)
        np.testing.assert_array_almost_equal(result["model_scores"], logbert)


# ===================================================================
# 5. Metric scoring — basic (LSTM AE + Transformer)
# ===================================================================


class TestMetricScoringBasic:
    @pytest.mark.unit
    def test_metric_scoring_basic(self, scorer):
        lstm = np.array([0.2, 0.9, 0.5])
        trans = np.array([0.3, 0.8, 0.4])

        result = scorer.score_metrics(lstm, trans)

        assert "unified_scores" in result
        assert len(result["unified_scores"]) == 3

    @pytest.mark.unit
    def test_metric_scoring_model_weights_applied(self, scorer):
        """model_score = 0.6*lstm + 0.4*transformer."""
        lstm = np.array([1.0])
        trans = np.array([0.0])

        result = scorer.score_metrics(lstm, trans)
        expected_model = 0.6
        np.testing.assert_almost_equal(
            result["model_scores"][0], expected_model, decimal=5
        )


# ===================================================================
# 6. Metric scoring — LSTM only (fallback)
# ===================================================================


class TestMetricScoringLstmOnly:
    @pytest.mark.unit
    def test_metric_scoring_lstm_only(self, scorer):
        lstm = np.array([0.7, 0.2])

        result = scorer.score_metrics(lstm_ae_scores=lstm, transformer_scores=None)

        assert len(result["unified_scores"]) == 2
        np.testing.assert_array_almost_equal(result["model_scores"], lstm)


# ===================================================================
# 7. Anomaly threshold
# ===================================================================


class TestAnomalyThreshold:
    @pytest.mark.unit
    def test_anomaly_threshold(self, scorer):
        """Signals with unified_score > 0.5 should be flagged anomalous."""
        # Force high model scores with ERROR levels to push above threshold
        tfidf = np.array([0.9, 0.1])
        logbert = np.array([0.95, 0.05])
        levels = ["ERROR", "DEBUG"]

        result = scorer.score_logs(tfidf, logbert, levels)

        # First signal should be anomalous (high model + high rarity)
        assert result["is_anomalous"][0] is np.bool_(True)
        # Second signal should not be anomalous (low model + low rarity)
        assert result["is_anomalous"][1] is np.bool_(False)

    @pytest.mark.unit
    def test_threshold_boundary(self, scorer):
        """Score exactly at 0.5 should NOT be anomalous (> not >=)."""
        # Build input where unified score comes out to exactly 0.5
        # unified = 0.8 * model + 0.2 * rarity
        # If model = 0.5 and rarity = 0.5 → unified = 0.5
        tfidf = np.array([0.5])
        levels = ["WARN"]  # rarity = 0.5
        result = scorer.score_logs(
            tfidf_scores=tfidf, logbert_scores=None, log_levels=levels
        )
        # 0.8 * 0.5 + 0.2 * 0.5 = 0.5 — not anomalous (strictly >)
        assert result["is_anomalous"][0] is np.bool_(False)


# ===================================================================
# 8. Rarity prior — log level weights
# ===================================================================


class TestRarityPriorLogLevels:
    @pytest.mark.unit
    def test_rarity_prior_log_levels(self, scorer):
        levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        # Use zero model scores so unified == (1-alpha) * rarity
        tfidf = np.zeros(5)

        result = scorer.score_logs(tfidf, None, levels)

        expected_rarity = np.array([0.1, 0.2, 0.5, 0.8, 1.0])
        np.testing.assert_array_almost_equal(result["rarity_priors"], expected_rarity)

    @pytest.mark.unit
    def test_rarity_prior_unknown_level_defaults(self, scorer):
        """Unknown log level should default to 0.2 (INFO-like)."""
        levels = ["TRACE"]
        tfidf = np.array([0.0])

        result = scorer.score_logs(tfidf, None, levels)
        assert result["rarity_priors"][0] == 0.2


# ===================================================================
# 9. Unified combination (logs + metrics)
# ===================================================================


class TestUnifiedCombination:
    @pytest.mark.unit
    def test_unified_combination(self, scorer):
        log_result = scorer.score_logs(
            tfidf_scores=np.array([0.9, 0.1]),
            logbert_scores=np.array([0.8, 0.2]),
            log_levels=["ERROR", "INFO"],
        )
        metric_result = scorer.score_metrics(
            lstm_ae_scores=np.array([0.7]),
            transformer_scores=np.array([0.6]),
        )

        combined = scorer.score_unified(log_result, metric_result)

        # Total signals: 2 log + 1 metric = 3
        assert len(combined["unified_scores"]) == 3
        assert len(combined["sources"]) == 3
        assert set(combined["sources"]) <= {"log", "metric"}

    @pytest.mark.unit
    def test_unified_ranked_descending(self, scorer):
        log_result = scorer.score_logs(
            tfidf_scores=np.array([0.1, 0.9]),
            logbert_scores=None,
            log_levels=["DEBUG", "CRITICAL"],
        )
        metric_result = scorer.score_metrics(
            lstm_ae_scores=np.array([0.5]),
            transformer_scores=None,
        )

        combined = scorer.score_unified(log_result, metric_result)

        scores = combined["unified_scores"]
        # Verify descending order
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    @pytest.mark.unit
    def test_unified_logs_only(self, scorer):
        log_result = scorer.score_logs(
            tfidf_scores=np.array([0.5]),
            logbert_scores=None,
            log_levels=["WARN"],
        )

        combined = scorer.score_unified(log_result, None)
        assert len(combined["unified_scores"]) == 1
        assert combined["sources"][0] == "log"

    @pytest.mark.unit
    def test_unified_metrics_only(self, scorer):
        metric_result = scorer.score_metrics(
            lstm_ae_scores=np.array([0.5]),
            transformer_scores=None,
        )

        combined = scorer.score_unified(None, metric_result)
        assert len(combined["unified_scores"]) == 1
        assert combined["sources"][0] == "metric"


# ===================================================================
# 10. Custom alpha
# ===================================================================


class TestCustomAlpha:
    @pytest.mark.unit
    def test_custom_alpha(self, custom_scorer):
        assert custom_scorer.alpha == 0.60
        assert custom_scorer.anomaly_threshold == 0.4

    @pytest.mark.unit
    def test_custom_alpha_affects_scoring(self):
        scorer_a = UnifiedAnomalyScorer(alpha=0.90)
        scorer_b = UnifiedAnomalyScorer(alpha=0.10)

        tfidf = np.array([0.8])
        levels = ["DEBUG"]  # rarity = 0.1

        result_a = scorer_a.score_logs(tfidf, None, levels)
        result_b = scorer_b.score_logs(tfidf, None, levels)

        # Higher alpha → more weight on model_score (0.8)
        # Lower alpha → more weight on rarity (0.1)
        assert result_a["unified_scores"][0] > result_b["unified_scores"][0]


# ===================================================================
# 11. Empty inputs
# ===================================================================


class TestEmptyInputs:
    @pytest.mark.unit
    def test_empty_log_inputs(self, scorer):
        result = scorer.score_logs(None, None, None)
        assert len(result["unified_scores"]) == 0
        assert len(result["is_anomalous"]) == 0

    @pytest.mark.unit
    def test_empty_metric_inputs(self, scorer):
        result = scorer.score_metrics(None, None)
        assert len(result["unified_scores"]) == 0
        assert len(result["is_anomalous"]) == 0

    @pytest.mark.unit
    def test_empty_unified(self, scorer):
        result = scorer.score_unified(None, None)
        assert len(result["unified_scores"]) == 0

    @pytest.mark.unit
    def test_empty_arrays(self, scorer):
        result = scorer.score_logs(np.array([]), np.array([]), [])
        assert len(result["unified_scores"]) == 0


# ===================================================================
# 12. Score range [0, 1]
# ===================================================================


class TestScoreRange:
    @pytest.mark.unit
    def test_score_range_logs(self, scorer):
        """All unified log scores must be in [0, 1]."""
        rng = np.random.default_rng(42)
        tfidf = rng.uniform(0, 1, size=100)
        logbert = rng.uniform(0, 1, size=100)
        levels = rng.choice(
            ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"], size=100
        ).tolist()

        result = scorer.score_logs(tfidf, logbert, levels)

        assert np.all(result["unified_scores"] >= 0.0)
        assert np.all(result["unified_scores"] <= 1.0)
        assert np.all(result["model_scores"] >= 0.0)
        assert np.all(result["model_scores"] <= 1.0)

    @pytest.mark.unit
    def test_score_range_metrics(self, scorer):
        """All unified metric scores must be in [0, 1]."""
        rng = np.random.default_rng(42)
        lstm = rng.uniform(0, 1, size=50)
        trans = rng.uniform(0, 1, size=50)

        result = scorer.score_metrics(lstm, trans)

        assert np.all(result["unified_scores"] >= 0.0)
        assert np.all(result["unified_scores"] <= 1.0)

    @pytest.mark.unit
    def test_score_range_extreme_inputs(self, scorer):
        """Even with extreme input values, output stays in [0, 1]."""
        tfidf = np.array([0.0, 1.0, 999.0, -5.0])
        levels = ["DEBUG", "CRITICAL", "ERROR", "INFO"]

        result = scorer.score_logs(tfidf, None, levels)

        assert np.all(result["unified_scores"] >= 0.0)
        assert np.all(result["unified_scores"] <= 1.0)
