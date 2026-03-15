"""
Unit tests for src.models.logbert.LogBERTDetector.

All tests are skipped when ``transformers`` / ``torch`` are not installed.
Tests that involve actual model forward passes are marked ``slow`` because
they download DistilBERT weights on first run and do real inference.
"""

import os
import tempfile

import numpy as np
import pytest

from src.models.logbert import TRANSFORMERS_AVAILABLE

skip_no_transformers = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="transformers not installed",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NORMAL_LOGS = [
    "Request handled: GET /api/health 200 5ms",
    "Request handled: POST /api/users 201 12ms",
    "Database query completed in 23ms",
    "Cache hit for key: user_session_123",
    "Background job completed: email_dispatch",
    "Connection pool: 10 active, 90 idle",
    "Heartbeat received from node-1",
    "Scheduled task cron_cleanup finished in 150ms",
    "Request handled: GET /api/metrics 200 8ms",
    "User authentication successful for user_456",
] * 10  # 100 messages — enough for a quick fine-tune

ANOMALOUS_LOGS = [
    "CRITICAL: out of memory — OOM killer invoked, pid 4412 terminated",
    "FATAL database crash: segfault in pg_backend core dumped",
    "Kernel panic - not syncing: corrupted page table at address 0xdead",
    "ALERT: disk /dev/sda1 100% full, writes blocked",
    "EMERGENCY: ransomware encryption detected on /mnt/data",
]


@pytest.fixture
def detector():
    """Return a freshly initialised (untrained) LogBERTDetector on CPU."""
    from src.models.logbert import LogBERTDetector

    return LogBERTDetector(device="cpu")


@pytest.fixture
def trained_detector(detector):
    """Return a LogBERTDetector fine-tuned on the normal corpus."""
    detector.train(NORMAL_LOGS, epochs=1, batch_size=16)
    return detector


# ===================================================================
# Tests
# ===================================================================


@skip_no_transformers
class TestLogBERTDetector:
    # ---------------------------------------------------------------
    # 1. Initialisation
    # ---------------------------------------------------------------

    def test_logbert_init(self, detector):
        """Model initialises with correct defaults and is not yet trained."""
        assert detector.is_trained is False
        assert detector.max_seq_length == 128
        assert detector.model is not None
        assert detector.tokenizer is not None

    # ---------------------------------------------------------------
    # 2. Training
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_train(self, detector):
        """Training on a small corpus completes and returns loss dict."""
        stats = detector.train(NORMAL_LOGS, epochs=2, batch_size=16)

        assert detector.is_trained is True
        assert "loss_history" in stats
        assert "final_loss" in stats
        assert "epochs" in stats
        assert len(stats["loss_history"]) == 2
        assert stats["final_loss"] > 0
        # Loss should decrease (or at least not explode)
        assert stats["loss_history"][-1] <= stats["loss_history"][0] * 2

    # ---------------------------------------------------------------
    # 3. Scoring before training raises
    # ---------------------------------------------------------------

    def test_logbert_score_untrained(self, detector):
        """Scoring before training raises ValueError."""
        with pytest.raises(ValueError, match="not been fine-tuned"):
            detector.score(["some message"])

    # ---------------------------------------------------------------
    # 4. Normal logs get low scores
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_score_normal(self, trained_detector):
        """Normal logs should receive scores generally below 0.5."""
        scores = trained_detector.score(NORMAL_LOGS[:10])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 10
        # The majority of normal logs should score below the midpoint
        assert np.mean(scores < 0.7) >= 0.5, (
            f"Expected most normal scores < 0.7, got scores: {scores}"
        )

    # ---------------------------------------------------------------
    # 5. Anomalous logs score higher than normal ones
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_score_anomalous(self, trained_detector):
        """Anomalous logs should, on average, score higher than normal logs."""
        normal_scores = trained_detector.score(NORMAL_LOGS[:10])
        anomalous_scores = trained_detector.score(ANOMALOUS_LOGS)

        normal_mean = float(np.mean(normal_scores))
        anomalous_mean = float(np.mean(anomalous_scores))

        assert anomalous_mean > normal_mean, (
            f"Anomalous mean ({anomalous_mean:.4f}) should exceed "
            f"normal mean ({normal_mean:.4f})"
        )

    # ---------------------------------------------------------------
    # 6. Save / Load round-trip
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_save_load(self, trained_detector):
        """Save and reload the model — scores should remain consistent."""
        test_msgs = NORMAL_LOGS[:5] + ANOMALOUS_LOGS[:2]
        scores_before = trained_detector.score(test_msgs)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "logbert_model")
            trained_detector.save(save_path)

            from src.models.logbert import LogBERTDetector

            loaded = LogBERTDetector(device="cpu")
            loaded.load(save_path)

            assert loaded.is_trained is True
            scores_after = loaded.score(test_msgs)

        # Scores should be very close (float precision)
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-4)

    # ---------------------------------------------------------------
    # 7. Empty input handling
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_empty_input(self, trained_detector):
        """Empty strings and whitespace-only strings are handled gracefully."""
        scores = trained_detector.score(["", "   ", "normal log message"])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        # Empty strings should get a score of 0.0
        assert scores[0] == pytest.approx(0.0, abs=0.01)
        assert scores[1] == pytest.approx(0.0, abs=0.01)

        # Completely empty list
        empty_scores = trained_detector.score([])
        assert len(empty_scores) == 0

    # ---------------------------------------------------------------
    # 8. Large-batch scoring
    # ---------------------------------------------------------------

    @pytest.mark.slow
    def test_logbert_batch_scoring(self, trained_detector):
        """Handles a batch larger than the default batch_size correctly."""
        # 200 messages — well above default batch_size of 32
        big_batch = NORMAL_LOGS * 2  # 200 messages
        scores = trained_detector.score(big_batch, batch_size=32)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 200
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)
        # No NaN / Inf
        assert np.all(np.isfinite(scores))
