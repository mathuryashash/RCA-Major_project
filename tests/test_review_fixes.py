"""Regression tests for two review findings.

1. anomaly_detection/anomaly_scorer.AnomalyDetector used to be a full,
   divergent copy of models.lstm_autoencoder.AnomalyDetector (wrong
   threshold percentile, no windows=/on_epoch support) that was never
   actually instantiated -- a dormant landmine. It is now a re-export of
   the one real implementation.

2. GrangerAnalyzer.run() used to swallow every per-pair exception with a
   bare `except Exception: continue` and no logging, making a systemic
   failure (e.g. a statsmodels break) indistinguishable in the UI/report
   from "genuinely no relationship" -- undermining the project's own
   honesty claim that it distinguishes "not tested" from "nothing found".
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_anomaly_scorer_reexports_the_one_real_detector():
    from anomaly_detection.anomaly_scorer import AnomalyDetector as ReExported
    from models.lstm_autoencoder import AnomalyDetector as Canonical

    assert ReExported is Canonical, (
        "there must be exactly one AnomalyDetector implementation, not a "
        "second copy with its own (potentially divergent) behaviour"
    )


def test_granger_analyzer_logs_pair_failures_instead_of_silently_dropping(caplog):
    from causal_inference.causal_engine import GrangerAnalyzer

    # Two constant columns: statsmodels' grangercausalitytests raises on a
    # singular / zero-variance series, exercising the except-branch without
    # needing to fabricate a specific statsmodels version failure.
    df = pd.DataFrame({
        "cpu_pct": [5.0] * 30,
        "mem_pct": [10.0] * 30,
    })
    analyzer = GrangerAnalyzer(max_lag=2, significance_level=0.05)

    with caplog.at_level(logging.WARNING, logger="causal_inference.causal_engine"):
        result = analyzer.run(df, ["cpu_pct", "mem_pct"])

    assert result == {}
    assert any("Granger test failed" in r.getMessage() for r in caplog.records), (
        "a failed pair-test must leave a trace, or it reads identically to "
        "'this pair truly has no relationship' in the final report"
    )
