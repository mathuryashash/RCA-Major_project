"""
Integration Tests — Full RCA Pipeline (End-to-End)
====================================================

Tests the complete 7-stage pipeline:
  Stage 1 : Synthetic data generation
  Stage 2 : Preprocessing & normalisation
  Stage 3 : LSTM Autoencoder training
  Stage 4 : Anomaly detection
  Stage 5 : Causal inference (Granger causality + graph building)
  Stage 6 : Root cause ranking
  Stage 7 : Report generation

All tests use very short training (2-3 epochs, small windows) so the suite
runs in < 2 minutes on CPU without requiring a GPU.

Run with:
    cd d:/vscode/majorprojectt/rca-system
    pytest tests/test_integration.py -v
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import networkx as nx

# ── path setup ────────────────────────────────────────────────────────────────
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from src.preprocessing.data_normalizer import DataPreprocessor
from src.models.lstm_autoencoder import LSTMAutoencoder, AnomalyDetectionTrainer
from src.causal_inference.granger_causality import (
    GrangerCausalityAnalyzer,
    TemporalPrecedenceAnalyzer,
    CausalGraphBuilder,
    CausalInferenceEngine,
)
from src.root_cause_ranking.scorer import RootCauseRanker, correlate_deployment_events


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def generator():
    return SyntheticMetricsGenerator(seed=42)


@pytest.fixture(scope="module")
def normal_df(generator):
    """7 days of normal data — enough for Granger tests."""
    return generator.generate_normal_behavior(duration_days=7)


@pytest.fixture(scope="module")
def failure_df(generator, normal_df):
    """Normal data with a 'database_slow_query' failure injected at day 5."""
    n = len(normal_df)
    failure_start = int(n * 0.7)
    df, meta = generator.inject_failure_scenario(
        normal_df,
        failure_type="database_slow_query",
        start_idx=failure_start,
        duration_samples=150,
        severity=1.0,
    )
    return df, meta, failure_start


@pytest.fixture(scope="module")
def preprocessor(normal_df):
    """Fitted DataPreprocessor (fitted on normal data only)."""
    pp = DataPreprocessor(window_size=30, stride=5)
    pp.fit(normal_df)
    return pp


@pytest.fixture(scope="module")
def trained_model(normal_df, preprocessor):
    """LSTM Autoencoder trained for 3 epochs on normal data windows."""
    import torch

    arr = preprocessor.transform(normal_df)   # (T, F)
    windows, _ = preprocessor.create_windows(arr, stride=10)   # (N, 30, F)

    split = int(len(windows) * 0.8)
    train_w = windows[:split]
    val_w   = windows[split:]

    n_features = arr.shape[1]
    model = LSTMAutoencoder(
        input_size=n_features,
        sequence_length=30,
        hidden_size=16,
        latent_size=8,
        num_layers=1,
    )
    trainer = AnomalyDetectionTrainer(model, lr=1e-3)
    trainer.fit(train_w, val_w, epochs=3, batch_size=16, patience=5)
    trainer.calibrate_thresholds(val_w, percentile=95.0)
    return trainer, n_features


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_normal_df_type(self, normal_df):
        assert isinstance(normal_df, pd.DataFrame)

    def test_normal_df_has_datetime_index(self, normal_df):
        assert isinstance(normal_df.index, pd.DatetimeIndex)

    def test_normal_df_no_nans(self, normal_df):
        assert not normal_df.isna().any().any()

    def test_failure_injection_shape_preserved(self, failure_df, normal_df):
        df, meta, _ = failure_df
        assert df.shape == normal_df.shape

    def test_failure_metadata_structure(self, failure_df):
        _, meta, _ = failure_df
        assert "failure_type" in meta
        assert "root_cause" in meta
        assert "causal_chain" in meta
        assert len(meta["causal_chain"]) > 0

    def test_failure_latency_increased(self, failure_df, normal_df):
        df, _, start = failure_df
        # p95 latency should spike significantly after failure injection
        before = normal_df["api_latency_p95_ms"].iloc[start - 10 : start].mean()
        after  = df["api_latency_p95_ms"].iloc[start + 100 : start + 150].mean()
        assert after > before * 1.5, "Latency should clearly increase after database_slow_query"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:

    def test_output_shape(self, preprocessor, normal_df):
        arr = preprocessor.transform(normal_df)
        assert arr.ndim == 2
        assert arr.shape[0] == len(normal_df)
        assert arr.shape[1] == preprocessor.n_features

    def test_output_dtype_float32(self, preprocessor, normal_df):
        arr = preprocessor.transform(normal_df)
        assert arr.dtype == np.float32

    def test_windows_shape(self, preprocessor, normal_df):
        arr = preprocessor.transform(normal_df)
        windows, _ = preprocessor.create_windows(arr, stride=10)
        assert windows.ndim == 3
        assert windows.shape[1] == preprocessor.window_size
        assert windows.shape[2] == preprocessor.n_features

    def test_quality_report_high_score(self, preprocessor, normal_df):
        report = preprocessor.validate_data_quality(normal_df)
        assert report["quality_score"] >= 80
        assert "issues" in report
        assert "missing_pct" in report

    def test_inverse_transform_roundtrip(self, preprocessor, normal_df):
        arr = preprocessor.transform(normal_df)
        recovered = preprocessor.inverse_transform(arr)
        # Should be close to original after normalize → denormalize
        assert recovered.shape == normal_df.shape
        for col in recovered.columns:
            diff = (recovered[col] - normal_df[col]).abs().mean()
            assert diff < 1e-3, f"Roundtrip error too large for {col}: {diff}"

    def test_unfitted_raises(self):
        from src.preprocessing.data_normalizer import DataPreprocessor
        pp = DataPreprocessor()
        gen = SyntheticMetricsGenerator(seed=0)
        df = gen.generate_normal_behavior(duration_days=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class TestLSTMAutoencoder:

    def test_model_forward_pass(self, trained_model, preprocessor, normal_df):
        import torch
        trainer, n_features = trained_model
        arr = preprocessor.transform(normal_df)
        windows, _ = preprocessor.create_windows(arr, stride=50)
        # Forward pass should succeed without error
        t = torch.from_numpy(windows[:4])
        recon, latent = trainer.model(t)
        assert recon.shape == t.shape
        assert latent.shape == (4, trainer.model.latent_size)

    def test_thresholds_calibrated(self, trained_model):
        trainer, _ = trained_model
        assert trainer.threshold_global is not None
        assert trainer.threshold_per_feature is not None
        assert trainer.threshold_global > 0

    def test_anomaly_scores_shape(self, trained_model, preprocessor, normal_df):
        trainer, _ = trained_model
        arr = preprocessor.transform(normal_df)
        windows, _ = preprocessor.create_windows(arr, stride=20)
        scores = trainer.get_anomaly_scores(windows)
        assert "sample_scores" in scores
        assert "feature_scores" in scores
        assert len(scores["sample_scores"]) == len(windows)

    def test_normal_data_low_anomaly_rate(self, trained_model, preprocessor, normal_df):
        """Most normal windows should NOT be flagged as anomalous."""
        trainer, _ = trained_model
        arr = preprocessor.transform(normal_df)
        windows, _ = preprocessor.create_windows(arr, stride=10)
        result = trainer.detect_anomalies(windows, score_threshold=1.0)
        false_positive_rate = result["is_anomaly"].mean()
        # At a 95th-percentile threshold, ~5% false positives expected
        assert false_positive_rate < 0.15, (
            f"False positive rate {false_positive_rate:.2%} is too high on normal data"
        )

    def test_failure_data_higher_anomaly_rate(
        self, trained_model, preprocessor, failure_df, normal_df
    ):
        """Failure windows should have more anomalies than normal windows."""
        trainer, _ = trained_model
        fail_df, _, fail_start = failure_df

        arr_fail = preprocessor.transform(fail_df.iloc[fail_start:])
        if len(arr_fail) < preprocessor.window_size:
            pytest.skip("Failure window too short for this test")

        windows_fail, _ = preprocessor.create_windows(arr_fail, stride=5)
        result_fail = trainer.detect_anomalies(windows_fail, score_threshold=1.0)
        anomaly_rate_fail = result_fail["is_anomaly"].mean()

        arr_norm = preprocessor.transform(normal_df.iloc[:fail_start])
        windows_norm, _ = preprocessor.create_windows(arr_norm, stride=10)
        result_norm = trainer.detect_anomalies(windows_norm, score_threshold=1.0)
        anomaly_rate_norm = result_norm["is_anomaly"].mean()

        assert anomaly_rate_fail >= anomaly_rate_norm, (
            "Failure data should have at least as many anomalies as normal data"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Causal Inference
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalInference:
    """Tests for Granger causality, temporal precedence, and graph building."""

    @pytest.fixture(scope="class")
    def anomaly_scores(self):
        return {
            "api_latency_p50_ms":   1.8,
            "api_latency_p95_ms":   2.1,
            "db_connections_active": 1.6,
            "error_rate_percent":   1.9,
        }

    @pytest.fixture(scope="class")
    def anomaly_times(self):
        base = pd.Timestamp("2024-01-01 12:00:00")
        return {
            "api_latency_p50_ms":   base,
            "api_latency_p95_ms":   base + pd.Timedelta(minutes=5),
            "db_connections_active": base + pd.Timedelta(minutes=15),
            "error_rate_percent":   base + pd.Timedelta(minutes=30),
        }

    def test_granger_returns_dict(self, failure_df, anomaly_scores):
        df, _, start = failure_df
        window = df.iloc[max(0, start - 50): start + 150]
        analyzer = GrangerCausalityAnalyzer(max_lag=3, significance_level=0.1)
        pairs = analyzer.test_all_pairs(window, list(anomaly_scores.keys()))
        assert isinstance(pairs, dict)

    def test_granger_pairs_have_required_keys(self, failure_df, anomaly_scores):
        df, _, start = failure_df
        window = df.iloc[max(0, start - 50): start + 150]
        analyzer = GrangerCausalityAnalyzer(max_lag=3, significance_level=0.1)
        pairs = analyzer.test_all_pairs(window, list(anomaly_scores.keys()))
        for (cause, effect), info in pairs.items():
            assert "is_causal" in info
            assert "p_value" in info
            assert "strength" in info
            assert cause != effect

    def test_temporal_precedence_filters_reversed_edges(self, anomaly_scores, anomaly_times):
        # Manually create a pair where effect appears BEFORE cause — should be filtered
        inverted_scores = {
            "api_latency_p50_ms":   1.8,
            "error_rate_percent":   1.9,
        }
        # error_rate appears at T+30, api_latency at T=0
        # edge (error_rate → api_latency) should be REMOVED by temporal filter
        causal_pairs = {
            ("error_rate_percent", "api_latency_p50_ms"): {
                "is_causal": True,
                "p_value":   0.01,
                "strength":  0.99,
                "best_lag":  1,
            }
        }
        analyzer = TemporalPrecedenceAnalyzer()
        valid = analyzer.verify_edges(causal_pairs, anomaly_times)
        # The reversed edge should be filtered out
        assert ("error_rate_percent", "api_latency_p50_ms") not in valid

    def test_causal_graph_is_directed(self, anomaly_scores, anomaly_times):
        causal_pairs = {
            ("api_latency_p50_ms", "db_connections_active"): {
                "is_causal": True, "p_value": 0.01,
                "strength": 0.9, "best_lag": 2,
                "temporal_valid": True, "lag_minutes": 15.0,
            }
        }
        builder = CausalGraphBuilder()
        G = builder.build(causal_pairs, anomaly_scores)
        assert isinstance(G, nx.DiGraph)

    def test_causal_graph_is_acyclic(self, anomaly_scores):
        # Even if we supply cyclic pairs, the builder should break them
        cyclic_pairs = {
            ("A", "B"): {"is_causal": True, "p_value": 0.01, "strength": 0.8, "best_lag": 1, "temporal_valid": True},
            ("B", "C"): {"is_causal": True, "p_value": 0.01, "strength": 0.7, "best_lag": 1, "temporal_valid": True},
            ("C", "A"): {"is_causal": True, "p_value": 0.02, "strength": 0.5, "best_lag": 1, "temporal_valid": True},
        }
        mock_scores = {"A": 1.5, "B": 1.8, "C": 1.6}
        builder = CausalGraphBuilder()
        G = builder.build(cyclic_pairs, mock_scores)
        assert nx.is_directed_acyclic_graph(G), "Causal graph must be a DAG"

    def test_causal_engine_returns_graph(self, failure_df, anomaly_scores, anomaly_times):
        df, _, start = failure_df
        window = df.iloc[max(0, start - 50): start + 150]
        engine = CausalInferenceEngine(max_lag=3, significance_level=0.1)
        G = engine.run(
            metrics_df=window,
            anomaly_scores=anomaly_scores,
            anomaly_first_detected=anomaly_times,
        )
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == len(anomaly_scores)

    def test_format_results_structure(self, failure_df, anomaly_scores, anomaly_times):
        df, _, start = failure_df
        window = df.iloc[max(0, start - 50): start + 150]
        engine = CausalInferenceEngine(max_lag=3, significance_level=0.1)
        G = engine.run(window, anomaly_scores, anomaly_times)
        result = engine.format_results(G)
        assert "edges" in result
        assert "potential_root_causes" in result
        assert "symptoms" in result
        assert "total_nodes" in result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Root Cause Ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestRootCauseRanking:

    @pytest.fixture(scope="class")
    def synthetic_graph(self):
        """Simple 4-node causal graph: A → B → C, A → D."""
        G = nx.DiGraph()
        G.add_node("A", anomaly_score=2.0)
        G.add_node("B", anomaly_score=1.8)
        G.add_node("C", anomaly_score=1.6)
        G.add_node("D", anomaly_score=1.5)
        G.add_edge("A", "B", strength=0.9, lag=1, p_value=0.01)
        G.add_edge("B", "C", strength=0.8, lag=2, p_value=0.02)
        G.add_edge("A", "D", strength=0.7, lag=1, p_value=0.03)
        return G

    @pytest.fixture(scope="class")
    def anomaly_scores(self):
        base = pd.Timestamp("2024-01-01 12:00:00")
        return {
            "A": 2.0, "B": 1.8, "C": 1.6, "D": 1.5,
        }

    @pytest.fixture(scope="class")
    def anomaly_times(self):
        base = pd.Timestamp("2024-01-01 12:00:00")
        return {
            "A": base,
            "B": base + pd.Timedelta(minutes=5),
            "C": base + pd.Timedelta(minutes=15),
            "D": base + pd.Timedelta(minutes=8),
        }

    def test_rank_returns_list(self, synthetic_graph, anomaly_scores, anomaly_times):
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        assert isinstance(ranked, list)
        assert len(ranked) == 4

    def test_rank_1_is_highest_score(self, synthetic_graph, anomaly_scores, anomaly_times):
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        scores = [rc.final_score for rc in ranked]
        assert scores == sorted(scores, reverse=True), "Ranked list must be sorted descending"
        assert ranked[0].rank == 1

    def test_source_node_ranks_first(self, synthetic_graph, anomaly_scores, anomaly_times):
        """Node A is the earliest anomaly and has the most outflow — should rank #1."""
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        top = ranked[0].metric
        assert top == "A", f"True root cause 'A' should rank #1, got '{top}'"

    def test_confidence_labels_valid(self, synthetic_graph, anomaly_scores, anomaly_times):
        valid_labels = {"Critical", "High", "Medium", "Low", "Very Low"}
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        for rc in ranked:
            assert rc.confidence in valid_labels

    def test_downstream_effects_populated(self, synthetic_graph, anomaly_scores, anomaly_times):
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        top = next(r for r in ranked if r.metric == "A")
        assert len(top.downstream_effects) > 0, "A should have downstream effects B, C, D"

    def test_justification_non_empty(self, synthetic_graph, anomaly_scores, anomaly_times):
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        for rc in ranked:
            assert len(rc.justification) > 10

    def test_summary_report_non_empty(self, synthetic_graph, anomaly_scores, anomaly_times):
        ranker = RootCauseRanker()
        ranked = ranker.rank(synthetic_graph, anomaly_scores, anomaly_times)
        summary = ranker.generate_summary_report(ranked, top_n=3)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_event_correlation_helper(self, anomaly_times):
        events_df = pd.DataFrame([{
            "timestamp":   pd.Timestamp("2024-01-01 11:45:00"),
            "description": "Schema migration deployed",
            "type":        "deployment",
        }])
        correlations = correlate_deployment_events(anomaly_times, events_df, window_hours=1.0)
        assert isinstance(correlations, list)
        # "A" was first at 12:00, event at 11:45 — delta = 15 min → should correlate
        metrics_correlated = {c["affected_metric"] for c in correlations}
        assert "A" in metrics_correlated

    def test_empty_graph_returns_empty_list(self):
        ranker = RootCauseRanker()
        G = nx.DiGraph()
        result = ranker.rank(G, {}, {})
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Report Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestReportGeneration:

    @pytest.fixture(scope="class")
    def sample_graph(self):
        G = nx.DiGraph()
        G.add_edge("db_query_latency", "api_latency", strength=0.9)
        G.add_edge("api_latency", "error_rate", strength=0.8)
        return G

    @pytest.fixture(scope="class")
    def sample_ranked(self):
        return [
            ("db_query_latency", 0.87, {
                "out_edges": ["api_latency"],
                "components": {
                    "temporal_priority": 1.0,
                    "anomaly_severity": 0.9,
                    "causal_outflow": 1.0,
                },
                "pagerank": 0.5,
            }),
            ("api_latency", 0.55, {
                "out_edges": ["error_rate"],
                "components": {
                    "temporal_priority": 0.5,
                    "anomaly_severity": 0.8,
                    "causal_outflow": 0.5,
                },
                "pagerank": 0.3,
            }),
        ]

    @pytest.fixture(scope="class")
    def sample_times(self):
        return {
            "db_query_latency": pd.Timestamp("2024-01-01 10:00:00"),
            "api_latency":      pd.Timestamp("2024-01-01 10:05:00"),
            "error_rate":       pd.Timestamp("2024-01-01 10:15:00"),
        }

    def test_report_is_string(self, sample_graph, sample_ranked, sample_times):
        from src.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        report = gen.generate_report("INC-TEST-001", sample_ranked, sample_graph, sample_times)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_incident_id(self, sample_graph, sample_ranked, sample_times):
        from src.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        report = gen.generate_report("INC-TEST-999", sample_ranked, sample_graph, sample_times)
        assert "INC-TEST-999" in report

    def test_report_contains_primary_root_cause(self, sample_graph, sample_ranked, sample_times):
        from src.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        report = gen.generate_report("INC-001", sample_ranked, sample_graph, sample_times)
        assert "db_query_latency" in report

    def test_report_includes_confidence(self, sample_graph, sample_ranked, sample_times):
        from src.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        report = gen.generate_report("INC-001", sample_ranked, sample_graph, sample_times)
        # Confidence percentage should appear
        assert "%" in report

    def test_empty_candidates_returns_fallback(self, sample_graph, sample_times):
        from src.reporting.report_generator import ReportGenerator
        gen = ReportGenerator()
        report = gen.generate_report("INC-EMPTY", [], sample_graph, sample_times)
        assert isinstance(report, str)


# ─────────────────────────────────────────────────────────────────────────────
# Full End-to-End Pipeline Test
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """
    Single integration test that wires every stage together and verifies
    the system produces a non-trivial output for a known failure scenario.
    Uses short training so this runs in < 60 s on CPU.
    """

    def test_full_pipeline_database_failure(self):
        """
        Run the complete RCA pipeline on a 'database_slow_query' scenario
        and assert that key outputs are structurally correct.
        """
        import torch

        # ── Stage 1: Data generation ──────────────────────────────────────────
        gen = SyntheticMetricsGenerator(seed=99)
        normal_df = gen.generate_normal_behavior(duration_days=7)
        failure_start = len(normal_df) - 200
        failure_df, metadata = gen.inject_failure_scenario(
            normal_df,
            failure_type="database_slow_query",
            start_idx=failure_start,
            duration_samples=200,
            severity=1.0,
        )

        # ── Stage 2: Preprocessing ────────────────────────────────────────────
        pp = DataPreprocessor(window_size=30, stride=10)
        pp.fit(normal_df)
        normal_arr = pp.transform(normal_df)     # (T_normal, F)
        failure_arr = pp.transform(failure_df)   # (T_failure, F)

        train_w, _ = pp.create_windows(normal_arr, stride=10)
        split = int(len(train_w) * 0.8)
        val_w = train_w[split:]
        train_w = train_w[:split]

        # ── Stage 3: LSTM Training ────────────────────────────────────────────
        n_features = normal_arr.shape[1]
        model = LSTMAutoencoder(
            input_size=n_features,
            sequence_length=30,
            hidden_size=16,
            latent_size=8,
            num_layers=1,
        )
        trainer = AnomalyDetectionTrainer(model, lr=1e-3)
        trainer.fit(train_w, val_w, epochs=3, batch_size=16, patience=5)
        trainer.calibrate_thresholds(val_w, percentile=95.0)

        # ── Stage 4: Anomaly Detection ────────────────────────────────────────
        fail_windows, win_indices, win_ts = pp.create_windows_from_df(
            failure_df, stride=5
        )
        detection = trainer.detect_anomalies(fail_windows, score_threshold=1.0)
        anomaly_flags = detection["anomalous_features"]    # (N, F)
        sample_scores = detection["scores"]["sample_scores"]  # (N,)

        feat_names = pp.get_feature_names()
        # Build anomaly_scores and anomaly_times from worst windows
        anomaly_scores: dict = {}
        anomaly_times:  dict = {}
        for fi, feat in enumerate(feat_names):
            col_scores = detection["scores"]["feature_scores"][:, fi]
            if col_scores.max() > 1.0:
                anomaly_scores[feat] = float(col_scores.max())
                first_anomaly_win = int(np.argmax(col_scores > 1.0))
                if first_anomaly_win < len(win_ts):
                    anomaly_times[feat] = pd.Timestamp(win_ts[first_anomaly_win])

        # Need at least 2 anomalous metrics for causal analysis
        if len(anomaly_scores) < 2:
            # Fallback: use top-N by score even if below threshold
            sorted_feats = sorted(
                zip(feat_names, detection["scores"]["feature_scores"].max(axis=0)),
                key=lambda x: x[1], reverse=True
            )[:4]
            for feat, score in sorted_feats:
                anomaly_scores[feat] = float(score)
                anomaly_times[feat] = pd.Timestamp(win_ts[0]) if len(win_ts) > 0 else pd.Timestamp("2024-01-01")

        anomalous_metrics = list(anomaly_scores.keys())

        # ── Stage 5: Causal Inference ─────────────────────────────────────────
        engine = CausalInferenceEngine(
            max_lag=3, significance_level=0.1, min_anomaly_score=0.0
        )
        causal_graph = engine.run(
            metrics_df=failure_df,
            anomaly_scores=anomaly_scores,
            anomaly_first_detected=anomaly_times if anomaly_times else None,
        )

        # Graph must have nodes for each anomalous metric
        assert causal_graph.number_of_nodes() == len(anomaly_scores), (
            f"Expected {len(anomaly_scores)} nodes, got {causal_graph.number_of_nodes()}"
        )
        assert nx.is_directed_acyclic_graph(causal_graph), "Causal graph must be a DAG"

        # ── Stage 6: Root Cause Ranking ───────────────────────────────────────
        ranker = RootCauseRanker()
        ranked = ranker.rank(
            causal_graph,
            anomaly_scores,
            anomaly_times if anomaly_times else None,
        )

        assert len(ranked) == len(anomaly_scores)
        assert ranked[0].rank == 1
        scores_list = [rc.final_score for rc in ranked]
        assert scores_list == sorted(scores_list, reverse=True)

        # ── Stage 7: Report Generation ────────────────────────────────────────
        from src.reporting.report_generator import ReportGenerator
        reporter = ReportGenerator()
        ranked_tuples = [
            (rc.metric, rc.final_score, {
                "out_edges":  rc.downstream_effects,
                "components": rc.factor_scores,
                "pagerank":   0.0,
            })
            for rc in ranked
        ]
        report = reporter.generate_report(
            incident_id="E2E-TEST-001",
            ranked_candidates=ranked_tuples,
            causal_graph=causal_graph,
            anomaly_times=anomaly_times,
        )

        assert isinstance(report, str)
        assert "E2E-TEST-001" in report
        assert len(report) > 200

        print(f"\n[E2E] Top root cause: {ranked[0].metric} "
              f"(confidence={ranked[0].confidence}, "
              f"score={ranked[0].final_score:.4f})")
        print(f"[E2E] Ground truth: {metadata['root_cause']}")
        print(f"[E2E] Causal chain: {' → '.join(metadata.get('causal_chain', []))}")
        print(f"[E2E] Causal graph edges: {causal_graph.number_of_edges()}")
