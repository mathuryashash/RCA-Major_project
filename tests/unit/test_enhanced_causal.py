"""
Tests for enhanced causal inference features (FR-23, FR-24).

Covers:
- Benjamini-Hochberg FDR correction
- PC Algorithm integration
- Majority-vote edge combination
- Cycle breaking
- Backward compatibility
- Edge cases (empty, constant, two-signal)
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.models.causal_inference import (
    CAUSAL_LEARN_AVAILABLE,
    CausalInferenceEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return CausalInferenceEngine(max_lag=4, alpha=0.05, fdr_alpha=0.05)


@pytest.fixture
def causal_df():
    """A->B with lag-2 causal relationship, plus independent C.  200 rows."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(200)
    b = np.roll(a, 2) + rng.standard_normal(200) * 0.1
    c = rng.standard_normal(200)
    return pd.DataFrame({"A": a, "B": b, "C": c})


@pytest.fixture
def noisy_many_signals():
    """6 independent random signals — likely to produce spurious edges without FDR."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({f"S{i}": rng.standard_normal(120) for i in range(6)})


# ===================================================================
# Tests
# ===================================================================


class TestFDRCorrection:
    """FR-23: Benjamini-Hochberg FDR correction."""

    @pytest.mark.unit
    def test_fdr_correction_applied(self, engine, noisy_many_signals):
        """FDR correction should produce fewer (or equal) edges than raw alpha
        on independent signals, because it controls the false-discovery rate."""
        graph_fdr = engine.build_causal_graph(
            noisy_many_signals, use_pc=False, use_fdr=True
        )
        graph_raw = engine.build_causal_graph(
            noisy_many_signals, use_pc=False, use_fdr=False
        )
        assert len(graph_fdr.edges()) <= len(graph_raw.edges()), (
            f"FDR graph has {len(graph_fdr.edges())} edges but raw has "
            f"{len(graph_raw.edges())} — FDR should not add more."
        )


class TestPCAlgorithm:
    """FR-24: PC Algorithm integration."""

    @pytest.mark.unit
    @pytest.mark.skipif(
        not CAUSAL_LEARN_AVAILABLE,
        reason="causal-learn not installed",
    )
    def test_pc_algorithm_integration(self, engine, causal_df):
        """PC algorithm runs without error and produces a DiGraph."""
        pc_graph = engine._build_pc_graph(causal_df)
        assert isinstance(pc_graph, nx.DiGraph)
        assert set(pc_graph.nodes) == set(causal_df.columns)


class TestMajorityVoteCombination:
    """Majority-vote (union) combination of Granger and PC edges."""

    @pytest.mark.unit
    def test_majority_vote_combination(self, engine):
        """Combined graph should contain the union of edges from both inputs."""
        nodes = ["A", "B", "C"]

        g1 = nx.DiGraph()
        g1.add_nodes_from(nodes)
        g1.add_edge("A", "B", weight=0.9, p_value=0.01)

        g2 = nx.DiGraph()
        g2.add_nodes_from(nodes)
        g2.add_edge("B", "C", weight=1.0, p_value=0.0)

        combined = engine._majority_vote_combine(g1, g2, nodes)

        assert combined.has_edge("A", "B"), "Missing Granger-only edge A->B"
        assert combined.has_edge("B", "C"), "Missing PC-only edge B->C"
        assert len(combined.edges()) == 2

    @pytest.mark.unit
    def test_majority_vote_weight_averaging(self, engine):
        """When both methods agree on an edge, weight should be the average."""
        nodes = ["X", "Y"]
        g1 = nx.DiGraph()
        g1.add_nodes_from(nodes)
        g1.add_edge("X", "Y", weight=0.8, p_value=0.05)

        g2 = nx.DiGraph()
        g2.add_nodes_from(nodes)
        g2.add_edge("X", "Y", weight=1.0, p_value=0.0)

        combined = engine._majority_vote_combine(g1, g2, nodes)
        w = combined["X"]["Y"]["weight"]
        assert abs(w - 0.9) < 1e-9, f"Expected avg weight 0.9, got {w}"


class TestCycleBreaking:
    """Cycle removal via weakest-edge deletion."""

    @pytest.mark.unit
    def test_cycle_breaking(self):
        """After _break_cycles the graph must be a DAG."""
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=0.9)
        g.add_edge("B", "C", weight=0.7)
        g.add_edge("C", "A", weight=0.3)  # weakest — should be removed

        CausalInferenceEngine._break_cycles(g)

        assert nx.is_directed_acyclic_graph(g), (
            f"Graph still has cycles: {list(nx.simple_cycles(g))}"
        )
        # The weakest edge (C->A, weight 0.3) should have been removed
        assert not g.has_edge("C", "A"), "Weakest edge C->A should be removed"
        assert g.has_edge("A", "B") and g.has_edge("B", "C")

    @pytest.mark.unit
    def test_cycle_breaking_multiple_cycles(self):
        """Handles graphs with multiple overlapping cycles."""
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=0.9)
        g.add_edge("B", "A", weight=0.2)  # cycle 1: A<->B
        g.add_edge("B", "C", weight=0.8)
        g.add_edge("C", "B", weight=0.1)  # cycle 2: B<->C

        CausalInferenceEngine._break_cycles(g)
        assert nx.is_directed_acyclic_graph(g)


class TestBackwardCompatibility:
    """Existing API contract must be preserved."""

    @pytest.mark.unit
    def test_backward_compatibility_granger_only(self, engine, causal_df):
        """use_pc=False, use_fdr=False reproduces the legacy code path."""
        graph = engine.build_causal_graph(causal_df, use_pc=False, use_fdr=False)
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes) == set(causal_df.columns)
        # The strong A->B causal signal should still be detected
        assert graph.has_edge("A", "B"), (
            f"Legacy mode should detect A->B. Edges: {list(graph.edges())}"
        )
        # Edges must still carry weight and p_value
        for u, v, data in graph.edges(data=True):
            assert "weight" in data
            assert "p_value" in data

    @pytest.mark.unit
    def test_default_call_signature(self, engine, causal_df):
        """Calling build_causal_graph(df) with no extra args should work."""
        graph = engine.build_causal_graph(causal_df)
        assert isinstance(graph, nx.DiGraph)

    @pytest.mark.unit
    def test_rank_root_causes_unchanged(self, engine):
        """rank_root_causes API accepts old positional args and returns dicts."""
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=0.9, p_value=0.01)
        g.add_edge("B", "C", weight=0.8, p_value=0.02)

        ranked = engine.rank_root_causes(g, {"A": 0.9, "B": 0.5, "C": 0.1})
        assert isinstance(ranked, list)
        assert len(ranked) == 3
        scores = [item["rca_score"] for item in ranked]
        assert scores == sorted(scores, reverse=True)


class TestEdgeCases:
    """Robust handling of degenerate inputs."""

    @pytest.mark.unit
    def test_empty_dataframe(self, engine):
        """Empty DataFrame should return a graph with no nodes/edges."""
        df = pd.DataFrame()
        graph = engine.build_causal_graph(df)
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) == 0
        assert len(graph.edges()) == 0

    @pytest.mark.unit
    def test_constant_columns(self, engine):
        """Constant (zero-variance) columns should not crash."""
        df = pd.DataFrame(
            {
                "A": np.zeros(100),
                "B": np.ones(100),
                "C": np.full(100, 42.0),
            }
        )
        graph = engine.build_causal_graph(df, use_pc=False, use_fdr=True)
        assert isinstance(graph, nx.DiGraph)
        # No meaningful edges can be detected from constants
        assert len(graph.edges()) == 0

    @pytest.mark.unit
    def test_two_signals_minimum(self, engine):
        """Exactly 2 signals should work fine."""
        rng = np.random.default_rng(7)
        a = rng.standard_normal(150)
        b = np.roll(a, 3) + rng.standard_normal(150) * 0.1
        df = pd.DataFrame({"X": a, "Y": b})

        graph = engine.build_causal_graph(df, use_pc=False, use_fdr=True)
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes) == {"X", "Y"}

    @pytest.mark.unit
    def test_single_column(self, engine):
        """A single column means no pairs to test — should return empty graph."""
        df = pd.DataFrame({"Solo": np.random.randn(50)})
        graph = engine.build_causal_graph(df)
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.edges()) == 0
