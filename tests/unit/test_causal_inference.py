"""
Unit tests for src.models.causal_inference.CausalInferenceEngine
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.models.causal_inference import CausalInferenceEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return CausalInferenceEngine(max_lag=6, alpha=0.05)


@pytest.fixture
def random_df():
    """DataFrame with 3 independent random columns, 100 rows."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "X": rng.standard_normal(100),
            "Y": rng.standard_normal(100),
            "Z": rng.standard_normal(100),
        }
    )


@pytest.fixture
def causal_df():
    """DataFrame where B causally follows A with lag 2, plus independent C."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(200)
    b = np.roll(a, 2) + rng.standard_normal(200) * 0.1  # B follows A with lag 2
    c = rng.standard_normal(200)  # independent
    return pd.DataFrame({"A": a, "B": b, "C": c})


@pytest.fixture
def simple_graph():
    """A->B->C directed graph with weights."""
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=0.9, p_value=0.01)
    g.add_edge("B", "C", weight=0.8, p_value=0.02)
    return g


# ===================================================================
# Tests
# ===================================================================


class TestCausalInferenceEngine:
    @pytest.mark.unit
    def test_build_graph_returns_digraph(self, engine, random_df):
        """build_causal_graph should return a networkx DiGraph."""
        graph = engine.build_causal_graph(random_df)
        assert isinstance(graph, nx.DiGraph)

    @pytest.mark.unit
    def test_graph_nodes_match_columns(self, engine, random_df):
        """Nodes of the returned graph should match DataFrame column names."""
        graph = engine.build_causal_graph(random_df)
        assert set(graph.nodes) == set(random_df.columns)

    @pytest.mark.unit
    def test_causal_signal_creates_edge(self, engine, causal_df):
        """When B = shifted A + noise, an edge from A to B should exist."""
        graph = engine.build_causal_graph(causal_df)
        assert graph.has_edge("A", "B"), (
            f"Expected edge A->B in graph. Edges found: {list(graph.edges())}"
        )

    @pytest.mark.unit
    def test_edge_has_weight_and_pvalue(self, engine, causal_df):
        """Edges should carry 'weight' and 'p_value' attributes."""
        graph = engine.build_causal_graph(causal_df)
        # At least one edge should exist in the causal DataFrame
        assert len(graph.edges()) > 0, "Expected at least one edge"
        for u, v, data in graph.edges(data=True):
            assert "weight" in data, f"Edge ({u}->{v}) missing 'weight'"
            assert "p_value" in data, f"Edge ({u}->{v}) missing 'p_value'"

    @pytest.mark.unit
    def test_rank_root_causes_returns_sorted_list(self, engine, simple_graph):
        """rank_root_causes should return a list of dicts sorted descending by rca_score."""
        anomaly_scores = {"A": 0.9, "B": 0.5, "C": 0.1}
        ranked = engine.rank_root_causes(simple_graph, anomaly_scores)

        assert isinstance(ranked, list)
        assert len(ranked) > 0
        # Each element is a dict with 'signal' and 'rca_score'
        for item in ranked:
            assert isinstance(item, dict)
            assert "signal" in item
            assert "rca_score" in item

        # Verify descending order by rca_score
        scores = [item["rca_score"] for item in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"Ranking not sorted descending: {ranked}"
        )

    @pytest.mark.unit
    def test_rank_highest_anomaly_node_first(self, engine, simple_graph):
        """In A->B->C with A having the highest anomaly score, A should rank first or near first."""
        anomaly_scores = {"A": 0.9, "B": 0.5, "C": 0.1}
        ranked = engine.rank_root_causes(simple_graph, anomaly_scores)

        node_names = [item["signal"] for item in ranked]
        # A should be in top 2 at minimum (PageRank + personalization biases toward A)
        assert "A" in node_names[:2], (
            f"Expected 'A' in top-2 ranked nodes, got: {ranked}"
        )

    @pytest.mark.unit
    def test_rank_empty_graph(self, engine):
        """An empty graph should return an empty list."""
        g = nx.DiGraph()
        ranked = engine.rank_root_causes(g, {})
        assert ranked == []

    @pytest.mark.unit
    def test_build_graph_with_independent_signals(self, engine, random_df):
        """Three independent random signals should produce few or no edges."""
        graph = engine.build_causal_graph(random_df)
        # With independent signals at alpha=0.05, we might get a false positive
        # but the total should be very small (≤2 out of 6 possible directed pairs)
        assert len(graph.edges()) <= 2, (
            f"Expected <=2 spurious edges for independent signals, "
            f"got {len(graph.edges())}: {list(graph.edges())}"
        )
