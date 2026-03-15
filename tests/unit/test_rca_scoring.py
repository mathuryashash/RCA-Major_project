"""
Unit tests for the 5-factor RCA scoring formula in CausalInferenceEngine.

Tests the rank_root_causes method with:
  rca_score = 0.35 * pagerank + 0.25 * temporal + 0.20 * anomaly
            + 0.10 * rarity + 0.10 * event_bonus

Uses only networkx.DiGraph — no dependency on other project modules.
"""

import networkx as nx
import pytest

from src.models.causal_inference import CausalInferenceEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return CausalInferenceEngine(max_lag=4, alpha=0.05, fdr_alpha=0.05)


@pytest.fixture
def simple_graph():
    """A->B->C directed chain with edge weights."""
    g = nx.DiGraph()
    g.add_edge("A", "B", weight=0.9, p_value=0.01)
    g.add_edge("B", "C", weight=0.8, p_value=0.02)
    return g


@pytest.fixture
def diamond_graph():
    """
    Diamond: root -> svc1, root -> svc2, svc1 -> leaf, svc2 -> leaf
    """
    g = nx.DiGraph()
    g.add_edge("root", "svc1", weight=0.9, p_value=0.01)
    g.add_edge("root", "svc2", weight=0.7, p_value=0.03)
    g.add_edge("svc1", "leaf", weight=0.8, p_value=0.02)
    g.add_edge("svc2", "leaf", weight=0.6, p_value=0.04)
    return g


# ===================================================================
# Tests
# ===================================================================


class TestRCAScoring:
    """5-factor RCA scoring formula tests."""

    # 1. Default weights sum to 1.0
    @pytest.mark.unit
    def test_default_weights(self):
        """The 5 default weights must sum to exactly 1.0."""
        w = CausalInferenceEngine.DEFAULT_RCA_WEIGHTS
        assert set(w.keys()) == {
            "pagerank",
            "temporal",
            "anomaly",
            "rarity",
            "event_bonus",
        }
        assert abs(sum(w.values()) - 1.0) < 1e-9, (
            f"Weights sum to {sum(w.values())}, expected 1.0"
        )

    # 2. Backward-compatible call (only graph, no extra args)
    @pytest.mark.unit
    def test_backward_compatible_call(self, engine, simple_graph):
        """rank_root_causes(graph) with no extra args should still work."""
        ranked = engine.rank_root_causes(simple_graph)
        assert isinstance(ranked, list)
        assert len(ranked) == 3  # A, B, C
        # Each item is a dict with the required keys
        for item in ranked:
            assert isinstance(item, dict)
            assert "signal" in item
            assert "rca_score" in item

    # 3. PageRank component normalized to [0, 1]
    @pytest.mark.unit
    def test_pagerank_component(self, engine, simple_graph):
        """All pagerank_score values must be in [0, 1] with max == 1.0."""
        ranked = engine.rank_root_causes(
            simple_graph, anomaly_scores={"A": 0.9, "B": 0.5, "C": 0.1}
        )
        pr_scores = [r["pagerank_score"] for r in ranked]
        for score in pr_scores:
            assert 0.0 <= score <= 1.0, f"pagerank_score {score} out of [0,1]"
        # At least one node should have the max normalized score of 1.0
        assert max(pr_scores) == pytest.approx(1.0, abs=1e-6)

    # 4. Temporal priority: first signal gets highest score
    @pytest.mark.unit
    def test_temporal_priority(self, engine, simple_graph):
        """The first signal in temporal_order should have temporal_priority == 1.0,
        and the last should have 0.0."""
        ranked = engine.rank_root_causes(
            simple_graph,
            anomaly_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            temporal_order=["A", "B", "C"],
        )
        by_signal = {r["signal"]: r for r in ranked}
        assert by_signal["A"]["temporal_priority"] == pytest.approx(1.0)
        assert by_signal["B"]["temporal_priority"] == pytest.approx(0.5)
        assert by_signal["C"]["temporal_priority"] == pytest.approx(0.0)

    # 5. Anomaly score component passes through correctly
    @pytest.mark.unit
    def test_anomaly_score_component(self, engine, simple_graph):
        """anomaly_score in output should match input values."""
        scores = {"A": 0.9, "B": 0.6, "C": 0.3}
        ranked = engine.rank_root_causes(simple_graph, anomaly_scores=scores)
        by_signal = {r["signal"]: r for r in ranked}
        for node, expected in scores.items():
            assert by_signal[node]["anomaly_score"] == pytest.approx(expected)

    # 6. Rarity prior defaults to 0.5 when not provided
    @pytest.mark.unit
    def test_rarity_prior_default(self, engine, simple_graph):
        """When rarity_priors is None, all nodes should get 0.5."""
        ranked = engine.rank_root_causes(simple_graph)
        for item in ranked:
            assert item["rarity_prior"] == pytest.approx(0.5), (
                f"Node {item['signal']} rarity_prior={item['rarity_prior']}, expected 0.5"
            )

    # 7. Event bonus based on log level
    @pytest.mark.unit
    def test_event_bonus_log_level(self, engine, simple_graph):
        """ERROR -> 1.0, WARN -> 0.5, INFO -> 0.0."""
        meta = {
            "A": {"log_level": "ERROR"},
            "B": {"log_level": "WARN"},
            "C": {"log_level": "INFO"},
        }
        ranked = engine.rank_root_causes(
            simple_graph,
            anomaly_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            event_metadata=meta,
        )
        by_signal = {r["signal"]: r for r in ranked}
        assert by_signal["A"]["event_bonus"] == pytest.approx(1.0)
        assert by_signal["B"]["event_bonus"] == pytest.approx(0.5)
        assert by_signal["C"]["event_bonus"] == pytest.approx(0.0)

    # 8. Event bonus based on metric score
    @pytest.mark.unit
    def test_event_bonus_metric(self, engine, simple_graph):
        """metric_score > 0.8 -> 1.0, > 0.5 -> 0.5, <= 0.5 -> 0.0."""
        meta = {
            "A": {"metric_score": 0.95},
            "B": {"metric_score": 0.65},
            "C": {"metric_score": 0.3},
        }
        ranked = engine.rank_root_causes(
            simple_graph,
            anomaly_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            event_metadata=meta,
        )
        by_signal = {r["signal"]: r for r in ranked}
        assert by_signal["A"]["event_bonus"] == pytest.approx(1.0)
        assert by_signal["B"]["event_bonus"] == pytest.approx(0.5)
        assert by_signal["C"]["event_bonus"] == pytest.approx(0.0)

    # 9. Full formula verification with known values
    @pytest.mark.unit
    def test_full_formula(self, engine):
        """Verify exact computation of the 5-factor formula with known inputs."""
        # Build a trivial 2-node graph (A->B)
        g = nx.DiGraph()
        g.add_edge("A", "B", weight=0.9, p_value=0.01)

        ranked = engine.rank_root_causes(
            g,
            anomaly_scores={"A": 0.8, "B": 0.4},
            temporal_order=["A", "B"],
            rarity_priors={"A": 0.7, "B": 0.3},
            event_metadata={
                "A": {"log_level": "CRITICAL"},  # bonus = 1.0
                "B": {"log_level": "DEBUG"},  # bonus = 0.0
            },
        )
        by_signal = {r["signal"]: r for r in ranked}

        # For node A:
        # temporal_priority: first of 2 -> 1.0
        # anomaly_score: 0.8
        # rarity_prior: 0.7
        # event_bonus: 1.0 (CRITICAL)
        # pagerank: normalized, A is the root in A->B
        a = by_signal["A"]
        assert a["temporal_priority"] == pytest.approx(1.0)
        assert a["anomaly_score"] == pytest.approx(0.8)
        assert a["rarity_prior"] == pytest.approx(0.7)
        assert a["event_bonus"] == pytest.approx(1.0)

        # Manually compute expected rca_score for A
        expected_a = (
            0.35 * a["pagerank_score"]
            + 0.25 * 1.0
            + 0.20 * 0.8
            + 0.10 * 0.7
            + 0.10 * 1.0
        )
        assert a["rca_score"] == pytest.approx(expected_a, abs=1e-9)

        # For node B:
        b = by_signal["B"]
        assert b["temporal_priority"] == pytest.approx(0.0)
        assert b["anomaly_score"] == pytest.approx(0.4)
        assert b["rarity_prior"] == pytest.approx(0.3)
        assert b["event_bonus"] == pytest.approx(0.0)

        expected_b = (
            0.35 * b["pagerank_score"]
            + 0.25 * 0.0
            + 0.20 * 0.4
            + 0.10 * 0.3
            + 0.10 * 0.0
        )
        assert b["rca_score"] == pytest.approx(expected_b, abs=1e-9)

        # A should score higher than B
        assert a["rca_score"] > b["rca_score"]

    # 10. Custom weights override
    @pytest.mark.unit
    def test_custom_weights(self, engine, simple_graph):
        """Override default weights and verify the formula uses them."""
        custom_weights = {
            "pagerank": 0.10,
            "temporal": 0.10,
            "anomaly": 0.10,
            "rarity": 0.10,
            "event_bonus": 0.60,
        }
        meta = {"A": {"log_level": "ERROR"}}  # bonus = 1.0

        ranked = engine.rank_root_causes(
            simple_graph,
            anomaly_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            temporal_order=["A", "B", "C"],
            event_metadata=meta,
            weights=custom_weights,
        )
        by_signal = {r["signal"]: r for r in ranked}

        # With event_bonus weight at 0.60, A (bonus=1.0) should dominate
        assert by_signal["A"]["rca_score"] > by_signal["B"]["rca_score"]
        assert by_signal["A"]["rca_score"] > by_signal["C"]["rca_score"]

        # Verify manually: A's rca with custom weights
        a = by_signal["A"]
        expected = (
            0.10 * a["pagerank_score"]
            + 0.10 * a["temporal_priority"]
            + 0.10 * a["anomaly_score"]
            + 0.10 * a["rarity_prior"]
            + 0.60 * a["event_bonus"]
        )
        assert a["rca_score"] == pytest.approx(expected, abs=1e-9)

    # 11. All scores are in [0, 1]
    @pytest.mark.unit
    def test_score_range(self, engine, diamond_graph):
        """All rca_scores and component scores must be in [0, 1]."""
        ranked = engine.rank_root_causes(
            diamond_graph,
            anomaly_scores={"root": 0.95, "svc1": 0.7, "svc2": 0.6, "leaf": 0.3},
            temporal_order=["root", "svc1", "svc2", "leaf"],
            rarity_priors={"root": 0.9, "svc1": 0.5, "svc2": 0.4, "leaf": 0.1},
            event_metadata={
                "root": {"log_level": "CRITICAL"},
                "svc1": {"log_level": "ERROR"},
                "svc2": {"log_level": "WARN"},
                "leaf": {"log_level": "INFO"},
            },
        )
        component_keys = [
            "rca_score",
            "pagerank_score",
            "temporal_priority",
            "anomaly_score",
            "rarity_prior",
            "event_bonus",
        ]
        for item in ranked:
            for key in component_keys:
                val = item[key]
                assert 0.0 <= val <= 1.0, (
                    f"Node {item['signal']}: {key}={val} out of [0,1]"
                )

    # 12. All 5 component scores present in output
    @pytest.mark.unit
    def test_component_scores_in_output(self, engine, simple_graph):
        """Each result dict must contain all 5 component scores plus signal and rca_score."""
        ranked = engine.rank_root_causes(simple_graph)
        required_keys = {
            "signal",
            "rca_score",
            "pagerank_score",
            "temporal_priority",
            "anomaly_score",
            "rarity_prior",
            "event_bonus",
        }
        for item in ranked:
            assert set(item.keys()) == required_keys, (
                f"Missing keys: {required_keys - set(item.keys())}"
            )

    # 13. Edge case: single node graph
    @pytest.mark.unit
    def test_single_node_graph(self, engine):
        """A graph with one node and no edges should return one result."""
        g = nx.DiGraph()
        g.add_node("solo")

        ranked = engine.rank_root_causes(
            g,
            anomaly_scores={"solo": 0.8},
            temporal_order=["solo"],
            rarity_priors={"solo": 0.6},
            event_metadata={"solo": {"log_level": "ERROR"}},
        )
        assert len(ranked) == 1
        r = ranked[0]
        assert r["signal"] == "solo"
        # Single node: pagerank should be 1.0 (only node)
        assert r["pagerank_score"] == pytest.approx(1.0)
        # Single node in temporal_order: gets 1.0
        assert r["temporal_priority"] == pytest.approx(1.0)
        assert r["anomaly_score"] == pytest.approx(0.8)
        assert r["rarity_prior"] == pytest.approx(0.6)
        assert r["event_bonus"] == pytest.approx(1.0)

        # Verify formula
        expected = 0.35 * 1.0 + 0.25 * 1.0 + 0.20 * 0.8 + 0.10 * 0.6 + 0.10 * 1.0
        assert r["rca_score"] == pytest.approx(expected, abs=1e-9)

    # 14. Empty graph returns empty list
    @pytest.mark.unit
    def test_empty_graph(self, engine):
        """An empty graph (no nodes) should return []."""
        g = nx.DiGraph()
        assert engine.rank_root_causes(g) == []

    # 15. CRITICAL log level treated same as ERROR
    @pytest.mark.unit
    def test_event_bonus_critical(self, engine):
        """CRITICAL log level should give event_bonus = 1.0."""
        g = nx.DiGraph()
        g.add_node("db")
        ranked = engine.rank_root_causes(
            g,
            anomaly_scores={"db": 0.5},
            event_metadata={"db": {"log_level": "CRITICAL"}},
        )
        assert ranked[0]["event_bonus"] == pytest.approx(1.0)

    # 16. WARNING alias for WARN
    @pytest.mark.unit
    def test_event_bonus_warning_alias(self, engine):
        """WARNING (full word) should also give event_bonus = 0.5."""
        g = nx.DiGraph()
        g.add_node("api")
        ranked = engine.rank_root_causes(
            g,
            anomaly_scores={"api": 0.5},
            event_metadata={"api": {"log_level": "WARNING"}},
        )
        assert ranked[0]["event_bonus"] == pytest.approx(0.5)

    # 17. Partial weights override (only some keys)
    @pytest.mark.unit
    def test_partial_weights_override(self, engine, simple_graph):
        """Overriding only some weight keys should leave others at defaults."""
        ranked = engine.rank_root_causes(
            simple_graph,
            weights={"pagerank": 0.50},  # only override pagerank
        )
        # The method should still work; other weights keep their defaults
        assert len(ranked) == 3
        for item in ranked:
            assert "rca_score" in item

    # 18. Temporal priority with nodes not in temporal_order
    @pytest.mark.unit
    def test_temporal_priority_missing_node(self, engine, simple_graph):
        """Nodes not in temporal_order should get temporal_priority = 0.0."""
        ranked = engine.rank_root_causes(
            simple_graph,
            anomaly_scores={"A": 0.5, "B": 0.5, "C": 0.5},
            temporal_order=["A"],  # only A in the order
        )
        by_signal = {r["signal"]: r for r in ranked}
        assert by_signal["A"]["temporal_priority"] == pytest.approx(1.0)
        assert by_signal["B"]["temporal_priority"] == pytest.approx(0.0)
        assert by_signal["C"]["temporal_priority"] == pytest.approx(0.0)
