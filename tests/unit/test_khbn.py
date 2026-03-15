"""
Unit tests for src.models.khbn.KHBNModel

All tests work WITHOUT pgmpy installed — the KHBN module gracefully
degrades to a NetworkX-based fallback.
"""

import os
import sys
import tempfile
import textwrap
from unittest.mock import patch

import networkx as nx
import pytest
import yaml

from src.models.khbn import KHBNModel, PGMPY_AVAILABLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOPOLOGY_YAML = textwrap.dedent("""\
    services:
      - name: load_balancer
        type: infrastructure
        depends_on: []
      - name: web_server
        type: application
        depends_on: [load_balancer]
      - name: app_server
        type: application
        depends_on: [web_server]
      - name: database
        type: database
        depends_on: [app_server]
      - name: cache
        type: infrastructure
        depends_on: [app_server]
      - name: message_queue
        type: infrastructure
        depends_on: [app_server]
      - name: storage
        type: infrastructure
        depends_on: [app_server]
      - name: dns
        type: infrastructure
        depends_on: []
      - name: network
        type: infrastructure
        depends_on: []
      - name: os_kernel
        type: system
        depends_on: []

    source_to_service:
      application: app_server
      database: database
      system: os_kernel

    metric_to_service:
      cpu_: os_kernel
      memory_: os_kernel
      disk_: storage
      network_: network
      query_latency: database
      connections: database
      cache_hit: cache
      http_: web_server
      request_: app_server
      queue_: message_queue
""")


@pytest.fixture
def topology_file(tmp_path):
    """Write the topology YAML to a temp file and return its path."""
    path = tmp_path / "service_topology.yaml"
    path.write_text(TOPOLOGY_YAML, encoding="utf-8")
    return str(path)


@pytest.fixture
def model(topology_file):
    """A KHBNModel instance loaded with the test topology."""
    return KHBNModel(topology_path=topology_file)


@pytest.fixture
def sample_anomaly_scores():
    """Anomaly scores matching the source/metric naming conventions."""
    return {
        "application": 0.4,
        "database": 0.9,
        "system": 0.2,
        "cpu_usage": 0.3,
        "memory_used": 0.5,
        "query_latency_p99": 0.85,
        "connections_active": 0.7,
    }


@pytest.fixture
def sample_causal_edges():
    """Causal edges — some valid by topology, some not."""
    return [
        # Valid: database → application (database depends on app_server,
        # but app_server is ancestor of database, so app→db is valid;
        # database → application maps to database → app_server — REVERSE direction)
        ("application", "database", 0.8),
        # Valid: same service (os_kernel)
        ("cpu_usage", "system", 0.6),
        # Invalid: database → os_kernel (no topology path)
        ("database", "system", 0.5),
    ]


# ===================================================================
# Tests
# ===================================================================


class TestKHBNModel:
    # 1. test_load_topology
    @pytest.mark.unit
    def test_load_topology(self, model):
        """load_topology should parse YAML and return a dict with services."""
        assert isinstance(model.topology, dict)
        assert "services" in model.topology
        assert len(model.topology["services"]) == 10

        service_names = {s["name"] for s in model.topology["services"]}
        expected = {
            "load_balancer",
            "web_server",
            "app_server",
            "database",
            "cache",
            "message_queue",
            "storage",
            "dns",
            "network",
            "os_kernel",
        }
        assert service_names == expected

    # 2. test_build_prior_dag
    @pytest.mark.unit
    def test_build_prior_dag(self, model):
        """DAG should have correct nodes and edges from topology dependencies."""
        dag = model.prior_dag
        assert isinstance(dag, nx.DiGraph)

        # 10 services = 10 nodes
        assert len(dag.nodes) == 10

        # Verify specific edges (dependency → dependent)
        assert dag.has_edge("load_balancer", "web_server")
        assert dag.has_edge("web_server", "app_server")
        assert dag.has_edge("app_server", "database")
        assert dag.has_edge("app_server", "cache")
        assert dag.has_edge("app_server", "message_queue")
        assert dag.has_edge("app_server", "storage")

        # Root services should have no incoming edges
        for root in ["load_balancer", "dns", "network", "os_kernel"]:
            assert dag.in_degree(root) == 0, f"{root} should be a root node"

        # DAG should be acyclic
        assert nx.is_directed_acyclic_graph(dag)

    # 3. test_map_signal_to_service
    @pytest.mark.unit
    def test_map_signal_to_service(self, model):
        """Signals should map to correct services via source labels and metric prefixes."""
        # Source label mappings
        assert model.map_signal_to_service("application") == "app_server"
        assert model.map_signal_to_service("database") == "database"
        assert model.map_signal_to_service("system") == "os_kernel"

        # Metric prefix mappings
        assert model.map_signal_to_service("cpu_usage") == "os_kernel"
        assert model.map_signal_to_service("memory_used") == "os_kernel"
        assert model.map_signal_to_service("disk_io_read") == "storage"
        assert model.map_signal_to_service("network_latency") == "network"
        assert model.map_signal_to_service("query_latency_p99") == "database"
        assert model.map_signal_to_service("connections_active") == "database"
        assert model.map_signal_to_service("cache_hit_ratio") == "cache"
        assert model.map_signal_to_service("http_errors") == "web_server"
        assert model.map_signal_to_service("request_rate") == "app_server"
        assert model.map_signal_to_service("queue_depth") == "message_queue"

        # Unknown signal falls back to itself
        assert model.map_signal_to_service("unknown_metric") == "unknown_metric"

    # 4. test_filter_edges_valid
    @pytest.mark.unit
    def test_filter_edges_valid(self, model):
        """Edges consistent with topology should be kept."""
        # app_server → database: app_server is ancestor of database ✓
        edges = [("application", "database", 0.8)]
        filtered = model.filter_edges_by_topology(edges)
        assert len(filtered) == 1
        assert filtered[0][0] == "application"
        assert filtered[0][1] == "database"

    # 5. test_filter_edges_invalid
    @pytest.mark.unit
    def test_filter_edges_invalid(self, model):
        """Edges violating topology should be removed."""
        # database → os_kernel: database is NOT an ancestor of os_kernel
        # database maps to 'database' service, system maps to 'os_kernel' service
        # No path from database to os_kernel in the topology
        edges = [("database", "system", 0.5)]
        filtered = model.filter_edges_by_topology(edges)
        assert len(filtered) == 0

    # 6. test_fit_basic
    @pytest.mark.unit
    def test_fit_basic(self, model, sample_anomaly_scores, sample_causal_edges):
        """fit() should complete without errors."""
        model.fit(sample_anomaly_scores, sample_causal_edges)
        assert model._fitted is True

    # 7. test_posterior_rank
    @pytest.mark.unit
    def test_posterior_rank(self, model, sample_anomaly_scores, sample_causal_edges):
        """posterior_rank() should return a sorted list of (signal, probability) tuples."""
        model.fit(sample_anomaly_scores, sample_causal_edges)
        ranked = model.posterior_rank()

        assert isinstance(ranked, list)
        assert len(ranked) > 0

        # Each element is a (signal_name, probability) tuple
        for item in ranked:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], (int, float))

        # Should be sorted descending by probability
        probs = [p for _, p in ranked]
        assert probs == sorted(probs, reverse=True), (
            f"Ranking not sorted descending: {ranked}"
        )

    # 8. test_posterior_rank_topology_influence
    @pytest.mark.unit
    def test_posterior_rank_topology_influence(self, model):
        """Topology should suppress unlikely root causes and boost likely ones.

        When two signals have similar anomaly scores but one is a root service
        (os_kernel) and the other is a leaf (database), the root service should
        rank higher because it is more likely to be a root cause.
        """
        anomaly_scores = {
            "system": 0.8,  # os_kernel — root service
            "database": 0.8,  # database — leaf service (depends on app_server)
        }
        # No causal edges — pure topology influence
        model.fit(anomaly_scores, [])
        ranked = model.posterior_rank()

        assert len(ranked) == 2
        # In fallback mode, root service (os_kernel, depth=0) gets boost 1.0
        # while database (depth=3) gets boost 1/(1+0.9) ≈ 0.526
        # So "system" should rank higher than "database"
        names = [name for name, _ in ranked]
        assert names[0] == "system", (
            f"Expected root service 'system' to rank first, got: {ranked}"
        )

    # 9. test_empty_evidence
    @pytest.mark.unit
    def test_empty_evidence(self, model):
        """Handles no anomalies (all scores 0) gracefully."""
        anomaly_scores = {
            "application": 0.0,
            "database": 0.0,
            "system": 0.0,
        }
        model.fit(anomaly_scores, [])
        ranked = model.posterior_rank()

        assert isinstance(ranked, list)
        assert len(ranked) == 3
        # All probabilities should be 0 or near-zero
        for _, prob in ranked:
            assert prob <= 0.1, f"Expected near-zero probability, got {prob}"

    # 10. test_fallback_without_pgmpy
    @pytest.mark.unit
    def test_fallback_without_pgmpy(self, topology_file):
        """Verify the model works in degraded mode when pgmpy is unavailable."""
        # Patch PGMPY_AVAILABLE to False to force fallback mode
        with patch("src.models.khbn.PGMPY_AVAILABLE", False):
            m = KHBNModel(topology_path=topology_file)
            scores = {"application": 0.7, "database": 0.9, "system": 0.3}
            m.fit(scores, [("application", "database", 0.8)])
            ranked = m.posterior_rank()

            assert isinstance(ranked, list)
            assert len(ranked) == 3

            # Should be sorted descending
            probs = [p for _, p in ranked]
            assert probs == sorted(probs, reverse=True)

            # database has highest anomaly score but is a deep node;
            # in fallback: database score=0.9, depth=3 → boost=1/(1+0.9)≈0.526
            # application score=0.7, depth=2 → boost=1/(1+0.6)≈0.625
            # Both get topology penalty; verify all results are present
            names = {name for name, _ in ranked}
            assert names == {"application", "database", "system"}

    # 11. test_cycle_in_data_edges
    @pytest.mark.unit
    def test_cycle_in_data_edges(self, model):
        """Cycles in causal edges should be handled (broken) without errors."""
        anomaly_scores = {
            "application": 0.6,
            "database": 0.8,
            "query_latency_p99": 0.7,
        }
        # Cycle: application → database → query_latency_p99 → application
        # (all map to app_server/database services)
        cyclic_edges = [
            ("application", "database", 0.8),
            ("database", "query_latency_p99", 0.6),
            ("query_latency_p99", "application", 0.4),
        ]
        # Should not raise
        model.fit(anomaly_scores, cyclic_edges)
        ranked = model.posterior_rank()

        assert isinstance(ranked, list)
        assert len(ranked) == 3

    # 12. test_load_real_topology_file
    @pytest.mark.unit
    def test_load_real_topology_file(self):
        """Load the actual config/service_topology.yaml if it exists."""
        real_path = os.path.join("config", "service_topology.yaml")
        if not os.path.exists(real_path):
            pytest.skip("config/service_topology.yaml not found")

        m = KHBNModel(topology_path=real_path)
        assert len(m.topology["services"]) == 10
        assert len(m.prior_dag.nodes) == 10

    # 13. test_no_topology_file
    @pytest.mark.unit
    def test_no_topology_file(self):
        """Model should instantiate even with a missing topology file."""
        m = KHBNModel(topology_path="/nonexistent/path.yaml")
        assert m.topology == {}
        assert len(m.prior_dag.nodes) == 0

        # fit + rank should still work
        m.fit({"sig_a": 0.8, "sig_b": 0.3}, [("sig_a", "sig_b", 0.5)])
        ranked = m.posterior_rank()
        assert len(ranked) == 2

    # 14. test_filter_edges_same_service
    @pytest.mark.unit
    def test_filter_edges_same_service(self, model):
        """Edges between signals in the same service should always be kept."""
        # cpu_usage and system both map to os_kernel
        edges = [("cpu_usage", "system", 0.7)]
        filtered = model.filter_edges_by_topology(edges)
        assert len(filtered) == 1

    # 15. test_filter_edges_unknown_signal
    @pytest.mark.unit
    def test_filter_edges_unknown_signal(self, model):
        """Edges involving unknown signals should be kept (conservative)."""
        edges = [("unknown_x", "unknown_y", 0.5)]
        filtered = model.filter_edges_by_topology(edges)
        assert len(filtered) == 1

    # 16. test_prior_dag_is_acyclic
    @pytest.mark.unit
    def test_prior_dag_is_acyclic(self, model):
        """The topology prior DAG must be acyclic."""
        assert nx.is_directed_acyclic_graph(model.prior_dag)

    # 17. test_fit_with_empty_scores
    @pytest.mark.unit
    def test_fit_with_empty_scores(self, model):
        """fit() with empty anomaly scores should not crash."""
        model.fit({}, [])
        ranked = model.posterior_rank()
        assert ranked == []
