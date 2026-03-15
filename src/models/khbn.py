"""
KHBN — Knowledge-Informed Hierarchical Bayesian Network.

Uses pgmpy to build a Bayesian Network with service topology as structural prior.
Posterior inference re-ranks root cause candidates by incorporating domain knowledge
about which services can actually cause failures in which other services.

Requires: pgmpy, networkx, PyYAML
"""

import os
import warnings
from typing import Optional

import networkx as nx
import yaml

warnings.filterwarnings("ignore")

# Graceful pgmpy import — fall back to NetworkX-based approximation
try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination

    # pgmpy >=1.0 renamed BayesianNetwork → DiscreteBayesianNetwork
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


class KHBNModel:
    """
    Knowledge-Informed Hierarchical Bayesian Network for root cause re-ranking.

    Uses the service dependency topology as a structural prior to:
    1. Filter spurious causal edges that violate known architecture
    2. Build a Bayesian Network with topology-informed CPDs
    3. Run posterior inference to re-rank root cause candidates

    If pgmpy is not installed, falls back to a simplified topology-distance
    + anomaly-score ranking (no actual Bayesian inference).
    """

    def __init__(self, topology_path: str = "config/service_topology.yaml"):
        self.topology_path = topology_path
        self.topology: dict = {}
        self.prior_dag: nx.DiGraph = nx.DiGraph()
        self.bn = None  # pgmpy BayesianNetwork (or None in fallback mode)
        self.node_list: list = []
        self.anomaly_scores: dict = {}
        self._fitted = False

        # Source/metric-to-service mappings (populated from topology)
        self.source_to_service: dict = {}
        self.metric_to_service: dict = {}

        # Load topology if file exists
        if os.path.exists(topology_path):
            self.topology = self.load_topology(topology_path)
            self.prior_dag = self.build_prior_dag()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_topology(self, path: str) -> dict:
        """Parse the service topology YAML and return the raw dict."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Populate lookup mappings
        self.source_to_service = data.get("source_to_service", {})
        self.metric_to_service = data.get("metric_to_service", {})

        return data

    def build_prior_dag(self) -> nx.DiGraph:
        """
        Build a directed graph from the topology dependencies.

        Edge direction: dependency → dependent (cause → effect).
        If service B depends_on [A], then the edge is A → B because a
        failure in A *causes* failures in B.
        """
        dag = nx.DiGraph()

        services = self.topology.get("services", [])
        for svc in services:
            name = svc["name"]
            dag.add_node(name, type=svc.get("type", "unknown"))

        for svc in services:
            name = svc["name"]
            for dep in svc.get("depends_on", []):
                # dep → name: failure in dep causes failure in name
                dag.add_edge(dep, name)

        return dag

    def fit(
        self,
        anomaly_scores: dict,
        causal_edges: list,
    ) -> None:
        """
        Build and fit the Bayesian Network.

        Parameters
        ----------
        anomaly_scores : dict
            {signal_name: float} — unified anomaly scores in [0, 1].
        causal_edges : list
            [(source, target, weight)] — edges from the causal graph.
        """
        self.anomaly_scores = anomaly_scores
        self.node_list = sorted(anomaly_scores.keys())

        if not self.node_list:
            self._fitted = True
            return

        # Filter causal edges by topology
        valid_edges = self.filter_edges_by_topology(causal_edges)

        if PGMPY_AVAILABLE:
            self._fit_pgmpy(valid_edges)
        else:
            self._fit_fallback(valid_edges)

        self._fitted = True

    def posterior_rank(self, evidence: Optional[dict] = None) -> list:
        """
        Run inference and return ranked root cause candidates.

        Parameters
        ----------
        evidence : dict, optional
            {signal_name: 0 or 1} — observed states. If None, uses
            anomaly_scores > 0.5 threshold as evidence.

        Returns
        -------
        list of (signal_name, posterior_probability) sorted descending.
        """
        if not self._fitted or not self.node_list:
            return []

        if PGMPY_AVAILABLE and self.bn is not None:
            return self._posterior_pgmpy(evidence)
        else:
            return self._posterior_fallback(evidence)

    def filter_edges_by_topology(self, causal_edges: list) -> list:
        """
        Remove causal edges that violate the service topology.

        An edge A → B is valid only if:
        1. A and B map to the *same* service, OR
        2. A's service is an ancestor of B's service in the topology DAG.
        """
        if not self.prior_dag.nodes:
            # No topology loaded — keep all edges
            return causal_edges

        valid = []
        for edge in causal_edges:
            source, target = edge[0], edge[1]
            weight = edge[2] if len(edge) > 2 else 1.0

            svc_source = self.map_signal_to_service(source)
            svc_target = self.map_signal_to_service(target)

            if svc_source == svc_target:
                # Same service — always valid
                valid.append((source, target, weight))
            elif (
                svc_source in self.prior_dag.nodes
                and svc_target in self.prior_dag.nodes
            ):
                # Check if svc_source is an ancestor of svc_target
                if nx.has_path(self.prior_dag, svc_source, svc_target):
                    valid.append((source, target, weight))
            else:
                # One or both services not in topology — keep the edge
                # (conservative: don't discard unknown signals)
                valid.append((source, target, weight))

        return valid

    def map_signal_to_service(self, signal_name: str) -> str:
        """
        Map a signal name (log source label or metric name) to a service.

        Resolution order:
        1. Exact match in source_to_service (log source labels)
        2. Prefix match in metric_to_service (metric prefixes)
        3. Fall back to signal_name itself (assume it IS the service name)
        """
        # 1. Exact match on source label
        if signal_name in self.source_to_service:
            return self.source_to_service[signal_name]

        # 2. Prefix match on metric name (longest prefix wins)
        best_match = ""
        best_service = signal_name  # default fallback
        for prefix, service in self.metric_to_service.items():
            if signal_name.startswith(prefix) and len(prefix) > len(best_match):
                best_match = prefix
                best_service = service

        return best_service

    # ------------------------------------------------------------------
    # pgmpy-based implementation
    # ------------------------------------------------------------------

    def _fit_pgmpy(self, valid_edges: list) -> None:
        """Build a pgmpy BayesianNetwork with topology + data edges."""
        # Combine topology prior edges with data-driven edges
        bn_edges = []
        edge_set = set()

        # Add topology prior edges (only for nodes present in anomaly_scores)
        for u, v in self.prior_dag.edges():
            # Map service names to signal names that are in our node list
            signals_u = [
                s for s in self.node_list if self.map_signal_to_service(s) == u
            ]
            signals_v = [
                s for s in self.node_list if self.map_signal_to_service(s) == v
            ]
            for su in signals_u:
                for sv in signals_v:
                    if su != sv and (su, sv) not in edge_set:
                        bn_edges.append((su, sv))
                        edge_set.add((su, sv))

        # Add valid data-driven edges
        for source, target, weight in valid_edges:
            if source in self.node_list and target in self.node_list:
                if source != target and (source, target) not in edge_set:
                    bn_edges.append((source, target))
                    edge_set.add((source, target))

        # Build DAG — break any cycles
        temp_dag = nx.DiGraph()
        temp_dag.add_nodes_from(self.node_list)
        for u, v in bn_edges:
            temp_dag.add_edge(u, v, weight=1.0)
        self._break_cycles(temp_dag)
        bn_edges = list(temp_dag.edges())

        if not bn_edges:
            # No edges — create isolated nodes BN
            self.bn = BayesianNetwork()
            for node in self.node_list:
                self.bn.add_node(node)
                # Marginal CPD: P(node=1) = anomaly_score
                score = min(max(self.anomaly_scores.get(node, 0.1), 0.01), 0.99)
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1.0 - score], [score]],
                )
                self.bn.add_cpds(cpd)
            self.bn.check_model()
            return

        # Create BayesianNetwork
        self.bn = BayesianNetwork(bn_edges)
        # Ensure all nodes are present
        for node in self.node_list:
            if node not in self.bn.nodes():
                self.bn.add_node(node)

        # Build CPDs
        for node in self.node_list:
            parents = sorted(self.bn.get_parents(node))
            score = min(max(self.anomaly_scores.get(node, 0.1), 0.01), 0.99)

            if not parents:
                # Root node — marginal CPD
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1.0 - score], [score]],
                )
            else:
                # Child node — conditional CPD
                n_parent_configs = 2 ** len(parents)
                # P(child=1 | parents): higher when more parents are anomalous
                values_0 = []  # P(node=0 | parent_config)
                values_1 = []  # P(node=1 | parent_config)

                for config_idx in range(n_parent_configs):
                    # Count how many parents are in state 1 (anomalous)
                    n_anomalous = bin(config_idx).count("1")
                    frac_anomalous = n_anomalous / len(parents)

                    # P(child=anomalous) increases with parent anomalies
                    # Base rate from own anomaly score, boosted by parent anomalies
                    p_anomalous = min(
                        score * (0.3 + 0.7 * frac_anomalous),
                        0.99,
                    )
                    p_anomalous = max(p_anomalous, 0.01)

                    values_0.append(1.0 - p_anomalous)
                    values_1.append(p_anomalous)

                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[values_0, values_1],
                    evidence=parents,
                    evidence_card=[2] * len(parents),
                )

            self.bn.add_cpds(cpd)

        try:
            self.bn.check_model()
        except Exception:
            # If model check fails, fall back to simpler structure
            self.bn = None

    def _posterior_pgmpy(self, evidence: Optional[dict] = None) -> list:
        """Run Variable Elimination on the pgmpy BN."""
        if self.bn is None:
            return self._posterior_fallback(evidence)

        # Build evidence dict: observe nodes with high anomaly scores
        if evidence is None:
            evidence = {}
            for node in self.node_list:
                score = self.anomaly_scores.get(node, 0.0)
                if score > 0.5:
                    evidence[node] = 1

        # Query nodes = those NOT in evidence
        query_nodes = [n for n in self.node_list if n not in evidence]

        if not query_nodes:
            # All nodes are evidence — rank by anomaly score directly
            ranked = sorted(
                self.anomaly_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked

        try:
            infer = VariableElimination(self.bn)
            results = []

            for node in query_nodes:
                phi = infer.query([node], evidence=evidence)
                # P(node=1) = posterior probability of being anomalous
                p_anomalous = float(phi.values[1])
                results.append((node, p_anomalous))

            # Add evidence nodes with their scores
            for node, state in evidence.items():
                results.append((node, self.anomaly_scores.get(node, float(state))))

            # Sort descending by posterior probability
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception:
            return self._posterior_fallback(evidence)

    # ------------------------------------------------------------------
    # Fallback (no pgmpy) — topology distance + anomaly score
    # ------------------------------------------------------------------

    def _fit_fallback(self, valid_edges: list) -> None:
        """Store edges for fallback ranking (no pgmpy needed)."""
        self._fallback_edges = valid_edges

    def _posterior_fallback(self, evidence: Optional[dict] = None) -> list:
        """
        Simplified ranking without Bayesian inference.

        Score = anomaly_score * topology_centrality_boost.

        Nodes closer to topology roots (fewer ancestors) get a boost
        because they are more likely to be root causes.
        """
        if not self.node_list:
            return []

        results = []
        for node in self.node_list:
            score = self.anomaly_scores.get(node, 0.0)
            service = self.map_signal_to_service(node)

            # Topology boost: root services (no incoming edges) get a boost
            if service in self.prior_dag.nodes:
                # Depth = shortest path from any root to this service
                ancestors = nx.ancestors(self.prior_dag, service)
                depth = len(ancestors)
                # Boost factor: 1.0 for roots, decreasing for deeper nodes
                topo_boost = 1.0 / (1.0 + 0.3 * depth)
            else:
                topo_boost = 0.8  # Unknown service gets slight penalty

            combined = score * topo_boost
            results.append((node, combined))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _break_cycles(graph: nx.DiGraph) -> None:
        """Remove the weakest edge in every cycle until the graph is a DAG."""
        while True:
            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                break

            weakest_edge = None
            weakest_weight = float("inf")
            for u, v, _ in cycle:
                w = graph[u][v].get("weight", 1.0)
                if w < weakest_weight:
                    weakest_weight = w
                    weakest_edge = (u, v)

            if weakest_edge:
                graph.remove_edge(*weakest_edge)
