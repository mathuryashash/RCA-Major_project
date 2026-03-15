import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")

# Graceful causal-learn import (FR-24)
try:
    from causallearn.search.ConstraintBased.PC import pc as pc_algorithm

    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False


class CausalInferenceEngine:
    """
    Module 4: Causal Inference Engine
    Implements Granger Causality, PC Algorithm, and Personalized PageRank
    for Root Cause Ranking.

    Enhancements (FR-23, FR-24):
    - Benjamini-Hochberg FDR correction for Granger p-values
    - PC Algorithm integration via causal-learn
    - Majority-vote edge combination (union of Granger + PC)
    - Automatic cycle breaking (weakest-edge removal)
    """

    def __init__(self, max_lag=6, alpha=0.05, fdr_alpha=0.05):
        self.max_lag = max_lag
        self.alpha = alpha
        self.fdr_alpha = fdr_alpha

    # ------------------------------------------------------------------
    # Public API (backward-compatible signatures)
    # ------------------------------------------------------------------

    def build_causal_graph(
        self, signals_df: pd.DataFrame, use_pc: bool = True, use_fdr: bool = True
    ) -> nx.DiGraph:
        """
        Build a causal directed graph from time-series signals.

        Parameters
        ----------
        signals_df : DataFrame
            Each column is a normalised anomaly-score time series.
        use_pc : bool
            If True (default) and causal-learn is installed, run the PC
            algorithm and combine with Granger via majority vote.
        use_fdr : bool
            If True (default), apply Benjamini-Hochberg FDR correction
            to Granger p-values instead of raw alpha thresholding.

        Returns
        -------
        nx.DiGraph
            Causal graph with 'weight' and 'p_value' edge attributes.
        """
        nodes = list(signals_df.columns)
        if len(nodes) < 2 or len(signals_df) == 0:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            return graph

        # --- Step 1: Granger graph (with optional FDR) -----------------
        granger_graph = self._build_granger_graph(signals_df, use_fdr=use_fdr)

        # --- Step 2: PC algorithm (optional) ---------------------------
        pc_graph = None
        if use_pc and CAUSAL_LEARN_AVAILABLE:
            pc_graph = self._build_pc_graph(signals_df)

        # --- Step 3: Combine -------------------------------------------
        if pc_graph is not None:
            combined = self._majority_vote_combine(granger_graph, pc_graph, nodes)
        else:
            combined = granger_graph

        # --- Step 4: Break cycles --------------------------------------
        self._break_cycles(combined)

        return combined

    # Default weights for the 5-factor RCA scoring formula (PRD line 454)
    DEFAULT_RCA_WEIGHTS = {
        "pagerank": 0.35,
        "temporal": 0.25,
        "anomaly": 0.20,
        "rarity": 0.10,
        "event_bonus": 0.10,
    }

    def rank_root_causes(
        self,
        graph: nx.DiGraph,
        anomaly_scores: dict | None = None,
        temporal_order: list | None = None,
        rarity_priors: dict | None = None,
        event_metadata: dict | None = None,
        weights: dict | None = None,
    ) -> list:
        """
        Rank nodes using the 5-factor RCA scoring formula.

        rca_score = w_pagerank * pagerank_score
                  + w_temporal * temporal_priority
                  + w_anomaly * anomaly_score
                  + w_rarity * rarity_prior
                  + w_event * event_bonus

        Parameters
        ----------
        graph : nx.DiGraph
            Causal graph (from build_causal_graph).
        anomaly_scores : dict, optional
            {node_name: float} anomaly score in [0,1]. Defaults to 0.5 for
            all nodes (backward-compatible).
        temporal_order : list, optional
            Signal names in order of first anomaly detection (earliest first).
            Defaults to alphabetical order of node names.
        rarity_priors : dict, optional
            {node_name: float} rarity prior in [0,1]. Defaults to 0.5.
        event_metadata : dict, optional
            {node_name: dict} with keys like 'log_level', 'metric_score'.
            Used to compute the event_bonus factor.
        weights : dict, optional
            Override default weights. Keys: 'pagerank', 'temporal',
            'anomaly', 'rarity', 'event_bonus'.

        Returns
        -------
        list of dicts sorted descending by rca_score. Each dict contains::

            {
                'signal': str,
                'rca_score': float,
                'pagerank_score': float,
                'temporal_priority': float,
                'anomaly_score': float,
                'rarity_prior': float,
                'event_bonus': float,
            }
        """
        if len(graph.nodes) == 0:
            return []

        nodes = list(graph.nodes)

        # --- Resolve defaults (backward compatibility) ---------------------
        if anomaly_scores is None:
            anomaly_scores = {node: 0.5 for node in nodes}

        if temporal_order is None:
            temporal_order = sorted(nodes)

        if rarity_priors is None:
            rarity_priors = {}

        if event_metadata is None:
            event_metadata = {}

        w = dict(self.DEFAULT_RCA_WEIGHTS)
        if weights is not None:
            w.update(weights)

        # --- Factor 1: PageRank (normalized to [0,1]) ----------------------
        pagerank_scores = self._compute_pagerank(graph, anomaly_scores)

        # --- Factor 2: Temporal priority -----------------------------------
        temporal_scores = self._compute_temporal_priority(nodes, temporal_order)

        # --- Factor 3: Anomaly scores (pass-through) -----------------------
        # Clamp to [0,1] for safety
        anomaly_map = {
            node: max(0.0, min(1.0, anomaly_scores.get(node, 0.5))) for node in nodes
        }

        # --- Factor 4: Rarity prior (default 0.5) -------------------------
        rarity_map = {
            node: max(0.0, min(1.0, rarity_priors.get(node, 0.5))) for node in nodes
        }

        # --- Factor 5: Event bonus -----------------------------------------
        event_bonus_map = self._compute_event_bonus(
            nodes, event_metadata, anomaly_scores
        )

        # --- Combine using weighted formula --------------------------------
        results = []
        for node in nodes:
            pr = pagerank_scores.get(node, 0.0)
            tp = temporal_scores.get(node, 0.0)
            an = anomaly_map[node]
            ra = rarity_map[node]
            eb = event_bonus_map[node]

            rca_score = (
                w["pagerank"] * pr
                + w["temporal"] * tp
                + w["anomaly"] * an
                + w["rarity"] * ra
                + w["event_bonus"] * eb
            )

            results.append(
                {
                    "signal": node,
                    "rca_score": rca_score,
                    "pagerank_score": pr,
                    "temporal_priority": tp,
                    "anomaly_score": an,
                    "rarity_prior": ra,
                    "event_bonus": eb,
                }
            )

        results.sort(key=lambda x: x["rca_score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # RCA factor computation helpers
    # ------------------------------------------------------------------

    def _compute_pagerank(self, graph: nx.DiGraph, anomaly_scores: dict) -> dict:
        """Compute PageRank and normalize scores to [0, 1]."""
        personalization = {
            node: anomaly_scores.get(node, 0.0) + 0.01 for node in graph.nodes
        }
        total = sum(personalization.values())
        personalization = {k: v / total for k, v in personalization.items()}

        try:
            scores = nx.pagerank(
                graph.reverse(),
                personalization=personalization,
                weight="weight",
            )
        except Exception:
            # Fallback: uniform scores
            n = len(graph.nodes)
            scores = {node: 1.0 / n if n > 0 else 0.0 for node in graph.nodes}

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    @staticmethod
    def _compute_temporal_priority(nodes: list, temporal_order: list) -> dict:
        """
        Compute temporal priority for each node.

        The first anomalous signal gets 1.0, the last gets 0.0.
        Formula: 1.0 - (rank / total_anomalous_signals)
        where rank is 0-based temporal order.

        Nodes not in temporal_order get 0.0.
        """
        # Filter to only nodes present in the graph
        ordered = [n for n in temporal_order if n in nodes]
        total = len(ordered)

        if total <= 1:
            # Single signal or empty: everyone in the list gets 1.0
            return {n: (1.0 if n in ordered else 0.0) for n in nodes}

        result = {}
        for node in nodes:
            if node in ordered:
                rank = ordered.index(node)
                result[node] = 1.0 - (rank / (total - 1))
            else:
                result[node] = 0.0
        return result

    @staticmethod
    def _compute_event_bonus(
        nodes: list, event_metadata: dict, anomaly_scores: dict
    ) -> dict:
        """
        Compute event bonus for each node.

        For logs (has 'log_level' key):
            ERROR/CRITICAL -> 1.0, WARN/WARNING -> 0.5, else -> 0.0
        For metrics (has 'metric_score' key or fallback to anomaly_score):
            score > 0.8 -> 1.0, > 0.5 -> 0.5, else -> 0.0
        No metadata -> 0.0
        """
        result = {}
        for node in nodes:
            meta = event_metadata.get(node, {})
            if not meta:
                result[node] = 0.0
                continue

            log_level = meta.get("log_level")
            if log_level is not None:
                level = log_level.upper()
                if level in ("ERROR", "CRITICAL"):
                    result[node] = 1.0
                elif level in ("WARN", "WARNING"):
                    result[node] = 0.5
                else:
                    result[node] = 0.0
                continue

            # Metric-based bonus
            metric_score = meta.get("metric_score")
            if metric_score is None:
                metric_score = anomaly_scores.get(node, 0.0)

            if metric_score > 0.8:
                result[node] = 1.0
            elif metric_score > 0.5:
                result[node] = 0.5
            else:
                result[node] = 0.0

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_granger_graph(
        self, signals_df: pd.DataFrame, use_fdr: bool = True
    ) -> nx.DiGraph:
        """Pairwise Granger causality with optional BH-FDR correction."""
        nodes = list(signals_df.columns)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        # Collect all pairwise test results first
        candidate_edges = []  # list of (source, target, min_p)

        for target in nodes:
            for source in nodes:
                if target == source:
                    continue
                try:
                    data = signals_df[[target, source]].values
                    results = grangercausalitytests(
                        data, maxlag=self.max_lag, verbose=False
                    )
                    p_values = [
                        round(results[i + 1][0]["ssr_ftest"][1], 4)
                        for i in range(self.max_lag)
                    ]
                    min_p = min(p_values)
                    candidate_edges.append((source, target, min_p))
                except Exception:
                    continue

        if not candidate_edges:
            return graph

        if use_fdr:
            # FR-23: Benjamini-Hochberg FDR correction
            raw_pvals = np.array([e[2] for e in candidate_edges])
            reject, qvals, _, _ = multipletests(
                raw_pvals, alpha=self.fdr_alpha, method="fdr_bh"
            )
            for idx, (source, target, min_p) in enumerate(candidate_edges):
                if reject[idx]:
                    graph.add_edge(
                        source,
                        target,
                        weight=1.0 - qvals[idx],
                        p_value=qvals[idx],
                    )
        else:
            # Legacy behaviour: raw alpha thresholding
            for source, target, min_p in candidate_edges:
                if min_p < self.alpha:
                    graph.add_edge(
                        source,
                        target,
                        weight=1.0 - min_p,
                        p_value=min_p,
                    )

        return graph

    def _build_pc_graph(self, signals_df: pd.DataFrame) -> nx.DiGraph:
        """Run the PC algorithm and return a DiGraph of discovered edges."""
        nodes = list(signals_df.columns)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        try:
            data_matrix = signals_df.values
            cg = pc_algorithm(
                data_matrix,
                alpha=self.fdr_alpha,
                indep_test="fisherz",
                show_progress=False,
            )
            adj = cg.G.graph  # numpy adjacency matrix
            # causal-learn adjacency encoding:
            #   adj[i,j] = -1 and adj[j,i] = 1  => i -> j
            #   adj[i,j] = -1 and adj[j,i] = -1 => i - j (undirected)
            #   adj[i,j] =  1 and adj[j,i] =  1  => i <-> j (bidirected)
            n = len(nodes)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if adj[i, j] == -1 and adj[j, i] == 1:
                        # directed edge i -> j
                        graph.add_edge(
                            nodes[i],
                            nodes[j],
                            weight=1.0,
                            p_value=0.0,
                        )
                    elif adj[i, j] == -1 and adj[j, i] == -1:
                        # undirected edge: add both directions
                        graph.add_edge(
                            nodes[i],
                            nodes[j],
                            weight=1.0,
                            p_value=0.0,
                        )
        except Exception:
            pass

        return graph

    def _majority_vote_combine(
        self, granger_graph: nx.DiGraph, pc_graph: nx.DiGraph, nodes: list
    ) -> nx.DiGraph:
        """
        Union-based majority-vote combination of Granger and PC graphs.
        Edge weight = average of Granger weight and PC confidence.
        """
        combined = nx.DiGraph()
        combined.add_nodes_from(nodes)

        all_edges = set(granger_graph.edges()) | set(pc_graph.edges())

        for u, v in all_edges:
            g_data = granger_graph.get_edge_data(u, v)
            p_data = pc_graph.get_edge_data(u, v)

            g_weight = g_data["weight"] if g_data else 0.0
            p_weight = p_data["weight"] if p_data else 0.0

            # Confidence: 1.0 if present, 0.0 if absent — averaged
            g_conf = 1.0 if g_data else 0.0
            p_conf = 1.0 if p_data else 0.0
            avg_weight = (g_weight + p_weight) / (g_conf + p_conf)

            # p_value: take Granger's if available, else 0.0
            p_val = g_data["p_value"] if g_data else 0.0

            combined.add_edge(u, v, weight=avg_weight, p_value=p_val)

        return combined

    @staticmethod
    def _break_cycles(graph: nx.DiGraph) -> None:
        """Remove the weakest edge in every cycle until the graph is a DAG."""
        while True:
            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                break

            # Find the weakest edge in this cycle
            weakest_edge = None
            weakest_weight = float("inf")
            for u, v, _ in cycle:
                w = graph[u][v].get("weight", 1.0)
                if w < weakest_weight:
                    weakest_weight = w
                    weakest_edge = (u, v)

            if weakest_edge:
                graph.remove_edge(*weakest_edge)


if __name__ == "__main__":
    # Test with dummy data
    engine = CausalInferenceEngine()

    # Simulate 3 signals: Root Cause (A) -> Intermediate (B) -> Symptom (C)
    t = np.linspace(0, 100, 100)
    a = np.sin(t) + np.random.normal(0, 0.1, 100)
    b = np.roll(a, 5) + np.random.normal(0, 0.1, 100)  # B follows A
    c = np.roll(b, 5) + np.random.normal(0, 0.1, 100)  # C follows B

    df = pd.DataFrame({"A": a, "B": b, "C": c})

    print("Building causal graph (enhanced mode)...")
    g = engine.build_causal_graph(df)
    print(f"Edges detected: {g.edges(data=True)}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(g)}")

    print("\nBuilding causal graph (legacy mode)...")
    g_legacy = engine.build_causal_graph(df, use_pc=False, use_fdr=False)
    print(f"Edges detected: {g_legacy.edges(data=True)}")

    print("\nRanking root causes (5-factor formula)...")
    ranks = engine.rank_root_causes(
        g,
        anomaly_scores={"A": 0.9, "B": 0.8, "C": 0.8},
        temporal_order=["A", "B", "C"],
        event_metadata={"A": {"log_level": "ERROR"}, "B": {"log_level": "WARN"}},
    )
    for r in ranks:
        print(
            f"Node: {r['signal']} | RCA: {r['rca_score']:.4f} | "
            f"PR: {r['pagerank_score']:.2f} TP: {r['temporal_priority']:.2f} "
            f"AN: {r['anomaly_score']:.2f} RA: {r['rarity_prior']:.2f} "
            f"EB: {r['event_bonus']:.2f}"
        )
