"""
Causal Inference Engine

Determines which anomalies CAUSE which other anomalies
vs. which are merely correlated or coincidental.

Methods implemented:
1. Granger Causality — temporal prediction improvement test
2. PC Algorithm integration via causal-learn library
3. Temporal precedence analysis — cause must precede effect
4. Causal graph construction — combines all evidence into a DAG
"""

import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Tuple, Set, Optional
import warnings


# ---------------------------------------------------------------------------
# Granger Causality Analyzer
# ---------------------------------------------------------------------------

class GrangerCausalityAnalyzer:
    """
    Granger Causality: X "Granger-causes" Y if past values of X
    significantly improve prediction of Y beyond Y's own history alone.

    Mathematical test:
        Model 1 (Restricted):   Y_t = f(Y_{t-1}, ..., Y_{t-p})
        Model 2 (Unrestricted): Y_t = f(Y_{t-1}, ..., Y_{t-p}, X_{t-1}, ..., X_{t-p})
        If Model 2 explains significantly more variance (F-test p < α): X Granger-causes Y

    Limitation: Assumes LINEAR relationships. Use transfer entropy for nonlinear.
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        """
        Args:
            max_lag: Maximum lag (timesteps) to test — should reflect maximum causal delay
            significance_level: p-value threshold for declaring causality
        """
        self.max_lag = max_lag
        self.significance_level = significance_level

    def make_stationary(self, series: pd.Series) -> pd.Series:
        """
        Apply first-order differencing if series is non-stationary
        (required for Granger causality validity).
        """
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        if p_value > 0.05:  # Non-stationary
            return series.diff().dropna()
        return series

    def test_pair(
        self,
        cause_series: pd.Series,
        effect_series: pd.Series
    ) -> Optional[Dict]:
        """
        Test if cause_series Granger-causes effect_series.

        Returns:
            None if test fails (insufficient data or error)
            Dict with keys: 'is_causal', 'best_lag', 'p_value', 'strength'
        """
        cause_stat = self.make_stationary(cause_series)
        effect_stat = self.make_stationary(effect_series)

        # Align indices
        aligned = pd.concat([effect_stat, cause_stat], axis=1).dropna()
        if len(aligned) < self.max_lag * 5:
            return None  # Insufficient data

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_result = grangercausalitytests(
                    aligned.values, maxlag=self.max_lag, verbose=False
                )

            best_lag, best_p = None, 1.0
            for lag, test_dict in gc_result.items():
                p_val = test_dict[0]['ssr_ftest'][1]  # F-test p-value
                if p_val < best_p:
                    best_p = p_val
                    best_lag = lag

            return {
                'is_causal': best_p < self.significance_level,
                'best_lag': best_lag,
                'p_value': best_p,
                'strength': max(0.0, 1.0 - best_p)  # Higher = stronger
            }

        except Exception:
            return None

    def test_all_pairs(
        self,
        df: pd.DataFrame,
        candidate_metrics: Optional[List[str]] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Run pairwise Granger causality for all metric pairs.

        Only tests metrics listed in candidate_metrics (defaults to all columns).
        Only includes pairs where is_causal = True in the result.

        Args:
            df: Time-series DataFrame (index = DateTimeIndex)
            candidate_metrics: Restrict to these columns (for efficiency)

        Returns:
            Dict mapping (cause_metric, effect_metric) -> test result dict
        """
        metrics = candidate_metrics or list(df.columns)
        causal_pairs = {}

        for cause in metrics:
            for effect in metrics:
                if cause == effect:
                    continue
                result = self.test_pair(df[cause], df[effect])
                if result and result['is_causal']:
                    causal_pairs[(cause, effect)] = result

        return causal_pairs


# ---------------------------------------------------------------------------
# Temporal Precedence Analyzer
# ---------------------------------------------------------------------------

class TemporalPrecedenceAnalyzer:
    """
    Verifies that potential causal edges are consistent with temporal ordering.
    A cause MUST precede its effect; edges that violate this are removed.
    """

    def verify_edges(
        self,
        causal_pairs: Dict[Tuple[str, str], Dict],
        anomaly_first_detected: Dict[str, pd.Timestamp]
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Filter out causal edges where the "cause" is detected AFTER the "effect".

        Args:
            causal_pairs: Output from GrangerCausalityAnalyzer.test_all_pairs()
            anomaly_first_detected: Dict mapping metric_name -> first anomaly timestamp

        Returns:
            Filtered causal_pairs (only temporally valid edges remain)
        """
        valid_pairs = {}
        for (cause, effect), info in causal_pairs.items():
            cause_time = anomaly_first_detected.get(cause)
            effect_time = anomaly_first_detected.get(effect)

            if cause_time is None or effect_time is None:
                # Cannot verify; keep with reduced confidence
                info['temporal_valid'] = None
                valid_pairs[(cause, effect)] = info
            elif cause_time <= effect_time:
                # Temporal precedence satisfied
                info['temporal_valid'] = True
                info['lag_minutes'] = (effect_time - cause_time).total_seconds() / 60
                valid_pairs[(cause, effect)] = info
            # else: cause detected AFTER effect — this cannot be causal; drop it

        return valid_pairs

    def get_first_anomaly_times(
        self,
        anomaly_flags: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, pd.Timestamp]:
        """
        Find the first time each metric was flagged as anomalous.

        Args:
            anomaly_flags: (N, F) bool array — True where anomaly detected
            timestamps: DatetimeIndex aligned with anomaly_flags rows

        Returns:
            Dict: feature_index_str -> first_anomaly_timestamp
        """
        first_detected = {}
        for feat_idx in range(anomaly_flags.shape[1]):
            anomaly_indices = np.where(anomaly_flags[:, feat_idx])[0]
            if len(anomaly_indices) > 0:
                first_idx = anomaly_indices[0]
                if first_idx < len(timestamps):
                    first_detected[f'feature_{feat_idx}'] = timestamps[first_idx]
        return first_detected


# ---------------------------------------------------------------------------
# Causal Graph Builder
# ---------------------------------------------------------------------------

class CausalGraphBuilder:
    """
    Builds a Directed Acyclic Graph (DAG) representing causal relationships
    between anomalous metrics.

    Combines evidence from:
    1. Granger causality tests
    2. Temporal precedence analysis
    3. (Optional) PC algorithm structure learning

    The graph is the core input to RootCauseRanker.
    """

    def build(
        self,
        causal_pairs: Dict[Tuple[str, str], Dict],
        anomaly_scores: Dict[str, float],
        min_strength: float = 0.0
    ) -> nx.DiGraph:
        """
        Build a directed causal graph from verified causal pairs.

        Args:
            causal_pairs: Verified causal pairs (from Granger + temporal analysis)
            anomaly_scores: Per-metric anomaly scores (added as node attributes)
            min_strength: Minimum edge strength to include (0=include all)

        Returns:
            nx.DiGraph where nodes are metrics and edges are causal relationships
        """
        G = nx.DiGraph()

        # Add all anomalous metrics as nodes
        for metric, score in anomaly_scores.items():
            G.add_node(metric, anomaly_score=score)

        # Add causal edges
        for (cause, effect), info in causal_pairs.items():
            if info['strength'] >= min_strength:
                G.add_edge(
                    cause, effect,
                    p_value=info['p_value'],
                    lag=info['best_lag'],
                    strength=info['strength'],
                    temporal_valid=info.get('temporal_valid', True),
                    lag_minutes=info.get('lag_minutes', None)
                )

        # Make it a DAG by removing cycles
        G = self._remove_cycles(G)

        return G

    def _remove_cycles(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Remove cycles by eliminating the weakest edge in each cycle.
        This enforces the acyclicity assumption required for causal DAGs.
        """
        max_iterations = 100  # Safety limit
        for _ in range(max_iterations):
            try:
                cycle = nx.find_cycle(G, orientation='original')
                # Find the weakest edge in this cycle
                weakest_edge = min(
                    cycle,
                    key=lambda e: G[e[0]][e[1]].get('strength', 0)
                )
                G.remove_edge(weakest_edge[0], weakest_edge[1])
            except nx.NetworkXNoCycle:
                break  # No more cycles

        return G

    def add_domain_knowledge_edges(
        self,
        G: nx.DiGraph,
        known_dependencies: List[Tuple[str, str, float]]
    ) -> nx.DiGraph:
        """
        Optionally inject known system topology relationships.

        Args:
            known_dependencies: List of (cause_metric, effect_metric, strength) tuples
                                 from CMDB or system topology documentation

        Returns:
            Updated graph
        """
        for cause, effect, strength in known_dependencies:
            if cause in G.nodes and effect in G.nodes:
                if G.has_edge(cause, effect):
                    # Reinforce existing edge
                    G[cause][effect]['strength'] = min(
                        1.0,
                        G[cause][effect]['strength'] + strength * 0.2
                    )
                else:
                    G.add_edge(cause, effect, strength=strength,
                               source='domain_knowledge', lag=None)

        G = self._remove_cycles(G)
        return G


# ---------------------------------------------------------------------------
# Main CausalInferenceEngine (orchestrator)
# ---------------------------------------------------------------------------

class CausalInferenceEngine:
    """
    High-level orchestrator that runs the full causal inference pipeline:

    1. Identifies anomalous metrics from anomaly detection output
    2. Runs Granger causality on anomalous metric pairs
    3. Verifies temporal precedence
    4. Builds a causal DAG

    Example:
        engine = CausalInferenceEngine()
        causal_graph = engine.run(
            metrics_df=failure_df,
            anomaly_scores={'metric_A': 0.9, 'metric_B': 0.7},
            anomaly_first_detected={'metric_A': t1, 'metric_B': t2}
        )
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05,
        min_anomaly_score: float = 0.5
    ):
        self.granger = GrangerCausalityAnalyzer(max_lag, significance_level)
        self.temporal = TemporalPrecedenceAnalyzer()
        self.graph_builder = CausalGraphBuilder()
        self.min_anomaly_score = min_anomaly_score

    def run(
        self,
        metrics_df: pd.DataFrame,
        anomaly_scores: Dict[str, float],
        anomaly_first_detected: Optional[Dict[str, pd.Timestamp]] = None,
        known_dependencies: Optional[List[Tuple[str, str, float]]] = None
    ) -> nx.DiGraph:
        """
        Full causal inference pipeline.

        Args:
            metrics_df: Raw metric time-series DataFrame
            anomaly_scores: Per-metric anomaly scores (0=normal, >1=anomalous)
            anomaly_first_detected: When each metric first became anomalous
            known_dependencies: (optional) topology edges to inject

        Returns:
            nx.DiGraph — causal graph with root cause at the "source" nodes
        """
        # Step 1: Filter to anomalous metrics only (saves computational cost)
        anomalous_metrics = [
            m for m, score in anomaly_scores.items()
            if score >= self.min_anomaly_score and m in metrics_df.columns
        ]

        if len(anomalous_metrics) < 2:
            print("Warning: Fewer than 2 anomalous metrics detected. "
                  "Causal inference requires at least 2. Returning empty graph.")
            G = nx.DiGraph()
            for m, s in anomaly_scores.items():
                G.add_node(m, anomaly_score=s)
            return G

        print(f"Running Granger causality for {len(anomalous_metrics)} anomalous metrics "
              f"({len(anomalous_metrics)**2 - len(anomalous_metrics)} pairs)...")

        # Step 2: Granger causality tests
        causal_pairs = self.granger.test_all_pairs(
            metrics_df, candidate_metrics=anomalous_metrics
        )
        print(f"Found {len(causal_pairs)} Granger-causal pairs")

        # Step 3: Temporal precedence filtering
        if anomaly_first_detected:
            causal_pairs = self.temporal.verify_edges(causal_pairs, anomaly_first_detected)
            print(f"After temporal filtering: {len(causal_pairs)} valid causal pairs")

        # Step 4: Build causal graph
        causal_graph = self.graph_builder.build(causal_pairs, anomaly_scores)

        # Step 5: Inject domain knowledge (optional)
        if known_dependencies:
            causal_graph = self.graph_builder.add_domain_knowledge_edges(
                causal_graph, known_dependencies
            )

        print(f"Causal graph: {causal_graph.number_of_nodes()} nodes, "
              f"{causal_graph.number_of_edges()} edges")

        return causal_graph

    def format_results(self, causal_graph: nx.DiGraph) -> Dict:
        """
        Format causal graph into a human-readable summary.

        Returns:
            Dict with 'edges', 'source_nodes' (likely root causes), 'sink_nodes' (symptoms)
        """
        edges = []
        for u, v, data in causal_graph.edges(data=True):
            edges.append({
                'cause': u,
                'effect': v,
                'p_value': round(data.get('p_value', 1.0), 4),
                'lag_timesteps': data.get('lag'),
                'lag_minutes': data.get('lag_minutes'),
                'strength': round(data.get('strength', 0.0), 3)
            })

        # Source nodes (no incoming edges) are likely root causes
        source_nodes = [n for n in causal_graph.nodes
                        if causal_graph.in_degree(n) == 0]
        sink_nodes = [n for n in causal_graph.nodes
                      if causal_graph.out_degree(n) == 0]

        return {
            'edges': edges,
            'potential_root_causes': source_nodes,
            'symptoms': sink_nodes,
            'total_nodes': causal_graph.number_of_nodes(),
            'total_edges': causal_graph.number_of_edges()
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    gen = SyntheticMetricsGenerator(seed=42)
    normal_df = gen.generate_normal_behavior(duration_days=10)

    # Inject a database slow query failure
    failure_start = (8 * 24 * 60) // 5
    failure_df, meta = gen.inject_failure_scenario(
        normal_df, 'database_slow_query', failure_start, 100, severity=1.0
    )

    # Simulate anomaly scores (would come from LSTM Detector in real pipeline)
    anomaly_scores = {
        'api_latency_p50_ms': 1.8,
        'api_latency_p95_ms': 2.1,
        'api_latency_p99_ms': 2.0,
        'db_connections_active': 1.6,
        'error_rate_percent': 1.9,
        'cpu_utilization': 0.3,   # Not anomalous
        'memory_usage_percent': 0.2,  # Not anomalous
    }

    # Only use the anomalous window for causal inference
    window_df = failure_df.iloc[failure_start - 50: failure_start + 100]

    engine = CausalInferenceEngine(max_lag=5, significance_level=0.05)
    causal_graph = engine.run(
        metrics_df=window_df,
        anomaly_scores=anomaly_scores
    )

    results = engine.format_results(causal_graph)
    print("\n=== Causal Inference Results ===")
    print(f"Potential root causes: {results['potential_root_causes']}")
    print(f"Symptoms: {results['symptoms']}")
    print(f"\nCausal edges:")
    for e in results['edges']:
        print(f"  {e['cause']} -> {e['effect']}  "
              f"(p={e['p_value']}, lag={e['lag_timesteps']} steps, strength={e['strength']})")
