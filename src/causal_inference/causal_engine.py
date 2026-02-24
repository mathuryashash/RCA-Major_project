"""
Causal Inference Engine for the RCA System.

Implements:
  1. Granger Causality Analysis  - determines which anomalous metrics statistically
                                    cause changes in other metrics.
  2. Causal Graph Construction   - builds a directed graph from Granger results,
                                    oriented by temporal precedence and cycle-pruning.
  3. Event Correlation           - correlates metric anomalies with deployment/config events.
  4. Root Cause Ranker           - composite scoring (causal outflow, temporal priority,
                                    anomaly severity, event correlation + PageRank).

Follows the architecture described in PRD.md §1.1.3 and §1.1.4.
"""

import numpy as np
import pandas as pd
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")  # suppress statsmodels verbose output


# ---------------------------------------------------------------------------
# 1.  Granger Causality Analysis
# ---------------------------------------------------------------------------

class GrangerAnalyzer:
    """
    Performs pairwise Granger causality tests on anomalous metric time-series.

    PRD Reference: §1.1.3  - Granger Causality Test (p-value < significance_level
                              means X Granger-causes Y).

    Usage:
        analyzer = GrangerAnalyzer(max_lag=10, significance_level=0.05)
        results  = analyzer.run(anomaly_df, anomalous_metrics)
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def _make_stationary(self, series: pd.Series) -> pd.Series:
        """
        Apply first-order differencing until the ADF test does not reject
        non-stationarity, or give up after 2 rounds.
        """
        result = series.copy()
        for _ in range(2):
            p_value = adfuller(result.dropna())[1]
            if p_value < 0.05:
                break
            result = result.diff()
        return result.dropna()

    def run(
        self,
        df: pd.DataFrame,
        anomalous_metrics: List[str],
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Test every ordered pair (cause, effect) for Granger causality.

        Parameters
        ----------
        df : pd.DataFrame
            Full time-series data (columns are metric names).
        anomalous_metrics : list[str]
            Subset of columns that were flagged as anomalous —
            limits combinatorial explosion.

        Returns
        -------
        dict  {(cause, effect): {'p_value': float, 'optimal_lag': int,
                                  'strength': float}}
            Only pairs where p_value < significance_level are included.
        """
        results: Dict[Tuple[str, str], Dict] = {}
        metrics = [m for m in anomalous_metrics if m in df.columns]

        for cause in metrics:
            for effect in metrics:
                if cause == effect:
                    continue

                try:
                    # Granger test requires: column-0 = effect, column-1 = cause
                    cause_series  = self._make_stationary(df[cause])
                    effect_series = self._make_stationary(df[effect])

                    # Align after differencing may shift indices
                    aligned = pd.concat(
                        [effect_series, cause_series], axis=1, join="inner"
                    ).dropna()

                    if len(aligned) < self.max_lag * 3:
                        continue  # Not enough data for this lag depth

                    gc = grangercausalitytests(
                        aligned.values, maxlag=self.max_lag, verbose=False
                    )

                    # Pick the lag with the smallest p-value (F-test)
                    best_lag, best_p = None, 1.0
                    for lag, test_dict in gc.items():
                        p = test_dict[0]["ssr_ftest"][1]
                        if p < best_p:
                            best_p, best_lag = p, lag

                    if best_p < self.significance_level:
                        results[(cause, effect)] = {
                            "p_value":     best_p,
                            "optimal_lag": best_lag,
                            "strength":    round(1.0 - best_p, 6),
                        }

                except Exception:
                    continue  # skip pairs with numerical issues

        return results


# ---------------------------------------------------------------------------
# 2.  Causal Graph Builder
# ---------------------------------------------------------------------------

class CausalGraphBuilder:
    """
    Constructs a directed causal graph (DAG) from Granger causality results
    and anomaly first-detection times.
    
    PRD Reference: §1.1.3 - build_causal_graph
    """

    def build(
        self,
        granger_results: Dict[Tuple[str, str], Dict],
        anomaly_scores: Dict[str, float],
        anomaly_first_seen: Optional[Dict[str, pd.Timestamp]] = None,
    ) -> nx.DiGraph:
        """
        Build a directed causal graph.

        Parameters
        ----------
        granger_results       : output of GrangerAnalyzer.run()
        anomaly_scores        : {metric: anomaly_score (0–1)}
        anomaly_first_seen    : {metric: first anomaly timestamp}
                                Used for cycle-breaking by temporal precedence.

        Returns
        -------
        nx.DiGraph  — nodes carry 'anomaly_score'; edges carry 'strength',
                      'p_value', 'lag'.
        """
        G = nx.DiGraph()

        # Add nodes with metadata
        for metric, score in anomaly_scores.items():
            G.add_node(metric, anomaly_score=score)

        # Add edges from significant Granger pairs
        for (cause, effect), info in granger_results.items():
            G.add_edge(
                cause, effect,
                strength=info["strength"],
                p_value=info["p_value"],
                lag=info["optimal_lag"],
            )

        # Prune cycles to maintain DAG property
        G = self._break_cycles(G, anomaly_first_seen)

        return G

    def _break_cycles(
        self,
        G: nx.DiGraph,
        anomaly_first_seen: Optional[Dict[str, pd.Timestamp]],
    ) -> nx.DiGraph:
        """
        Iteratively remove the weakest edge inside any detected cycle.
        If temporal precedence data is available, prefer to remove edges
        that violate the "cause precedes effect" rule.
        """
        max_iterations = len(G.edges) + 1

        for _ in range(max_iterations):
            try:
                cycle = nx.find_cycle(G, orientation="original")
            except nx.NetworkXNoCycle:
                break

            # Collect candidate edges to remove from this cycle
            cycle_edges = [(u, v) for u, v, _ in cycle]

            # Prefer removing an edge that violates temporal precedence
            removed = False
            if anomaly_first_seen:
                for u, v in cycle_edges:
                    t_u = anomaly_first_seen.get(u)
                    t_v = anomaly_first_seen.get(v)
                    if t_u is not None and t_v is not None and t_u > t_v:
                        # u appeared AFTER v — edge u->v violates temporal ordering
                        if G.has_edge(u, v):
                            G.remove_edge(u, v)
                            removed = True
                            break

            if not removed:
                # Fall back: remove the weakest edge (lowest Granger strength)
                weakest, weakest_strength = None, float("inf")
                for u, v in cycle_edges:
                    strength = G[u][v].get("strength", 0.0) if G.has_edge(u, v) else float("inf")
                    if strength < weakest_strength:
                        weakest_strength, weakest = strength, (u, v)
                if weakest and G.has_edge(*weakest):
                    G.remove_edge(*weakest)

        return G


# ---------------------------------------------------------------------------
# 3.  Event Correlator
# ---------------------------------------------------------------------------

class EventCorrelator:
    """
    Correlates metric anomaly timestamps with external system events
    (deployments, config changes, feature-flag toggles, etc.)

    PRD Reference: §1.1.4 - Event Correlation (5% scoring weight)
    """

    def correlate(
        self,
        anomaly_first_seen: Dict[str, pd.Timestamp],
        events_df: pd.DataFrame,
        max_lag_hours: float = 24.0,
    ) -> List[Dict]:
        """
        For each anomalous metric, find system events that immediately 
        preceded its first anomalous reading.

        Parameters
        ----------
        anomaly_first_seen : {metric: timestamp}
        events_df          : DataFrame with columns ['timestamp', 'description',
                              'type' (optional), 'component' (optional)]
        max_lag_hours      : Maximum look-back window in hours.

        Returns
        -------
        list of dicts sorted by correlation_score (descending).
        """
        if events_df is None or events_df.empty:
            return []

        correlations = []
        required_cols = {"timestamp", "description"}
        if not required_cols.issubset(events_df.columns):
            return []

        for metric, anomaly_time in anomaly_first_seen.items():
            for _, event in events_df.iterrows():
                event_time = pd.Timestamp(event["timestamp"])

                # Cause must precede its effect
                delta_hours = (anomaly_time - event_time).total_seconds() / 3600.0
                if 0 < delta_hours <= max_lag_hours:
                    # Closer in time -> higher score
                    correlation_score = 1.0 / (1.0 + delta_hours)
                    correlations.append({
                        "metric":            metric,
                        "event_description": event["description"],
                        "event_time":        event_time,
                        "event_type":        event.get("type", "unknown"),
                        "anomaly_time":      anomaly_time,
                        "delta_hours":       round(delta_hours, 3),
                        "correlation_score": round(correlation_score, 6),
                    })

        return sorted(correlations, key=lambda x: x["correlation_score"], reverse=True)


# ---------------------------------------------------------------------------
# 4.  Root Cause Ranker
# ---------------------------------------------------------------------------

class RootCauseRanker:
    """
    Scores every node in the causal graph as a candidate root cause using a
    weighted composite metric, augmented by PageRank on the reversed graph.

    PRD Reference: §1.1.4 — Multi-factor scoring framework
        Weight breakdown:
          causal_outflow      40 %  (many downstream effects -> likely root cause)
          temporal_priority   30 %  (early appearance -> likely root cause)
          causal_inflow       20 %  (few upstream causes -> likely root cause)
          anomaly_severity     5 %
          event_correlation    5 %
    """

    DEFAULT_WEIGHTS = {
        "causal_outflow":   0.40,
        "temporal_priority": 0.30,
        "causal_inflow":    0.20,
        "anomaly_severity": 0.05,
        "event_correlation": 0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def rank(
        self,
        causal_graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        anomaly_first_seen: Dict[str, pd.Timestamp],
        event_correlations: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Score and rank each node in the causal graph.

        Returns
        -------
        list of dicts sorted by composite_score descending:
            {rank, metric, composite_score, confidence, scores_breakdown,
             downstream_effects, causal_chain}
        """
        if len(causal_graph.nodes) == 0:
            return []

        candidates: List[Dict] = []

        # PageRank on *reversed* graph: high rank = influential source
        reversed_g = causal_graph.reverse()
        try:
            pagerank = nx.pagerank(reversed_g, weight="strength")
        except Exception:
            n = len(causal_graph.nodes)
            pagerank = {node: 1.0 / n for node in causal_graph.nodes}

        # Temporal reference points
        times = [t for t in anomaly_first_seen.values() if t is not None]
        if times:
            t_earliest = min(times)
            t_latest   = max(times)
            t_range    = (t_latest - t_earliest).total_seconds() or 1.0
        else:
            t_earliest = t_latest = t_range = None

        # Degree normalisation
        out_degrees = dict(causal_graph.out_degree())
        in_degrees  = dict(causal_graph.in_degree())
        max_out = max(out_degrees.values()) if out_degrees else 1
        max_in  = max(in_degrees.values())  if in_degrees  else 1

        for metric in causal_graph.nodes:
            s: Dict[str, float] = {}

            # 1. Causal outflow
            s["causal_outflow"] = out_degrees.get(metric, 0) / (max_out or 1)

            # 2. Causal inflow penalty (fewer ancestors -> better root cause)
            s["causal_inflow"] = 1.0 - in_degrees.get(metric, 0) / (max_in or 1)

            # 3. Temporal priority
            if t_range and metric in anomaly_first_seen and anomaly_first_seen[metric]:
                offset = (anomaly_first_seen[metric] - t_earliest).total_seconds()
                s["temporal_priority"] = 1.0 - (offset / t_range)
            else:
                s["temporal_priority"] = 0.0

            # 4. Anomaly severity
            s["anomaly_severity"] = float(anomaly_scores.get(metric, 0.0))

            # 5. Event correlation
            if event_correlations:
                relevant = [
                    ec for ec in event_correlations if ec["metric"] == metric
                ]
                s["event_correlation"] = relevant[0]["correlation_score"] if relevant else 0.0
            else:
                s["event_correlation"] = 0.0

            # Weighted composite (70%) + PageRank component (30%)
            composite = sum(self.weights[k] * s[k] for k in self.weights)
            composite = 0.70 * composite + 0.30 * pagerank.get(metric, 0.0)
            composite = round(min(composite, 1.0), 6)

            # Determine confidence label
            confidence = self._confidence_label(composite)

            # Downstream effects (direct successors in causal graph)
            downstream = list(causal_graph.successors(metric))

            # Full causal chain via DFS
            causal_chain = self._trace_chain(causal_graph, metric)

            candidates.append({
                "metric":           metric,
                "composite_score":  composite,
                "confidence":       confidence,
                "scores_breakdown": {k: round(v, 4) for k, v in s.items()},
                "pagerank":         round(pagerank.get(metric, 0.0), 6),
                "downstream_effects": downstream,
                "causal_chain":     causal_chain,
            })

        # Sort descending
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        # Add rank labels (1-indexed)
        for i, candidate in enumerate(candidates, start=1):
            candidate["rank"] = i

        return candidates

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.95:
            return "Critical"
        elif score >= 0.85:
            return "High"
        elif score >= 0.70:
            return "Medium"
        elif score >= 0.50:
            return "Low"
        else:
            return "Very Low"

    @staticmethod
    def _trace_chain(G: nx.DiGraph, start: str) -> List[str]:
        """
        Depth-first traversal from `start` to capture propagation chain.
        Returns list of node names in discovery order.
        """
        chain = [start]
        visited = {start}
        stack = list(G.successors(start))

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                chain.append(node)
                stack.extend(G.successors(node))

        return chain


# ---------------------------------------------------------------------------
# 5.  High-Level Pipeline Wrapper
# ---------------------------------------------------------------------------

class CausalInferencePipeline:
    """
    Orchestrates the full causal-inference -> root-cause-ranking workflow.

    Example
    -------
    >>> pipeline = CausalInferencePipeline()
    >>> results  = pipeline.run(
    ...     df=metrics_df,
    ...     anomalous_metrics=['cpu_utilization', 'db_connections_active', 'error_rate_percent'],
    ...     anomaly_scores={'cpu_utilization': 0.92, 'db_connections_active': 0.85, 'error_rate_percent': 0.97},
    ...     anomaly_first_seen={'cpu_utilization': t0, 'db_connections_active': t1, 'error_rate_percent': t2},
    ...     events_df=events,
    ... )
    >>> for rc in results['root_causes'][:3]:
    ...     print(rc['rank'], rc['metric'], rc['confidence'], rc['composite_score'])
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05,
        rc_weights: Optional[Dict[str, float]] = None,
    ):
        self.granger        = GrangerAnalyzer(max_lag, significance_level)
        self.graph_builder  = CausalGraphBuilder()
        self.event_corr     = EventCorrelator()
        self.ranker         = RootCauseRanker(rc_weights)

    def run(
        self,
        df: pd.DataFrame,
        anomalous_metrics: List[str],
        anomaly_scores: Dict[str, float],
        anomaly_first_seen: Dict[str, pd.Timestamp],
        events_df: Optional[pd.DataFrame] = None,
        max_event_lag_hours: float = 24.0,
    ) -> Dict:
        """
        Full pipeline from raw metrics -> ranked root causes.

        Returns
        -------
        {
          'granger_results'    : dict of significant causal pairs,
          'causal_graph'       : nx.DiGraph,
          'event_correlations' : list,
          'root_causes'        : list (sorted by composite_score desc),
        }
        """
        print(f"[CausalInferencePipeline] Running Granger causality on "
              f"{len(anomalous_metrics)} anomalous metrics …")

        granger_results = self.granger.run(df, anomalous_metrics)
        print(f"  -> {len(granger_results)} significant causal pairs found.")

        causal_graph = self.graph_builder.build(
            granger_results, anomaly_scores, anomaly_first_seen
        )
        print(f"  -> Causal graph: {len(causal_graph.nodes)} nodes, "
              f"{len(causal_graph.edges)} edges.")

        event_correlations = self.event_corr.correlate(
            anomaly_first_seen, events_df, max_event_lag_hours
        ) if events_df is not None else []
        print(f"  -> {len(event_correlations)} event correlations found.")

        root_causes = self.ranker.rank(
            causal_graph, anomaly_scores, anomaly_first_seen, event_correlations
        )
        print(f"  -> Root cause ranking complete.")

        return {
            "granger_results":    granger_results,
            "causal_graph":       causal_graph,
            "event_correlations": event_correlations,
            "root_causes":        root_causes,
        }


# ---------------------------------------------------------------------------
# 6.  Self-test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-test that uses synthetic metric data to validate the pipeline.
    Requires: numpy, pandas, statsmodels, networkx (all in requirements.txt).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    print("=" * 60)
    print(" Causal Inference Engine — Self-Test")
    print("=" * 60)

    # 1. Generate synthetic data with a known database failure
    gen = SyntheticMetricsGenerator(seed=42)
    normal_df = gen.generate_normal_behavior(duration_days=30, sampling_interval_minutes=5)
    failure_start = len(normal_df) - 200
    failed_df, meta = gen.inject_failure_scenario(
        normal_df,
        failure_type="database_slow_query",
        start_idx=failure_start,
        duration_samples=200,
        severity=1.0,
    )

    # Use timestamp as index for time-aware Granger tests
    failed_df = failed_df.set_index("timestamp")

    feature_cols = list(failed_df.columns)

    # 2. Simulate anomaly detection output
    # In a real pipeline these come from the LSTM AnomalyDetector
    anomaly_scores = {
        "api_latency_p50_ms":   0.92,
        "api_latency_p95_ms":   0.95,
        "db_connections_active": 0.88,
        "error_rate_percent":   0.97,
    }
    anomalous_metrics = list(anomaly_scores.keys())

    # First anomaly time = failure start timestamp
    t_base = failed_df.index[failure_start]
    anomaly_first_seen = {
        "api_latency_p50_ms":   t_base,
        "api_latency_p95_ms":   t_base + pd.Timedelta(minutes=5),
        "db_connections_active": t_base + pd.Timedelta(minutes=10),
        "error_rate_percent":   t_base + pd.Timedelta(minutes=15),
    }

    # 3. Simulate a deployment event 20 min before failure
    events_df = pd.DataFrame([{
        "timestamp":   t_base - pd.Timedelta(minutes=20),
        "description": "Schema migration v3.4.1 deployed to production DB",
        "type":        "deployment",
    }])

    # 4. Run pipeline
    pipeline = CausalInferencePipeline(max_lag=5, significance_level=0.05)
    results = pipeline.run(
        df=failed_df[anomalous_metrics],
        anomalous_metrics=anomalous_metrics,
        anomaly_scores=anomaly_scores,
        anomaly_first_seen=anomaly_first_seen,
        events_df=events_df,
    )

    print("\n--- Top Root Cause Candidates ---")
    for rc in results["root_causes"]:
        print(
            f"  #{rc['rank']}  {rc['metric']:35s}  "
            f"score={rc['composite_score']:.4f}  "
            f"confidence={rc['confidence']}"
        )

    print("\n--- Causal Graph Edges ---")
    for u, v, data in results["causal_graph"].edges(data=True):
        print(f"  {u} -> {v}  (strength={data.get('strength', 0):.4f}, lag={data.get('lag', '?')})")

    if results["event_correlations"]:
        top_event = results["event_correlations"][0]
        print(f"\n--- Top Event Correlation ---")
        print(f"  {top_event['event_description']} -> affects {top_event['metric']}"
              f"  (score={top_event['correlation_score']:.4f})")

    print("\nSelf-test complete.")
