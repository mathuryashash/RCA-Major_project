"""
Root Cause Ranker

Ranks candidate anomalies as potential root causes using a multi-factor
composite scoring framework as described in the PRD.

Scoring factors (weights configurable):
1. Causal Outflow     (40%): How many things does this metric cause downstream?
2. Causal Inflow      (20%): Fewer incoming causes -> more likely primary root cause
3. Temporal Priority  (30%): Earlier anomaly detection -> more likely root cause
4. Anomaly Severity   ( 5%): How far from normal?
5. Event Correlation  ( 5%): Did a deployment/config change precede this?

Also uses PageRank on the causal graph as a supplementary signal.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RootCauseScore:
    """Represents a ranked root cause candidate."""
    metric: str
    rank: int
    final_score: float
    confidence: str           # 'Critical', 'High', 'Medium', 'Low', 'Very Low'
    confidence_pct: float     # 0–100
    factor_scores: Dict[str, float] = field(default_factory=dict)
    downstream_effects: List[str] = field(default_factory=list)
    justification: str = ""


class RootCauseRanker:
    """
    Scores and ranks all anomalous metrics as potential root causes.

    Usage:
        ranker = RootCauseRanker()
        ranked = ranker.rank(
            causal_graph         = causal_graph,
            anomaly_scores       = {'metric_A': 0.9, ...},
            anomaly_first_detected = {'metric_A': t1, ...}
        )
        for rc in ranked:
            print(rc.rank, rc.metric, rc.confidence_pct, rc.justification)
    """

    WEIGHTS = {
        'causal_outflow':  0.40,
        'causal_inflow':   0.20,
        'temporal':        0.30,
        'severity':        0.05,
        'event':           0.05,
    }

    CONFIDENCE_LEVELS = [
        (95, 'Critical'),
        (85, 'High'),
        (70, 'Medium'),
        (50, 'Low'),
        (0,  'Very Low'),
    ]

    def rank(
        self,
        causal_graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        anomaly_first_detected: Optional[Dict[str, pd.Timestamp]] = None,
        event_correlations: Optional[List[Dict]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> List[RootCauseScore]:
        """
        Rank candidate root causes.

        Args:
            causal_graph: Directed causal DAG from CausalInferenceEngine
            anomaly_scores: Normalized anomaly scores per metric (>1 = anomalous)
            anomaly_first_detected: Timestamp of first anomaly per metric
            event_correlations: List of {affected_metric, correlation_score, event, ...}
                                 from deployment/config change events
            weights: Override default scoring weights

        Returns:
            List of RootCauseScore, sorted by final_score descending
        """
        w = weights or self.WEIGHTS
        nodes = list(causal_graph.nodes())
        if not nodes:
            return []

        # PageRank on reversed graph: high rank = nodes that influence many others
        try:
            rev = causal_graph.reverse()
            pagerank = nx.pagerank(rev, weight='strength', alpha=0.85)
        except Exception:
            pagerank = {n: 1.0 / len(nodes) for n in nodes}

        # Temporal alignment
        if anomaly_first_detected and anomaly_first_detected:
            all_times = list(anomaly_first_detected.values())
            earliest = min(all_times)
            latest = max(all_times)
            time_range_sec = max((latest - earliest).total_seconds(), 1.0)
        else:
            earliest = None
            time_range_sec = 1.0

        # Event correlation lookup
        event_map: Dict[str, float] = {}
        if event_correlations:
            for ec in event_correlations:
                m = ec.get('affected_metric', '')
                if m and ec.get('correlation_score', 0) > event_map.get(m, 0):
                    event_map[m] = ec['correlation_score']

        max_out = max((causal_graph.out_degree(n) for n in nodes), default=1)
        max_in  = max((causal_graph.in_degree(n)  for n in nodes), default=1)

        scored: List[Tuple[float, RootCauseScore]] = []

        for metric in nodes:
            factors: Dict[str, float] = {}

            # 1. Causal outflow: more downstream = more likely root cause
            out = causal_graph.out_degree(metric)
            factors['causal_outflow'] = out / max(max_out, 1)

            # 2. Causal inflow: fewer predecessors = more likely root cause
            inn = causal_graph.in_degree(metric)
            factors['causal_inflow'] = 1.0 - (inn / max(max_in, 1))

            # 3. Temporal priority: earlier detection = more likely root cause
            if earliest and metric in (anomaly_first_detected or {}):
                t = anomaly_first_detected[metric]
                offset_sec = (t - earliest).total_seconds()
                factors['temporal'] = 1.0 - (offset_sec / time_range_sec)
            else:
                factors['temporal'] = 0.5  # Unknown temporal order

            # 4. Anomaly severity
            raw_score = anomaly_scores.get(metric, 0.0)
            # Normalize to 0–1: scores > 1 are anomalous; cap at 3 for normalization
            factors['severity'] = min(raw_score / 3.0, 1.0)

            # 5. Event correlation
            factors['event'] = event_map.get(metric, 0.0)

            # Composite score (weighted sum)
            composite = sum(w.get(k, 0) * v for k, v in factors.items())

            # Blend with PageRank signal (30% weight)
            pagerank_score = pagerank.get(metric, 0.0)
            final_score = 0.70 * composite + 0.30 * pagerank_score

            # Downstream effects (nodes reachable from this metric)
            try:
                descendants = list(nx.descendants(causal_graph, metric))
            except Exception:
                descendants = []

            # Confidence level
            confidence_pct = min(99.0, final_score * 100)
            confidence = self._get_confidence_level(confidence_pct)

            rc = RootCauseScore(
                metric=metric,
                rank=0,  # filled after sorting
                final_score=round(final_score, 4),
                confidence=confidence,
                confidence_pct=round(confidence_pct, 1),
                factor_scores={k: round(v, 3) for k, v in factors.items()},
                downstream_effects=descendants,
            )
            scored.append((final_score, rc))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        result = []
        for rank, (_, rc) in enumerate(scored, start=1):
            rc.rank = rank
            rc.justification = self._generate_justification(rc, anomaly_first_detected, earliest)
            result.append(rc)

        return result

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _get_confidence_level(self, pct: float) -> str:
        for threshold, label in self.CONFIDENCE_LEVELS:
            if pct >= threshold:
                return label
        return 'Very Low'

    def _generate_justification(
        self,
        rc: RootCauseScore,
        anomaly_first_detected: Optional[Dict],
        earliest: Optional[pd.Timestamp]
    ) -> str:
        lines = [
            f"=== Root Cause: {rc.metric} (Rank #{rc.rank}) ===",
            f"Confidence: {rc.confidence} ({rc.confidence_pct:.1f}%)",
            "",
        ]

        # Temporal
        if earliest and anomaly_first_detected and rc.metric in anomaly_first_detected:
            t = anomaly_first_detected[rc.metric]
            delay_min = (t - earliest).total_seconds() / 60
            if delay_min < 1:
                lines.append("• TEMPORAL: This is the EARLIEST detected anomaly — "
                             "strongly suggests it is the root cause.")
            else:
                lines.append(f"• TEMPORAL: Detected {delay_min:.1f} minutes after first anomaly.")
        else:
            lines.append("• TEMPORAL: Detection time not available.")

        # Causal outflow
        ds = rc.downstream_effects
        if ds:
            lines.append(
                f"• CAUSAL IMPACT: This metric directly or indirectly causes "
                f"{len(ds)} downstream effect(s): {', '.join(ds[:5])}{'...' if len(ds) > 5 else ''}"
            )
        else:
            lines.append("• CAUSAL IMPACT: No downstream effects detected — possible leaf node.")

        # Factor scores
        lines.append("• FACTOR SCORES:")
        for k, v in rc.factor_scores.items():
            lines.append(f"    {k:20s}: {v:.3f}")

        return "\n".join(lines)

    def generate_summary_report(
        self,
        ranked_causes: List[RootCauseScore],
        top_n: int = 3
    ) -> str:
        """
        Generate a concise human-readable summary of the top root causes.
        """
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║           ROOT CAUSE ANALYSIS SUMMARY                ║",
            "╚══════════════════════════════════════════════════════╝",
            ""
        ]

        if not ranked_causes:
            lines.append("No root causes identified. Insufficient anomaly data.")
            return "\n".join(lines)

        for rc in ranked_causes[:top_n]:
            lines.append(f"Rank #{rc.rank}: {rc.metric}")
            lines.append(f"  Confidence: {rc.confidence} ({rc.confidence_pct:.1f}%)")
            lines.append(f"  Downstream effects: {', '.join(rc.downstream_effects) or 'None'}")
            lines.append("")

        if len(ranked_causes) > top_n:
            lines.append(
                f"... and {len(ranked_causes) - top_n} other candidate(s) with lower confidence."
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event correlation helper
# ---------------------------------------------------------------------------

def correlate_deployment_events(
    anomaly_timestamps: Dict[str, pd.Timestamp],
    events_df: pd.DataFrame,
    window_hours: float = 24.0
) -> List[Dict]:
    """
    Correlate anomaly detection times with deployment/config change events.

    Args:
        anomaly_timestamps: Dict of metric -> first_anomaly_timestamp
        events_df: DataFrame with columns: ['timestamp', 'description', 'type']
        window_hours: Maximum time after event to attribute to causation

    Returns:
        List of correlation dicts sorted by correlation_score descending
    """
    correlations = []

    for _, event in events_df.iterrows():
        event_time = event['timestamp']
        for metric, anomaly_time in anomaly_timestamps.items():
            time_diff_h = (anomaly_time - event_time).total_seconds() / 3600
            if 0 < time_diff_h <= window_hours:
                correlations.append({
                    'event': event.get('description', 'Unknown event'),
                    'event_type': event.get('type', 'unknown'),
                    'event_time': event_time,
                    'affected_metric': metric,
                    'anomaly_time': anomaly_time,
                    'time_delta_hours': round(time_diff_h, 2),
                    'correlation_score': round(1.0 / (1.0 + time_diff_h), 4)
                })

    return sorted(correlations, key=lambda x: x['correlation_score'], reverse=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Build a simple synthetic causal graph
    G = nx.DiGraph()
    G.add_node('api_latency_p95_ms', anomaly_score=2.1)
    G.add_node('db_connections_active', anomaly_score=1.6)
    G.add_node('error_rate_percent', anomaly_score=1.9)
    G.add_node('api_latency_p50_ms', anomaly_score=1.8)

    G.add_edge('api_latency_p95_ms', 'db_connections_active', strength=0.8, lag=2, p_value=0.01)
    G.add_edge('db_connections_active', 'error_rate_percent', strength=0.7, lag=3, p_value=0.02)
    G.add_edge('api_latency_p50_ms', 'api_latency_p95_ms', strength=0.9, lag=1, p_value=0.005)

    anomaly_scores = {
        'api_latency_p50_ms': 1.8,
        'api_latency_p95_ms': 2.1,
        'db_connections_active': 1.6,
        'error_rate_percent': 1.9,
    }

    import pandas as pd
    base_time = pd.Timestamp('2024-01-01 12:00:00')
    anomaly_first_detected = {
        'api_latency_p50_ms': base_time + pd.Timedelta(minutes=0),
        'api_latency_p95_ms': base_time + pd.Timedelta(minutes=5),
        'db_connections_active': base_time + pd.Timedelta(minutes=15),
        'error_rate_percent': base_time + pd.Timedelta(minutes=30),
    }

    ranker = RootCauseRanker()
    ranked = ranker.rank(G, anomaly_scores, anomaly_first_detected)

    print(ranker.generate_summary_report(ranked, top_n=3))
    print()
    for rc in ranked:
        print(rc.justification)
        print()
