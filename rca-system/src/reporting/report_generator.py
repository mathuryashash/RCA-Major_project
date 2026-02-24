"""
Report Generator & Visualization

Generates reports in multiple formats from RCA pipeline results:
  1. Executive summary (concise, for leadership)
  2. Technical report (detailed, for engineers)
  3. JSON report (machine-readable, for integrations)
  4. HTML report (rich-formatted, for sharing)

Also provides a lightweight ASCII causal-graph visualiser that
works without any plotting dependencies.
"""

import json
import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

# Import only what is available at runtime; the type hints are kept as strings
# to avoid circular import issues in the larger pipeline.
try:
    from src.root_cause_ranking.scorer import RootCauseScore
except Exception:
    RootCauseScore = object  # type: ignore


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _box(title: str, width: int = 60) -> str:
    """Return a simple ASCII box header line."""
    bar = "=" * width
    padded = title.center(width - 2)
    return f"\n{bar}\n {padded}\n{bar}"


def _format_timestamp(ts) -> str:
    """Format a timestamp to a human-readable string."""
    if ts is None:
        return "N/A"
    if isinstance(ts, str):
        return ts
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(ts)


def _confidence_bar(pct: float, width: int = 20) -> str:
    """Return an ASCII progress bar for a confidence percentage."""
    filled = int(round(pct / 100 * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:.1f}%"


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates reports from RCA pipeline results in multiple formats.

    Usage:
        rg = ReportGenerator()

        # After running the pipeline:
        print(rg.generate_executive_summary(results, failure_duration_min=45))
        print(rg.generate_technical_report(results, metrics_df))
        html = rg.generate_html_report(results, metrics_df)
        json_str = rg.generate_json_report(results)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_executive_summary(
        self,
        results: Dict,
        failure_duration_min: int = 0,
        estimated_impact_dollars: Optional[float] = None,
        status: str = "Investigation",
        incident_id: Optional[str] = None,
    ) -> str:
        """
        One-page concise summary for executive / on-call leadership.

        Args:
            results:                    Output from RCAPipeline.analyze()
            failure_duration_min:       Estimated incident duration in minutes
            estimated_impact_dollars:   Optional dollar cost estimate
            status:                     'Resolved', 'In-progress', or 'Investigation'
            incident_id:                Optional incident identifier

        Returns:
            Formatted multi-line string
        """
        ranked: List = results.get("ranked_causes", [])
        anomaly_scores: Dict = results.get("anomaly_scores", {})
        timing: Dict = results.get("timing", {})
        causal_summary: Dict = results.get("causal_graph_summary", {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = []
        lines.append(_box("ROOT CAUSE ANALYSIS — EXECUTIVE SUMMARY"))
        lines.append(f"\n  Incident ID  : {incident_id or 'N/A'}")
        lines.append(f"  Generated    : {now}")
        lines.append(f"  Status       : {status}")
        lines.append(f"  Duration     : {failure_duration_min} minutes")
        if estimated_impact_dollars is not None:
            lines.append(f"  Est. Impact  : ${estimated_impact_dollars:,.0f}")

        lines.append("\n" + "-" * 60)
        lines.append("  WHAT HAPPENED")
        lines.append("-" * 60)

        anomalous_count = sum(1 for s in anomaly_scores.values() if s >= 1.0)
        lines.append(
            f"  {anomalous_count} metric(s) deviated significantly from normal baseline.\n"
            f"  Causal analysis identified {causal_summary.get('total_edges', 0)} "
            f"causal relationship(s) among "
            f"{causal_summary.get('total_nodes', 0)} anomalous metric(s)."
        )

        lines.append("\n" + "-" * 60)
        lines.append("  TOP ROOT CAUSES")
        lines.append("-" * 60)
        if ranked:
            for rc in ranked[:3]:
                lines.append(
                    f"  #{rc.rank}  {rc.metric}\n"
                    f"      Confidence : {_confidence_bar(rc.confidence_pct)}\n"
                    f"      Level      : {rc.confidence}\n"
                    f"      Impact     : {len(rc.downstream_effects)} downstream effect(s)"
                )
        else:
            lines.append("  No root causes identified — insufficient anomaly data.")

        lines.append("\n" + "-" * 60)
        lines.append("  RECOMMENDED IMMEDIATE ACTIONS")
        lines.append("-" * 60)
        if ranked:
            top = ranked[0]
            lines.append(
                f"  1. Investigate {top.metric} first (highest confidence root cause)."
            )
            if top.downstream_effects:
                lines.append(
                    f"  2. Monitor / mitigate downstream metrics: "
                    f"{', '.join(top.downstream_effects[:3])}."
                )
        else:
            lines.append("  1. Collect more data to enable root cause identification.")

        total_sec = sum(timing.values())
        lines.append(
            f"\n  Analysis completed in {total_sec:.1f}s  "
            f"(detection={timing.get('anomaly_detection_sec', 0)}s, "
            f"causal={timing.get('causal_inference_sec', 0)}s, "
            f"ranking={timing.get('ranking_sec', 0)}s)"
        )
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def generate_technical_report(
        self,
        results: Dict,
        metrics_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Detailed technical report for SRE/on-call engineers.

        Sections:
          1. Incident Timeline
          2. Root Cause Analysis
          3. Failure Propagation Chain
          4. Anomaly Details
          5. Recommendations
        """
        ranked: List = results.get("ranked_causes", [])
        anomaly_scores: Dict = results.get("anomaly_scores", {})
        anomaly_first_detected: Dict = results.get("anomaly_first_detected", {})
        causal_summary: Dict = results.get("causal_graph_summary", {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = []
        lines.append(_box("ROOT CAUSE ANALYSIS — TECHNICAL REPORT", width=70))
        lines.append(f"\nGenerated: {now}\n")

        # --- Section 1: Incident Timeline ---
        lines.append(_box("SECTION 1: INCIDENT TIMELINE", width=70))
        sorted_metrics = sorted(
            anomaly_first_detected.items(), key=lambda x: str(x[1])
        )
        if sorted_metrics:
            lines.append(
                f"\n{'Metric':<40}{'First Detected':<30}{'Score':>10}"
            )
            lines.append("-" * 80)
            for metric, ts in sorted_metrics:
                score = anomaly_scores.get(metric, 0)
                flag = " ⚠️" if score >= 1.0 else ""
                lines.append(
                    f"  {metric:<38}{_format_timestamp(ts):<30}{score:>8.3f}{flag}"
                )
        else:
            lines.append("\n  No anomaly timestamps recorded.")

        if metrics_df is not None:
            lines.append(
                f"\n  Metrics DataFrame: {len(metrics_df)} rows "
                f"× {len(metrics_df.columns)} columns"
            )
            lines.append(
                f"  Time range: {_format_timestamp(metrics_df.index[0])} "
                f"-> {_format_timestamp(metrics_df.index[-1])}"
            )

        # --- Section 2: Root Cause Analysis ---
        lines.append("\n" + _box("SECTION 2: ROOT CAUSE ANALYSIS", width=70))
        if ranked:
            for rc in ranked[:5]:
                lines.append(
                    f"\n  Rank #{rc.rank}: {rc.metric}"
                )
                lines.append(
                    f"  Confidence : {_confidence_bar(rc.confidence_pct, width=25)} "
                    f"[{rc.confidence}]"
                )
                lines.append(f"  Final Score: {rc.final_score:.4f}")
                lines.append("\n  Factor Breakdown:")
                for factor, score in rc.factor_scores.items():
                    lines.append(f"    {factor:<25}: {score:.3f}")
                if rc.downstream_effects:
                    lines.append(
                        f"\n  Downstream ({len(rc.downstream_effects)} effect(s)): "
                        f"{', '.join(rc.downstream_effects)}"
                    )
                lines.append("\n  " + "-" * 65)
        else:
            lines.append("\n  No root causes identified.")

        # --- Section 3: Failure Propagation Chain ---
        lines.append("\n" + _box("SECTION 3: CAUSAL GRAPH SUMMARY", width=70))
        edges: List[Dict] = causal_summary.get("edges", [])
        if edges:
            lines.append(
                f"\n  {'Cause':<35}{'Effect':<35}{'p-value':>8}  {'lag':>5}  {'strength':>8}"
            )
            lines.append("  " + "-" * 95)
            for e in edges:
                lines.append(
                    f"  {e['cause']:<35}{e['effect']:<35}"
                    f"{e['p_value']:>8.4f}  "
                    f"{str(e.get('lag_timesteps', 'N/A')):>5}  "
                    f"{e['strength']:>8.3f}"
                )
        else:
            lines.append("\n  No causal edges found.")

        potential_rc = causal_summary.get("potential_root_causes", [])
        symptoms = causal_summary.get("symptoms", [])
        lines.append(f"\n  Graph source nodes (potential root causes): {potential_rc}")
        lines.append(f"  Graph sink nodes (symptoms)               : {symptoms}")

        # --- Section 4: Anomaly Details ---
        lines.append("\n" + _box("SECTION 4: ANOMALY DETAILS", width=70))
        sorted_anomalies = sorted(
            anomaly_scores.items(), key=lambda x: -x[1]
        )
        lines.append(
            f"\n  {'Metric':<40}{'Anomaly Score':>15}{'Status':>12}"
        )
        lines.append("  " + "-" * 68)
        for metric, score in sorted_anomalies:
            status = "ANOMALOUS" if score >= 1.0 else "normal"
            lines.append(f"  {metric:<40}{score:>15.4f}{status:>12}")

        # --- Section 5: Recommendations ---
        lines.append("\n" + _box("SECTION 5: RECOMMENDATIONS", width=70))
        lines.append(self._generate_recommendations(ranked))

        lines.append("\n" + "=" * 70 + "\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def generate_json_report(
        self,
        results: Dict,
        incident_id: Optional[str] = None,
        failure_duration_min: int = 0,
        status: str = "Investigation",
    ) -> str:
        """
        Serialise pipeline results to a JSON string.

        Schema matches the PRD-specified format plus timing metadata.

        Returns:
            Formatted JSON string (pretty-printed, 2-space indent)
        """
        ranked: List = results.get("ranked_causes", [])
        now = datetime.now(timezone.utc).isoformat()

        doc = {
            "incident_id": incident_id or f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            "timestamp": now,
            "duration_minutes": failure_duration_min,
            "status": status,
            "summary": {
                "anomalous_metrics": [
                    m for m, s in results.get("anomaly_scores", {}).items() if s >= 1.0
                ],
                "total_metrics_analyzed": len(results.get("anomaly_scores", {})),
                "causal_edges_found": results.get("causal_graph_summary", {}).get("total_edges", 0),
            },
            "root_causes": [
                {
                    "rank": rc.rank,
                    "metric": rc.metric,
                    "confidence_label": rc.confidence,
                    "confidence_pct": rc.confidence_pct,
                    "final_score": rc.final_score,
                    "factor_scores": rc.factor_scores,
                    "downstream_effects": rc.downstream_effects,
                    "justification": rc.justification,
                }
                for rc in ranked
            ],
            "causal_graph": results.get("causal_graph_summary", {}),
            "anomaly_scores": results.get("anomaly_scores", {}),
            "anomaly_first_detected": {
                k: _format_timestamp(v)
                for k, v in results.get("anomaly_first_detected", {}).items()
            },
            "timing": results.get("timing", {}),
        }

        return json.dumps(doc, indent=2, default=str)

    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        results: Dict,
        metrics_df: Optional[pd.DataFrame] = None,
        incident_id: Optional[str] = None,
        failure_duration_min: int = 0,
    ) -> str:
        """
        Generate a self-contained HTML report for browser viewing or email.

        Returns:
            Complete HTML document as string
        """
        ranked: List = results.get("ranked_causes", [])
        anomaly_scores: Dict = results.get("anomaly_scores", {})
        causal_summary: Dict = results.get("causal_graph_summary", {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        iid = incident_id or f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # Build root cause table rows
        rc_rows = ""
        for rc in ranked[:5]:
            confidence_color = {
                "Critical": "#c0392b",
                "High": "#e67e22",
                "Medium": "#f1c40f",
                "Low": "#2ecc71",
                "Very Low": "#95a5a6",
            }.get(rc.confidence, "#95a5a6")

            rc_rows += f"""
            <tr>
                <td style="font-weight:bold;">#{rc.rank}</td>
                <td><code>{rc.metric}</code></td>
                <td>
                    <span style="background:{confidence_color};color:white;
                          padding:2px 8px;border-radius:4px;font-size:0.85em;">
                        {rc.confidence} ({rc.confidence_pct:.1f}%)
                    </span>
                </td>
                <td>{rc.final_score:.4f}</td>
                <td>{', '.join(rc.downstream_effects[:4]) or '—'}</td>
            </tr>
            """

        # Anomaly scores table
        anomaly_rows = ""
        for metric, score in sorted(anomaly_scores.items(), key=lambda x: -x[1]):
            bg = "#fee2e2" if score >= 1.0 else "#f0fdf4"
            anomaly_rows += f"""
            <tr style="background:{bg};">
                <td><code>{metric}</code></td>
                <td style="text-align:right;">{score:.4f}</td>
                <td>{'⚠️ Anomalous' if score >= 1.0 else '✓ Normal'}</td>
            </tr>
            """

        # Causal edges table
        edge_rows = ""
        for e in causal_summary.get("edges", []):
            edge_rows += f"""
            <tr>
                <td><code>{e['cause']}</code></td>
                <td>-></td>
                <td><code>{e['effect']}</code></td>
                <td style="text-align:right;">{e['p_value']:.4f}</td>
                <td style="text-align:right;">{e['strength']:.3f}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RCA Report — {iid}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          margin: 0; background: #f8fafc; color: #1e293b; }}
  .container {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; color: #475569; border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px; margin-top: 32px; }}
  .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px;
           font-size: 0.9rem; }}
  th {{ background: #1e3a5f; color: white; text-align: left; padding: 8px 12px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }}
  tr:hover {{ background: #f1f5f9; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px;
          font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px;
             font-size:0.8rem; font-weight:600; color:white; }}
  .card {{ background: white; border-radius: 8px; padding: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); margin-bottom: 24px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                   gap: 16px; margin-bottom: 24px; }}
  .stat {{ background: white; border-radius: 8px; padding: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); text-align: center; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: #1e3a5f; }}
  .stat-label {{ font-size: 0.85rem; color: #64748b; }}
</style>
</head>
<body>
<div class="container">
  <h1>🔍 Root Cause Analysis Report</h1>
  <div class="meta">Incident: <strong>{iid}</strong> &nbsp;|&nbsp;
       Generated: <strong>{now}</strong> &nbsp;|&nbsp;
       Duration: <strong>{failure_duration_min} min</strong></div>

  <div class="summary-grid">
    <div class="stat">
      <div class="stat-value">{sum(1 for s in anomaly_scores.values() if s >= 1.0)}</div>
      <div class="stat-label">Anomalous Metrics</div>
    </div>
    <div class="stat">
      <div class="stat-value">{causal_summary.get('total_edges', 0)}</div>
      <div class="stat-label">Causal Edges Found</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(ranked)}</div>
      <div class="stat-label">Root Cause Candidates</div>
    </div>
  </div>

  <div class="card">
    <h2>🎯 Top Root Cause Candidates</h2>
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Metric</th><th>Confidence</th>
          <th>Score</th><th>Downstream Effects</th>
        </tr>
      </thead>
      <tbody>{rc_rows or '<tr><td colspan="5">No root causes identified.</td></tr>'}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>📊 Anomaly Scores</h2>
    <table>
      <thead>
        <tr><th>Metric</th><th>Anomaly Score</th><th>Status</th></tr>
      </thead>
      <tbody>{anomaly_rows or '<tr><td colspan="3">No anomaly data.</td></tr>'}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>🔗 Causal Graph Edges</h2>
    <table>
      <thead>
        <tr><th>Cause</th><th></th><th>Effect</th><th>p-value</th><th>Strength</th></tr>
      </thead>
      <tbody>{edge_rows or '<tr><td colspan="5">No causal edges found.</td></tr>'}</tbody>
    </table>
    <p style="margin-top:12px;color:#64748b;font-size:0.85rem;">
      Potential root causes (graph sources):
      <strong>{', '.join(causal_summary.get('potential_root_causes', [])) or 'N/A'}</strong>
      &nbsp;|&nbsp;
      Symptoms (graph sinks):
      <strong>{', '.join(causal_summary.get('symptoms', [])) or 'N/A'}</strong>
    </p>
  </div>

  <div class="card">
    <h2>💡 Recommendations</h2>
    <pre style="margin:0;white-space:pre-wrap;font-family:inherit;
                font-size:0.9rem;">{self._generate_recommendations(ranked).strip()}</pre>
  </div>
</div>
</body>
</html>"""

        return html

    # ------------------------------------------------------------------

    def visualize_causal_graph_ascii(
        self,
        causal_graph: nx.DiGraph,
        top_root_cause: Optional[str] = None,
        max_nodes: int = 15,
    ) -> str:
        """
        ASCII art representation of the causal DAG.

        Uses NetworkX topological sort to produce a level-based layout.
        Works without any plotting library.

        Args:
            causal_graph: The directed causal graph from CausalInferenceEngine
            top_root_cause: If provided, marked with [RC] in the output
            max_nodes: Truncate display if graph has more nodes

        Returns:
            Multi-line ASCII string
        """
        if causal_graph.number_of_nodes() == 0:
            return "  (empty causal graph — no anomalous metrics above threshold)\n"

        lines: List[str] = ["\n  === CAUSAL GRAPH (ASCII) ===\n"]

        # Try topological sort for level-based display
        try:
            topo = list(nx.topological_sort(causal_graph))
        except nx.NetworkXUnfeasible:
            topo = list(causal_graph.nodes())

        topo = topo[:max_nodes]
        if causal_graph.number_of_nodes() > max_nodes:
            lines.append(
                f"  (showing first {max_nodes} of "
                f"{causal_graph.number_of_nodes()} nodes)\n"
            )

        for node in topo:
            rc_tag = " [ROOT CAUSE]" if node == top_root_cause else ""
            score = causal_graph.nodes[node].get("anomaly_score", 0)
            successors = list(causal_graph.successors(node))

            lines.append(f"  ● {node}{rc_tag}  (score={score:.3f})")
            for succ in successors:
                edge_data = causal_graph[node][succ]
                strength = edge_data.get("strength", 0)
                lag = edge_data.get("lag", "?")
                lines.append(
                    f"      └─-> {succ}  "
                    f"(strength={strength:.3f}, lag={lag} steps)"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_recommendations(self, ranked: List) -> str:
        """Generate text recommendations based on ranked root causes."""
        if not ranked:
            return (
                "  1. Collect more data to enable root cause identification.\n"
                "  2. Check data ingestion pipeline for missing metrics.\n"
                "  3. Ensure sufficient normal-behavior baseline data exists.\n"
            )

        top = ranked[0]
        lines: List[str] = []

        lines.append(
            f"  IMMEDIATE (within 5 minutes):\n"
            f"    — Investigate {top.metric}: this is the highest-confidence root cause.\n"
            f"    — Check recent deployments and configuration changes.\n"
        )

        if top.downstream_effects:
            lines.append(
                f"  SHORT-TERM (within 1 hour):\n"
                f"    — Monitor and mitigate downstream effects: "
                f"{', '.join(top.downstream_effects[:5])}.\n"
                f"    — Consider rolling back recent changes if {top.metric} "
                f"was affected by a deployment.\n"
            )

        lines.append(
            f"  LONG-TERM (within 1 week):\n"
            f"    — Add alerting for {top.metric} to catch issues earlier.\n"
            f"    — Review system capacity and scaling policies.\n"
            f"    — Run post-mortem to document findings for future reference.\n"
        )

        if len(ranked) > 1:
            lines.append(
                f"  PARALLEL INVESTIGATION:\n"
                f"    — Consider investigating #{ranked[1].rank} ({ranked[1].metric}) "
                f"as an alternative root cause.\n"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Build a minimal fake results dict to test the report generator
    from dataclasses import dataclass, field as dc_field
    from typing import List as L, Dict as D

    @dataclass
    class FakeRC:
        metric: str
        rank: int
        final_score: float
        confidence: str
        confidence_pct: float
        factor_scores: D = dc_field(default_factory=dict)
        downstream_effects: L = dc_field(default_factory=list)
        justification: str = ""

    fake_ranked = [
        FakeRC(
            metric="db_connections_active",
            rank=1,
            final_score=0.87,
            confidence="High",
            confidence_pct=87.0,
            factor_scores={"causal_outflow": 0.9, "causal_inflow": 0.8, "temporal": 1.0, "severity": 0.7, "event": 0.0},
            downstream_effects=["api_latency_p95_ms", "error_rate_percent"],
            justification="DB connections explain the latency spike.",
        ),
        FakeRC(
            metric="api_latency_p95_ms",
            rank=2,
            final_score=0.61,
            confidence="Medium",
            confidence_pct=61.0,
            factor_scores={"causal_outflow": 0.5, "causal_inflow": 0.4, "temporal": 0.8, "severity": 0.9, "event": 0.0},
            downstream_effects=["error_rate_percent"],
            justification="High p95 latency is likely a downstream symptom.",
        ),
    ]

    fake_results = {
        "ranked_causes": fake_ranked,
        "anomaly_scores": {
            "db_connections_active": 1.8,
            "api_latency_p95_ms": 2.1,
            "error_rate_percent": 1.6,
            "cpu_utilization": 0.3,
        },
        "anomaly_first_detected": {
            "db_connections_active": "2024-01-01 12:00:00 UTC",
            "api_latency_p95_ms": "2024-01-01 12:05:00 UTC",
            "error_rate_percent": "2024-01-01 12:15:00 UTC",
        },
        "causal_graph_summary": {
            "edges": [
                {"cause": "db_connections_active", "effect": "api_latency_p95_ms",
                 "p_value": 0.001, "lag_timesteps": 2, "strength": 0.85},
                {"cause": "api_latency_p95_ms", "effect": "error_rate_percent",
                 "p_value": 0.01, "lag_timesteps": 3, "strength": 0.70},
            ],
            "potential_root_causes": ["db_connections_active"],
            "symptoms": ["error_rate_percent"],
            "total_nodes": 3,
            "total_edges": 2,
        },
        "timing": {
            "anomaly_detection_sec": 1.2,
            "causal_inference_sec": 8.5,
            "ranking_sec": 0.1,
        },
    }

    rg = ReportGenerator()

    print(rg.generate_executive_summary(
        fake_results,
        failure_duration_min=45,
        estimated_impact_dollars=25000,
        incident_id="INC-2024-001",
    ))

    print(rg.generate_technical_report(fake_results))

    json_str = rg.generate_json_report(fake_results, incident_id="INC-2024-001")
    print("JSON report (first 500 chars):")
    print(json_str[:500])

    # Save HTML
    html = rg.generate_html_report(fake_results, incident_id="INC-2024-001")
    with open("rca_report_test.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n✅ HTML report written to rca_report_test.html")
