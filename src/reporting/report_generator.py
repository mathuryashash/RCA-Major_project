import networkx as nx
from typing import Dict, List, Tuple
from datetime import datetime

class ReportGenerator:
    """
    Generates human-readable RCA reports from ranked anomalies,
    confidence scores, and the causal graph constraint topology. 
    """
    
    def __init__(self):
        pass
        
    def generate_report(self, 
                        incident_id: str,
                        ranked_candidates: List[Tuple[str, float, Dict]], 
                        causal_graph: nx.DiGraph,
                        anomaly_times: Dict[str, datetime]) -> str:
        """
        Creates a structured markdown report for the given incident details.
        """
        if not ranked_candidates:
            return "No anomalies detected or no causal relationships found."
            
        top_cause, top_score, top_explanation = ranked_candidates[0]
        confidence_percent = min(100.0, top_score * 100)

        # Without a surviving edge there is no causal evidence: graph influence
        # is uniform and outflow is zero for every metric, so the order comes
        # from timing and severity alone. Announcing a "primary root cause"
        # then contradicted the evidence section directly below it, and the
        # top two candidates could sit four ten-thousandths apart.
        has_causal_evidence = causal_graph is not None and causal_graph.number_of_edges() > 0
        runner_up = ranked_candidates[1][1] if len(ranked_candidates) > 1 else None
        too_close = runner_up is not None and abs(top_score - runner_up) < 0.01

        if has_causal_evidence:
            headline = [
                f"**Primary Root Cause Identified:** `{top_cause}`",
                f"**System Confidence:** {confidence_percent:.1f}%",
            ]
        else:
            headline = [
                f"**Leading Correlated Metric:** `{top_cause}` "
                f"(ranked {confidence_percent:.1f}% on timing and severity only)",
                "**No causal evidence.** No causal edge survived, so this is not a "
                "root cause claim -- it is the metric that deviated earliest and "
                "hardest.",
            ]
            if too_close:
                headline.append(
                    f"**Ranking is not meaningful here:** `{ranked_candidates[1][0]}` "
                    f"scores {ranked_candidates[1][1] * 100:.1f}%, within a hundredth "
                    "of the leader, so their order is arbitrary."
                )

        report = [
            f"# Root Cause Analysis Report: Incident {incident_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            *headline,
            f"**Start Time:** {anomaly_times.get(top_cause, 'Unknown')}",
            "",
            "## Causal Chain Summary",
            "Based on structural and temporal analysis, the failure propagated as follows:"
        ]
        
        # Build causal chain textual summary for the top cause
        downstream = top_explanation.get('out_edges', [])
        if not downstream:
            report.append(f"- Anomaly detected in `{top_cause}`, but no downstream propagation was observed.")
        else:
            report.append(f"1. **Origin:** An issue originated at `{top_cause}`.")
            if len(downstream) == 1:
                report.append(f"2. **Propagation:** This directly caused abnormal behavior in `{downstream[0]}`.")
            else:
                report.append(f"2. **Propagation:** This resulted in cascaded anomalies across `{', '.join(downstream)}`.")
        
        report.extend([
            "",
            "## Detailed Evidence Scoring",
            "The system concluded this via the following heuristic components:"
        ])
        
        comps = top_explanation.get('components', {})
        report.extend([
            f"- **Temporal Priority**: {comps.get('temporal_priority', 0.0):.2f} (How early the anomaly occurred)",
            f"- **Severity Score**: {comps.get('anomaly_severity', 0.0):.2f} (Magnitude of normal deviation)",
            f"- **Causal Outflow**: {comps.get('causal_outflow', 0.0):.2f} (Number of affected downstream systems)",
            f"- **Graph Influence (PageRank)**: {top_explanation.get('pagerank', 0.0):.2f} (Topology centrality)",
        ])
        
        if len(ranked_candidates) > 1:
            report.extend([
                "",
                "## Alternative Root Causes Considered"
            ])
            for i in range(1, min(4, len(ranked_candidates))):
                alt_cause, alt_score, _ = ranked_candidates[i]
                report.append(f"{i}. `{alt_cause}` (Score: {alt_score * 100:.1f}%)")
                
        return "\n".join(report)

if __name__ == "__main__":
    generator = ReportGenerator()
    
    # Dummy data
    G = nx.DiGraph()
    G.add_edge('Database_Queries', 'API_Gateway')
    
    ranked = [
        ('Database_Queries', 0.85, {
            'out_edges': ['API_Gateway'], 
            'components': {'temporal_priority': 1.0, 'anomaly_severity': 0.9, 'causal_outflow': 1.0},
            'pagerank': 0.4
        }),
        ('API_Gateway', 0.45, {
            'out_edges': [], 
            'components': {'temporal_priority': 0.0, 'anomaly_severity': 0.8, 'causal_outflow': 0.0},
            'pagerank': 0.1
        })
    ]
    
    import pandas as pd
    times = {
        'Database_Queries': pd.Timestamp('2024-01-01 10:00:00'),
        'API_Gateway': pd.Timestamp('2024-01-01 10:05:00')
    }
    
    res = generator.generate_report("INC-001", ranked, G, times)
    print(res)
