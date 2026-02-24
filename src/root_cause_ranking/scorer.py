import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd

class RootCauseRanker:
    """
    Ranks candidate root causes using a composite score
    based on causal graph properties, temporal precedence,
    and anomaly severity.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default heuristic weights matching the documentation
        self.weights = weights or {
            'causal_outflow': 0.30,
            'causal_inflow_penalty': 0.15,
            'temporal_priority': 0.25,
            'anomaly_severity': 0.15,
            'event_correlation': 0.15
        }
    
    def rank(self, causal_graph: nx.DiGraph, anomaly_scores: Dict[str, float], 
             anomaly_first_detected: Dict[str, pd.Timestamp], event_correlations: List[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Score and rank each anomalous metric as a potential root cause.
        
        Returns: sorted list of (metric, score, explanation_dict) tuples
        """
        if len(causal_graph.nodes) == 0:
            return []
            
        candidates = []
        
        # Compute PageRank on reversed graph (high rank = highly influential *cause*)
        reversed_graph = causal_graph.reverse()
        try:
            # We use undirected/unweighted pagerank if strength isn't fully reliable
            pagerank_scores = nx.pagerank(reversed_graph)
        except Exception:
            # Fallback if graph is empty or disconnected in a weird way
            pagerank_scores = {n: 1.0/len(causal_graph) for n in causal_graph.nodes}
        
        all_anomaly_times = list(anomaly_first_detected.values())
        if all_anomaly_times:
            earliest = min(all_anomaly_times)
            latest = max(all_anomaly_times)
            time_range = (latest - earliest).total_seconds() or 1.0
        else:
            time_range = 1.0
            earliest = None
            
        max_out = max([causal_graph.out_degree(n) for n in causal_graph.nodes] + [1])
        max_in = max([causal_graph.in_degree(n) for n in causal_graph.nodes] + [1])
        
        for metric in causal_graph.nodes:
            score_components = {}
            
            # 1. Causal outflow: how many things does this cause directly?
            out_degree = causal_graph.out_degree(metric)
            score_components['causal_outflow'] = out_degree / max_out
            
            # 2. Causal inflow penalty: true root causes have 0 incoming edges
            in_degree = causal_graph.in_degree(metric)
            score_components['causal_inflow_penalty'] = 1.0 - (in_degree / max_in)
            
            # 3. Temporal priority: earlier anomalies are more likely root causes
            if earliest and metric in anomaly_first_detected:
                time_offset = (anomaly_first_detected[metric] - earliest).total_seconds()
                score_components['temporal_priority'] = 1.0 - (time_offset / time_range)
            else:
                score_components['temporal_priority'] = 0.0
            
            # 4. Anomaly severity
            score_components['anomaly_severity'] = anomaly_scores.get(metric, 0.0)
            
            # 5. Event correlation (e.g., deployment/config change)
            if event_correlations:
                relevant = [ec for ec in event_correlations if ec.get('affected_metric') == metric]
                if relevant:
                    score_components['event_correlation'] = relevant[0].get('correlation_score', 0.0)
                else:
                    score_components['event_correlation'] = 0.0
            else:
                score_components['event_correlation'] = 0.0
            
            # Composite score using weights
            heuristic_score = sum(
                self.weights[k] * score_components[k] for k in self.weights
            )
            
            # Incorporate PageRank (70% heuristic, 30% PageRank influence)
            final_score = 0.7 * heuristic_score + 0.3 * pagerank_scores.get(metric, 0.0)
            
            explanation = {
                'components': score_components,
                'out_edges': list(causal_graph.successors(metric)),
                'in_edges': list(causal_graph.predecessors(metric)),
                'pagerank': pagerank_scores.get(metric, 0.0)
            }
            
            candidates.append((metric, final_score, explanation))
            
        # Sort descending by final score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

if __name__ == "__main__":
    print("Testing Root Cause Ranker...")
    
    # Create mock graph A -> B -> C
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    
    scores = {'A': 0.9, 'B': 0.7, 'C': 0.8}
    
    times = {
        'A': pd.Timestamp('2024-01-01 10:00:00'),
        'B': pd.Timestamp('2024-01-01 10:05:00'),
        'C': pd.Timestamp('2024-01-01 10:10:00')
    }
    
    ranker = RootCauseRanker()
    ranked = ranker.rank(G, scores, times)
    
    for metric, score, info in ranked:
         print(f"Metric: {metric} | Score: {score:.3f} | In: {info['in_edges']} Out: {info['out_edges']}")
