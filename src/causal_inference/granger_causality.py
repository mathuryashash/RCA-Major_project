import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple
import networkx as nx

class GrangerCausalityAnalyzer:
    """
    Implements causal inference using pairwise Granger causality tests.
    X 'Granger-causes' Y if past values of X improve the prediction of Y.
    """
    
    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance_level
    
    def test_causality(self, df: pd.DataFrame, anomalous_metrics: List[str]) -> Dict[Tuple[str, str], Dict]:
        """
        Compute pairwise Granger causality between anomalous metrics.
        Returns: {(cause, effect): {'p_value': ..., 'optimal_lag': ..., 'strength': ...}}
        """
        results = {}
        
        for cause_metric in anomalous_metrics:
            for effect_metric in anomalous_metrics:
                if cause_metric == effect_metric:
                    continue
                
                try:
                    # Granger causality test expects [y, x] where we test if x causes y
                    # We drop NAs since statsmodels cannot handle them
                    test_data = df[[effect_metric, cause_metric]].dropna()
                    
                    if len(test_data) < self.max_lag * 3:
                        # Not enough data for the test
                        continue
                    
                    gc_result = grangercausalitytests(
                        test_data, 
                        maxlag=self.max_lag, 
                        verbose=False
                    )
                    
                    # Find the best lag (lowest p-value from the SSR F-test)
                    best_lag = None
                    best_p = 1.0
                    for lag, test_dict in gc_result.items():
                        # index 0 is the test results, 'ssr_ftest' is the F-test, index 1 is the p-value
                        p_val = test_dict[0]['ssr_ftest'][1]
                        if p_val < best_p:
                            best_p = p_val
                            best_lag = lag
                    
                    # If statistically significant, record it
                    if best_p < self.significance:
                        results[(cause_metric, effect_metric)] = {
                            'p_value': best_p,
                            'optimal_lag': best_lag,
                            'strength': 1 - best_p  # Higher is stronger causality
                        }
                        
                except Exception as e:
                    # e.g., constant data causing singular matrix errors in OLS
                    pass
        
        return results
    
    def build_causal_graph(self, granger_results: Dict[Tuple[str, str], Dict], 
                           anomaly_scores: Dict[str, float] = None) -> nx.DiGraph:
        """
        Construct a directed causal graph from Granger results.
        Nodes are metrics, edges are causal relationships.
        """
        G = nx.DiGraph()
        
        if anomaly_scores:
            for metric, score in anomaly_scores.items():
                G.add_node(metric, anomaly_score=score)
                
        for (cause, effect), info in granger_results.items():
            # Add nodes if not already present
            if not G.has_node(cause):
                G.add_node(cause)
            if not G.has_node(effect):
                G.add_node(effect)
                
            G.add_edge(
                cause, effect,
                p_value=info['p_value'],
                lag=info['optimal_lag'],
                strength=info['strength']
            )
            
        # Break cycles to form a DAG using temporal precedence or edge strength
        G = self._break_cycles(G)
        return G
        
    def _break_cycles(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove edges that create cycles by dropping the weakest link."""
        # Simple cycle breaking based on edge strength
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                # cycle is a list of nodes e.g. [A, B, C] meaning A->B->C->A
                weakest_edge = None
                weakest_strength = float('inf')
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(u, v):
                        strength = G[u][v].get('strength', 0)
                        if strength < weakest_strength:
                            weakest_strength = strength
                            weakest_edge = (u, v)
                            
                if weakest_edge and G.has_edge(*weakest_edge):
                    G.remove_edge(*weakest_edge)
        except Exception:
            pass
            
        return G

if __name__ == "__main__":
    import numpy as np
    # Quick test
    analyzer = GrangerCausalityAnalyzer(max_lag=3)
    
    # Create synthetic data where A causes B with lag 1
    t = np.arange(100)
    A = np.sin(t) + np.random.normal(0, 0.1, 100)
    B = np.zeros(100)
    B[1:] = A[:-1] * 2 + np.random.normal(0, 0.1, 99)
    
    df = pd.DataFrame({'A': A, 'B': B})
    
    print("Testing Granger Causality Analyzer...")
    res = analyzer.test_causality(df, ['A', 'B'])
    print(f"Results: {res}")
    
    graph = analyzer.build_causal_graph(res)
    print(f"Graph nodes: {graph.nodes()}")
    print(f"Graph edges: {graph.edges(data=True)}")
