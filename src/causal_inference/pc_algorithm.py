import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

class CausalDiscoveryEngine:
    """
    Learns causal graph structure from observed anomaly data utilizing 
    the Peter-Clark (PC) algorithm from the causal-learn library.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def learn_causal_structure_pc(self, df: pd.DataFrame, anomalous_metrics: List[str]) -> nx.DiGraph:
        """
        Uses the PC algorithm with Fisher's Z conditional independence test 
        to learn a causal Directed Acyclic Graph (DAG).
        """
        if len(anomalous_metrics) < 2:
            # Cannot build a causal graph with < 2 nodes
            G = nx.DiGraph()
            for m in anomalous_metrics:
                G.add_node(m)
            return G
            
        # PC algorithm requires complete data without NAs
        data = df[anomalous_metrics].dropna().values
        
        # If dropping NAs leaves too little data, return disconnected graph
        if data.shape[0] < 10:
             G = nx.DiGraph()
             for m in anomalous_metrics:
                 G.add_node(m)
             return G

        # Run PC
        cg = pc(data, alpha=self.alpha, indep_test=fisherz, verbose=False, show_progress=False)
        
        G = nx.DiGraph()
        for metric in anomalous_metrics:
            G.add_node(metric)
            
        # cg.G.graph is the adjacency matrix representing the resulting PAG/DAG
        # The causal-learn adjacency matrix encoding:
        # cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == 1  => j -> i
        # cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == -1 => undirected edge i - j
        
        adj_matrix = cg.G.graph
        n_nodes = len(anomalous_metrics)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    # Directed edge from j to i
                    G.add_edge(anomalous_metrics[j], anomalous_metrics[i])
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                    # Undirected edge
                    G.add_edge(anomalous_metrics[i], anomalous_metrics[j], undirected=True)
                    G.add_edge(anomalous_metrics[j], anomalous_metrics[i], undirected=True)
                    
        return G
        
    def temporal_precedence_filter(self, G: nx.DiGraph, 
                                   anomaly_first_detected: Dict[str, pd.Timestamp]) -> nx.DiGraph:
        """
        Filters causal edges based on temporal precedence.
        Causes MUST occur before or at the same time as their effects.
        """
        edges_to_remove = []
        for u, v in G.edges():
            # If an edge is supposed to be undocumented/undirected, it appears twice.
            # We'll orient it based on time.
            if u in anomaly_first_detected and v in anomaly_first_detected:
                time_u = anomaly_first_detected[u]
                time_v = anomaly_first_detected[v]
                
                # If cause u strictly happened AFTER effect v, the direction is wrong.
                if time_u > time_v:
                    edges_to_remove.append((u, v))
                    
        G.remove_edges_from(edges_to_remove)
        return G


if __name__ == "__main__":
    import numpy as np
    
    # Generate some confounded/causal data for test
    n_samples = 500
    x = np.random.randn(n_samples)
    y = 0.5 * x + np.random.randn(n_samples) * 0.1 # x -> y
    z = 0.8 * y + np.random.randn(n_samples) * 0.1 # y -> z
    
    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})
    metrics = ['X', 'Y', 'Z']
    
    engine = CausalDiscoveryEngine(alpha=0.05)
    print("Running PC Algorithm...")
    graph = engine.learn_causal_structure_pc(df, metrics)
    
    print("Learned Graph Edges:", graph.edges(data=True))
    
    # Test temporal precedence
    # Assume we detected Z first (wrong), Y next, X last
    timestamps = {
        'Z': pd.Timestamp('2024-01-01 10:00:00'),
        'Y': pd.Timestamp('2024-01-01 10:05:00'),
        'X': pd.Timestamp('2024-01-01 10:10:00')
    }
    
    filtered_graph = engine.temporal_precedence_filter(graph, timestamps)
    print("Filtered Edges (Should drop violations):", filtered_graph.edges())
