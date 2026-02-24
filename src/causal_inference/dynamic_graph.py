import networkx as nx
import logging
from typing import Dict, List, Tuple
from causal_inference.jaeger_connector import JaegerConnector

class DynamicGraphGenerator:
    """
    Combines static/statistical Granger Causality with dynamic architecture topology 
    from distributed tracing to validate causal edges.
    
    PRD Section 1.1.1 & 1.1.3:
    Using real service topologies prevents impossible causal chains 
    (e.g. Database causing a Load Balancer failure when there's no path between them).
    """

    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        self.jaeger = JaegerConnector(jaeger_url)
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def extract_service_from_metric(self, metric_name: str) -> str:
        """
        Heuristic to extract a service name from a metric name.
        Example: 'api_server_cpu_utilization' -> 'api_server'
        In a real production environment, metrics usually have 'app' or 'service' labels.
        This represents that mapping.
        """
        # A simple heuristic for demo/testing purposes
        parts = metric_name.split('_')
        if len(parts) >= 2:
           return f"{parts[0]}_{parts[1]}"
        return "unknown_service"

    def is_path_possible(
        self, 
        topology: nx.DiGraph, 
        source_service: str, 
        target_service: str
    ) -> bool:
        """
        Checks if the architectural topology permits causality to flow from source to target.
        Note: Causality often flows *up* the call stack (Database fails -> API Server fails).
        So we check if target_service calls source_service, OR if source calls target.
        """
        if source_service == target_service:
            return True # Same service, internal metric causality is always possible
            
        if not topology.has_node(source_service) or not topology.has_node(target_service):
             # If telemetry is missing for a node, we default to True to avoid dropping valid statistical edges
             return True
             
        # Check downstream flow (A calls B)
        has_downstream = nx.has_path(topology, source_service, target_service)
        # Check upstream flow (A is called by B) - This is the most common failure direction
        has_upstream = nx.has_path(topology, target_service, source_service)
        
        return has_downstream or has_upstream

    def refine_causal_graph(
        self, 
        statistical_graph: nx.DiGraph, 
        lookback_minutes: int = 60
    ) -> nx.DiGraph:
        """
        Takes a causal graph built by Granger Causality and prunes edges that are 
        architecturally impossible according to recent distributed traces.
        """
        self.logger.info("Fetching real-time topology from Jaeger...")
        topology = self.jaeger.build_dependency_graph(lookback_minutes)
        
        if len(topology.nodes) == 0:
            self.logger.warning("No topology data found. Falling back to purely statistical graph.")
            return statistical_graph
            
        refined_graph = statistical_graph.copy()
        edges_to_remove = []
        
        for u, v in refined_graph.edges():
            service_u = self.extract_service_from_metric(u)
            service_v = self.extract_service_from_metric(v)
            
            if not self.is_path_possible(topology, service_u, service_v):
                self.logger.info(f"Pruning architecturally impossible edge: {u} -> {v}")
                edges_to_remove.append((u, v))
                
        for u, v in edges_to_remove:
            refined_graph.remove_edge(u, v)
            
        self.logger.info(f"Refinement complete. Pruned {len(edges_to_remove)} edges based on topology.")
        return refined_graph

if __name__ == "__main__":
    generator = DynamicGraphGenerator()
    
    # Create a dummy statistical graph
    stat_g = nx.DiGraph()
    stat_g.add_edge("api_server_latency", "db_server_connections", strength=0.9)
    stat_g.add_edge("frontend_app_errors", "db_server_connections", strength=0.8)
    
    # Normally this connects to Jaeger. We inject a mock topology here.
    mock_topology = nx.DiGraph()
    mock_topology.add_edge("frontend_app", "api_server") # frontend calls api
    mock_topology.add_edge("api_server", "db_server")    # api calls db
    # Notice frontend does NOT directly call db
    
    # Override the fetcher for the test
    generator.jaeger.build_dependency_graph = lambda x: mock_topology
    
    print(f"Original edges: {stat_g.edges()}")
    refined_g = generator.refine_causal_graph(stat_g)
    print(f"Refined edges: {refined_g.edges()}")
