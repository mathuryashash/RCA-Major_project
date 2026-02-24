import logging
import requests
import networkx as nx
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class JaegerConnector:
    """
    Connects to a Jaeger Tracing backend to dynamically extract Service Dependency Topology.
    PRD Section 1.1.1 (Data Ingestion - Distributed Tracing)
    """
    
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        self.base_url = jaeger_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def fetch_services(self) -> List[str]:
        """Fetch all known services from Jaeger."""
        try:
            response = requests.get(f"{self.api_url}/services", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch services from Jaeger: {e}")
            return []

    def fetch_traces(self, service: str, lookback_minutes: int = 60, limit: int = 100) -> List[Dict]:
        """Fetch recent traces for a specific service."""
        try:
            # Jaeger expects timestamps in microseconds
            end_time = int(datetime.now().timestamp() * 1000000)
            start_time = int((datetime.now() - timedelta(minutes=lookback_minutes)).timestamp() * 1000000)
            
            params = {
                'service': service,
                'start': start_time,
                'end': end_time,
                'limit': limit
            }
            
            response = requests.get(f"{self.api_url}/traces", params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('data', [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch traces for service {service}: {e}")
            return []

    def build_dependency_graph(self, lookback_minutes: int = 60, sample_limit: int = 50) -> nx.DiGraph:
        """
        Builds a directed graph representing service-to-service dependencies by analyzing traces.
        Nodes: Services
        Edges: Caller -> Callee (with call count and avg duration weights)
        """
        G = nx.DiGraph()
        services = self.fetch_services()
        
        self.logger.info(f"Building dynamic topology from {len(services)} services...")
        
        # We need a map to track edge statistics
        edge_stats = {} # (caller, callee) -> {'count': int, 'duration_sum': float}
        
        for service in services:
            # We don't want to query traces for internal Jaeger components
            if service.startswith('jaeger-'): 
                continue
                
            traces = self.fetch_traces(service, lookback_minutes, sample_limit)
            
            for trace in traces:
                spans = trace.get('spans', [])
                
                # Create maps for quick lookup within the trace
                span_to_service = {}
                for span in spans:
                    # Resolve process ID to service name
                    process_id = span.get('processID')
                    processes = trace.get('processes', {})
                    if process_id in processes:
                        span_to_service[span['spanID']] = processes[process_id]['serviceName']
                
                for span in spans:
                    span_id = span['spanID']
                    callee = span_to_service.get(span_id)
                    
                    # Look for parent spans to define directed Caller -> Callee relationship
                    references = span.get('references', [])
                    for ref in references:
                        if ref.get('refType') == 'CHILD_OF':
                            parent_id = ref.get('spanID')
                            caller = span_to_service.get(parent_id)
                            
                            if caller and callee and caller != callee:
                                edge = (caller, callee)
                                duration = span.get('duration', 0) # in microseconds
                                
                                if edge not in edge_stats:
                                    edge_stats[edge] = {'count': 0, 'duration_sum': 0}
                                    
                                edge_stats[edge]['count'] += 1
                                edge_stats[edge]['duration_sum'] += duration

        # Populate the networkx Graph
        for (caller, callee), stats in edge_stats.items():
            avg_duration_ms = (stats['duration_sum'] / stats['count']) / 1000.0
            
            G.add_edge(
                caller, 
                callee, 
                weight=stats['count'],       # How often it's called
                avg_latency_ms=avg_duration_ms # Average speed of interaction
            )
            
        self.logger.info(f"Topology built: {len(G.nodes)} nodes, {len(G.edges)} edges.")
        return G

if __name__ == "__main__":
    connector = JaegerConnector()
    print("Testing Jaeger connection...")
    
    # Needs a real Jaeger instance running at localhost:16686 to yield data
    services = connector.fetch_services()
    if services:
        print(f"Found services: {services}")
        G = connector.build_dependency_graph()
        print(f"Graph nodes: {list(G.nodes)}")
        print(f"Graph edges: {list(G.edges(data=True))}")
    else:
        print("No services found or unable to connect. (Ensure Jaeger is running locally)")
