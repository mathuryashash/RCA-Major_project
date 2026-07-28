"""Static laptop subsystem constraints for statistical RCA edges.

The desktop collector does not have service traces.  This module therefore
uses an explicit, inspectable map of operating-system subsystems instead of
quietly reaching out to Jaeger or accepting an unknown topology.
"""

from __future__ import annotations

import networkx as nx


SUBSYSTEMS = {
    "cpu": ("cpu_pct", "cpu_pct_max_core", "cpu_freq_mhz", "cpu_freq_ratio"),
    "memory": ("mem_pct", "mem_available_mb", "swap_pct", "swap_used_bytes", "swap_used_delta"),
    "disk": ("disk_read_bps", "disk_write_bps", "disk_busy_pct", "disk_free_pct"),
    "network": ("net_sent_bps", "net_recv_bps"),
    "power": ("battery_pct", "battery_drain_rate", "power_plugged"),
    "process": ("process_count",),
}

# Directed dependency paths.  An edge is retained only when this map permits
# propagation in the observed statistical direction (or within a subsystem).
DEPENDENCIES = (
    ("power", "cpu"), ("power", "memory"), ("power", "disk"),
    ("cpu", "memory"), ("cpu", "disk"), ("cpu", "network"),
    ("memory", "disk"), ("memory", "network"),
    ("disk", "network"), ("process", "cpu"), ("process", "memory"),
    ("process", "disk"), ("process", "network"),
)


class DynamicGraphGenerator:
    """Prune statistically significant edges that contradict laptop topology."""

    def __init__(self) -> None:
        self.topology = nx.DiGraph()
        self.topology.add_edges_from(DEPENDENCIES)

    @staticmethod
    def subsystem_for_metric(metric_name: str) -> str | None:
        for subsystem, metrics in SUBSYSTEMS.items():
            if metric_name in metrics:
                return subsystem
        return None

    def is_path_possible(self, source_metric: str, target_metric: str) -> bool:
        source = self.subsystem_for_metric(source_metric)
        target = self.subsystem_for_metric(target_metric)
        # Unknown metrics have no declared relationship; do not turn a weak
        # statistical guess into a reported causal edge.
        if source is None or target is None:
            return False
        return source == target or nx.has_path(self.topology, source, target)

    def refine_causal_graph(self, statistical_graph: nx.DiGraph) -> nx.DiGraph:
        refined = statistical_graph.copy()
        for source, target in list(refined.edges):
            if not self.is_path_possible(source, target):
                refined.remove_edge(source, target)
        return refined
