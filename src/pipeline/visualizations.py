"""Figure builders — no Qt imports. Pure functions: graph/data in, Figure out.

Matplotlib rather than Plotly, and the reason is size rather than taste.
Plotly renders to HTML, which needs a browser to display, which meant shipping
QtWebEngine: 258 MB of DLLs plus 29 MB of resources, 53 MB of translations and
a 20 MB software OpenGL fallback. Roughly a third of the installed application
existed to draw two charts. Matplotlib draws them into a native Qt widget for
28 MB, keeps pan and zoom through the standard navigation toolbar, and loses
only hover tooltips.

The palette matches the desktop theme deliberately. An earlier version of the
empty-graph case inherited Plotly's white default, so the single most important
honest state -- no causal edge survived -- rendered as a bright rectangle in a
dark application and read as a broken chart rather than as a finding.
"""

from typing import Dict

import networkx as nx
import pandas as pd
from matplotlib.figure import Figure

#: Matches desktop/theme.py. Figures are part of the application, not guests.
SURFACE = "#151a2e"
TEXT = "#e2e8f0"
MUTED = "#8b949e"
ROOT_CAUSE = "#ff4757"
SOURCE = "#ffa502"
INTERMEDIATE = "#70a1ff"
EDGE = "#667eea"


def _figure() -> Figure:
    fig = Figure(figsize=(7.2, 4.4), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    return fig


def draw_causal_graph(G: nx.DiGraph, root_cause_metric: str) -> Figure:
    """The surviving causal edges, or an explicit statement that there are none."""
    fig = _figure()
    axes = fig.add_subplot(111)
    axes.set_facecolor(SURFACE)

    if len(G.nodes) == 0:
        axes.set_title("No causal link was established", color=TEXT, fontsize=13)
        axes.text(
            0.5, 0.5,
            "Either none exists in this data, or the window was too short to\n"
            "test one. The ranking beside this reflects timing and severity only.",
            ha="center", va="center", color=MUTED, fontsize=10,
            transform=axes.transAxes,
        )
        axes.set_axis_off()
        fig.tight_layout()
        return fig

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:                       # noqa: BLE001 - layout is cosmetic
        pos = nx.spring_layout(G, seed=1, k=1.5)

    for source, target, data in G.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        strength = float(data.get("strength", 0.0) or 0.0)
        width = 1.0 + min(strength * 4.0, 6.0)
        alpha = min(0.35 + strength, 1.0)
        axes.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=EDGE, alpha=alpha,
                            linewidth=width, shrinkA=14, shrinkB=16),
        )
        axes.text((x0 + x1) / 2, (y0 + y1) / 2, f"lag={data.get('lag', '?')}",
                  fontsize=7, color=MUTED, ha="center", va="center")

    def category(node: str) -> str:
        if node == root_cause_metric:
            return "root"
        return "source" if G.in_degree(node) == 0 else "intermediate"

    styles = {
        "root": (ROOT_CAUSE, 320, "Root cause"),
        "source": (SOURCE, 200, "Source node"),
        "intermediate": (INTERMEDIATE, 150, "Intermediate"),
    }
    for name, (colour, size, label) in styles.items():
        nodes = [n for n in G.nodes if category(n) == name]
        if not nodes:
            continue
        axes.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes],
                     s=size, c=colour, edgecolors="white", linewidths=1.5,
                     zorder=3, label=label)

    for node in G.nodes:
        x, y = pos[node]
        axes.annotate(node, (x, y), textcoords="offset points", xytext=(0, 13),
                      ha="center", fontsize=8, color=TEXT, zorder=4)

    axes.set_title("Causal dependency graph", color=TEXT, fontsize=13)
    legend = axes.legend(loc="upper right", fontsize=8, framealpha=0.0,
                         labelcolor=TEXT)
    if legend is not None:
        legend.get_frame().set_facecolor(SURFACE)
    axes.set_axis_off()
    axes.margins(0.18)
    fig.tight_layout()
    return fig


def build_timeline_figure(
    incident_scaled: pd.DataFrame,
    anomaly_scores: Dict[str, float],
    anomaly_times: Dict[str, pd.Timestamp],
    top_n: int = 5,
) -> Figure:
    """Top-N anomalous metrics over time, with the moment each first crossed."""
    fig = _figure()
    axes = fig.add_subplot(111)
    axes.set_facecolor(SURFACE)

    ranked = sorted(anomaly_scores, key=anomaly_scores.get, reverse=True)[:top_n]
    columns = [c for c in ranked if c in incident_scaled.columns]

    if not columns:
        axes.set_title("No anomalous metrics in this window", color=TEXT, fontsize=13)
        axes.set_axis_off()
        fig.tight_layout()
        return fig

    series = incident_scaled.set_index("timestamp")[columns]
    for column in columns:
        axes.plot(series.index, series[column], linewidth=1.4, label=column)

    # The left-to-right order of these lines is exactly what the ranking treats
    # as temporal priority, so they are worth drawing rather than describing.
    for column in columns:
        crossed = anomaly_times.get(column)
        if crossed is not None:
            axes.axvline(crossed, color=ROOT_CAUSE, linestyle="--",
                         linewidth=1.0, alpha=0.75)

    axes.set_title("Most anomalous metrics (scaled 0–1)", color=TEXT, fontsize=13)
    axes.set_xlabel("Time", color=MUTED, fontsize=9)
    axes.set_ylabel("Scaled value", color=MUTED, fontsize=9)
    axes.tick_params(colors=MUTED, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color("#2c3138")
    axes.grid(True, alpha=0.15, linestyle=":")
    legend = axes.legend(loc="upper left", fontsize=8, framealpha=0.0,
                         labelcolor=TEXT)
    if legend is not None:
        legend.get_frame().set_facecolor(SURFACE)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
