"""
Plotly figure builders — no Streamlit or Qt imports. Pure functions:
graph/data in, plotly.graph_objects.Figure out. Shared by dashboard.py
(if it chooses to import them) and the PySide6 desktop app.
"""

from typing import Dict, List

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def draw_causal_graph(G: nx.DiGraph, root_cause_metric: str) -> go.Figure:
    """Return a Plotly figure for the causal graph with arrows and legend."""
    if len(G.nodes) == 0:
        # Same dark ground as every other figure. Without it Plotly defaults to
        # white, so the single most important honest state -- "no causal edge
        # survived" -- rendered as a bright rectangle in a dark application and
        # read as a broken chart rather than as a finding.
        return go.Figure().update_layout(
            title=dict(text="No causal link was established", font=dict(size=16)),
            annotations=[dict(
                text=("Either none exists in this data, or the window was too "
                      "short to test one.<br>The ranking beside this reflects "
                      "timing and severity only."),
                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=13, color="#8b949e"), align="center",
            )],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            paper_bgcolor="#151a2e", plot_bgcolor="#151a2e",
            font=dict(color="#e2e8f0"),
        )

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=1, k=1.5)

    fig = go.Figure()

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lag = d.get("lag", "?")
        strength = d.get("strength", 0.0)
        edge_width = 1.0 + min(strength * 5.0, 10.0)
        opacity = min(0.3 + strength, 1.0)
        edge_color = f"rgba(102, 126, 234, {opacity})"

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=edge_width, color=edge_color),
            hoverinfo="text",
            hovertext=f"{u} → {v} (lag={lag}, str={strength:.3f})",
            showlegend=False,
        ))
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=edge_width,
            arrowcolor=edge_color,
        )
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_annotation(
            x=mx, y=my, text=f"lag={lag}", showarrow=False,
            font=dict(size=9, color="rgba(180, 190, 220, 0.7)"),
        )

    node_categories = {}
    for n in G.nodes:
        if n == root_cause_metric:
            node_categories[n] = "root_cause"
        elif G.in_degree(n) == 0:
            node_categories[n] = "source"
        else:
            node_categories[n] = "intermediate"

    color_map = {"root_cause": "#ff4757", "source": "#ffa502", "intermediate": "#70a1ff"}
    size_map = {"root_cause": 34, "source": 26, "intermediate": 22}

    for cat, cat_label in [("root_cause", "🔴 Root Cause"), ("source", "🟠 Source Node"), ("intermediate", "🔵 Intermediate")]:
        cat_nodes = [n for n in G.nodes if node_categories[n] == cat]
        if not cat_nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in cat_nodes], y=[pos[n][1] for n in cat_nodes],
            mode="markers+text", text=cat_nodes, textposition="top center",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertext=[f"<b>{n}</b><br>Out-degree: {G.out_degree(n)}<br>In-degree: {G.in_degree(n)}" for n in cat_nodes],
            hoverinfo="text", name=cat_label,
            marker=dict(size=[size_map[cat]] * len(cat_nodes), color=color_map[cat],
                        line=dict(width=3, color="white"), symbol="circle"),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Causal Dependency Graph", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520,
        paper_bgcolor="#151a2e",
        plot_bgcolor="#151a2e",
        font=dict(color="#e2e8f0"),
    )
    return fig


def build_timeline_figure(
    incident_scaled: pd.DataFrame,
    anomaly_scores: Dict[str, float],
    anomaly_times: Dict[str, pd.Timestamp],
    top_n: int = 5,
) -> go.Figure:
    """Top-N anomalous metric trends over time, with vertical anomaly markers."""
    top = sorted(anomaly_scores, key=anomaly_scores.get, reverse=True)[:top_n]
    viz_cols = [c for c in top if c in incident_scaled.columns]
    ts_df = incident_scaled.set_index("timestamp")[viz_cols].copy()

    fig = go.Figure()
    for i, col in enumerate(viz_cols):
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[col], mode="lines", name=col))
        if col in anomaly_times:
            x_val = anomaly_times[col].timestamp() * 1000
            pos = "top left" if i % 2 == 0 else "bottom right"
            fig.add_vline(x=x_val, line_dash="dash", line_color="red",
                          annotation_text=f"{col} anomaly", annotation_position=pos)

    fig.update_layout(
        title="Top-5 Anomalous Metrics (scaled 0–1)",
        xaxis_title="Time", yaxis_title="Scaled Value", height=420,
        paper_bgcolor="#151a2e", plot_bgcolor="#151a2e", font=dict(color="#e2e8f0", family="Inter"),
    )
    return fig
