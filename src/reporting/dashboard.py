"""
Streamlit Dashboard for AI-Powered Root Cause Analysis (RCA) System
====================================================================

Interactive web dashboard that lets users:
1. Generate synthetic training data and train the LSTM Autoencoder
2. Inject a failure scenario and run the full RCA pipeline
3. View ranked root causes, causal graph, anomaly timelines, and
   a machine-readable JSON report

Run with:
    cd d:/vscode/majorprojectt/src
    streamlit run reporting/dashboard.py
"""

import sys
import os
import time
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# ── resolve import paths ───────────────────────────────────────────────────────
# When launched via `streamlit run reporting/dashboard.py` from /src, the CWD
# is /src, so we just need the parent (the /src dir itself) on path.
_here = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.dirname(_here)   # .../src
if _src not in sys.path:
    sys.path.insert(0, _src)

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from models.lstm_autoencoder import AnomalyDetector
from causal_inference.causal_engine import (
    GrangerAnalyzer,
    CausalGraphBuilder,
    EventCorrelator,
    RootCauseRanker as CausalRanker,
    CausalInferencePipeline,
)
from reporting.report_generator import ReportGenerator

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Powered RCA System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 AI-Powered Root Cause Analysis (RCA) System")
st.markdown(
    "Diagnose production failures using **LSTM Autoencoders**, "
    "**Granger Causality**, and **Multi-factor Root Cause Scoring**."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — control panel
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Control Panel")

pipeline_state = st.sidebar.radio(
    "Select Pipeline Stage",
    ["1 — Data Generation & Training", "2 — Run RCA Inference"],
    index=0,
)

st.sidebar.markdown("---")

# Training parameters
baseline_days = st.sidebar.slider("Baseline Training Days", 10, 60, 30)
training_epochs = st.sidebar.slider("LSTM Training Epochs", 1, 30, 5)
window_size = st.sidebar.slider("LSTM Window Size (timesteps)", 6, 60, 12)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Inference parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Failure Injection")
failure_type = st.sidebar.selectbox(
    "Failure Scenario",
    ["database_slow_query", "memory_leak", "cpu_spike",
     "network_partition", "thread_pool_exhaustion", "disk_io_spike"],
)
severity = st.sidebar.slider("Severity (0.1 – 1.0)", 0.1, 1.0, 0.8, 0.05)
max_granger_lag = st.sidebar.slider("Granger Max Lag", 2, 10, 5)

# ─────────────────────────────────────────────────────────────────────────────
# Utility: cached data + model
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_training_data(days: int, _seed: int):
    """Generate normal baseline data (cached so training is reproducible)."""
    gen = SyntheticMetricsGenerator(seed=_seed)
    return gen.generate_normal_behavior(duration_days=days)


@st.cache_resource(show_spinner=False)
def get_trained_model(normal_data_hash: int, _normal_data, n_features: int,
                      ws: int, epochs: int):
    """Train (or load from cache) the LSTM Autoencoder."""
    feat_cols = [c for c in _normal_data.columns if c != "timestamp"]
    detector = AnomalyDetector(n_features=n_features, window_size=ws)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    values = scaler.fit_transform(_normal_data[feat_cols].values)
    detector.train(values.astype(np.float32), epochs=epochs, batch_size=32)
    return detector, feat_cols, scaler


def _df_hash(df) -> int:
    """Cheap hash for a DataFrame to use as cache key."""
    return hash(df.shape) ^ hash(tuple(df.columns.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Plotly causal graph
# ─────────────────────────────────────────────────────────────────────────────

def draw_causal_graph(G: nx.DiGraph, root_cause_metric: str) -> go.Figure:
    """Return a Plotly figure for the causal graph."""
    if len(G.nodes) == 0:
        return go.Figure().update_layout(title="No causal edges identified")

    pos = nx.spring_layout(G, seed=1, k=1.5)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    node_colors = [
        "red" if n == root_cause_metric else
        "orange" if G.in_degree(n) == 0 else
        "#4a90e2"
        for n in G.nodes
    ]
    node_text = [
        f"{n}<br>out={G.out_degree(n)} in={G.in_degree(n)}"
        for n in G.nodes
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=list(G.nodes),
        textposition="top center",
        hovertext=node_text,
        marker=dict(size=14, color=node_colors,
                    line=dict(width=2, color="white")),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Causal Graph (Red = Top Root Cause, Orange = Source Node)",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Data Generation & Training
# ─────────────────────────────────────────────────────────────────────────────

if pipeline_state == "1 — Data Generation & Training":
    st.subheader("Stage 1: Data Generation & LSTM Training")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"Will generate **{baseline_days} days** of synthetic normal metrics "
            f"and train an LSTM Autoencoder for **{training_epochs} epoch(s)**."
        )
        train_button = st.button("🚀 Generate Data & Train Model", type="primary")

    if train_button:
        with st.spinner(f"Generating {baseline_days} days of normal metrics …"):
            normal_df = get_training_data(baseline_days, int(seed))

        st.success(f"✅ Data generated: **{len(normal_df):,} samples**, "
                   f"**{len(normal_df.columns) - 1} metrics**")

        with col2:
            st.dataframe(normal_df.head(8), use_container_width=True)

        feat_cols_preview = [c for c in normal_df.columns if c != "timestamp"]

        tab1, tab2 = st.tabs(["📈 Metric Sample", "🤖 Train Model"])

        with tab1:
            fig = px.line(
                normal_df.head(200).set_index("timestamp")[feat_cols_preview[:4]],
                title="First 200 Timesteps — Normal Metrics",
                labels={"value": "Metric Value", "timestamp": "Time"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            with st.spinner(
                f"Training LSTM Autoencoder on {len(normal_df):,} samples "
                f"for {training_epochs} epoch(s) …"
            ):
                t0 = time.time()
                detector, feat_cols, scaler = get_trained_model(
                    _df_hash(normal_df), normal_df,
                    len(feat_cols_preview), window_size, training_epochs
                )
                elapsed = time.time() - t0

            st.success(
                f"✅ Model trained in **{elapsed:.1f}s** | "
                f"Window size: **{window_size}** | "
                f"Features: **{len(feat_cols)}**"
            )
            st.info(
                "Thresholds calibrated using the 99th-percentile reconstruction "
                "error on the validation split. Proceed to Stage 2 to run RCA."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Full RCA Inference
# ─────────────────────────────────────────────────────────────────────────────

elif pipeline_state == "2 — Run RCA Inference":
    st.subheader("Stage 2: RCA Inference Pipeline")

    st.markdown(
        f"**Failure to inject:** `{failure_type}` | "
        f"**Severity:** `{severity}` | "
        f"**Granger max lag:** `{max_granger_lag}`"
    )

    run_button = st.button("▶ Simulate Incident & Run Full RCA", type="primary")

    if run_button:
        # ── Step 1: Generate training + incident data ─────────────────────────
        progress = st.progress(0, "Step 1/5 — Generating data …")

        with st.spinner("Generating baseline + failure data …"):
            normal_df = get_training_data(baseline_days, int(seed))
            feat_cols = [c for c in normal_df.columns if c != "timestamp"]

            gen2 = SyntheticMetricsGenerator(seed=int(seed) + 1)
            incident_base = gen2.generate_normal_behavior(duration_days=3)
            failure_start_idx = len(incident_base) - 200
            incident_df, metadata = gen2.inject_failure_scenario(
                incident_base,
                failure_type=failure_type,
                start_idx=failure_start_idx,
                duration_samples=200,
                severity=severity,
            )

        progress.progress(20, "Step 2/5 — Preprocessing …")

        # ── Step 2: Preprocess ────────────────────────────────────────────────
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        normal_scaled_vals = scaler.fit_transform(normal_df[feat_cols].values)

        incident_clean = incident_df[feat_cols].ffill().bfill()
        incident_scaled_vals = np.clip(
            scaler.transform(incident_clean.values), 0.0, 1.0
        )
        incident_scaled = pd.DataFrame(incident_scaled_vals, columns=feat_cols)
        incident_scaled.insert(0, "timestamp", incident_df["timestamp"].values)

        ground_truth = metadata.get("root_cause", "unknown")
        causal_chain = metadata.get("causal_chain", [])

        st.success(
            f"✅ Data ready | Ground truth: **{ground_truth}** | "
            f"Chain: {' -> '.join(causal_chain)}"
        )

        progress.progress(35, "Step 3/5 — Training LSTM Autoencoder …")

        # ── Step 3: Train LSTM ────────────────────────────────────────────────
        with st.spinner(
            f"Training LSTM ({training_epochs} epoch[s], window={window_size}) …"
        ):
            detector, _fc, _sc = get_trained_model(
                _df_hash(normal_df), normal_df,
                len(feat_cols), window_size, training_epochs
            )

        st.success(
            f"✅ LSTM trained | {len(normal_df):,} samples | "
            f"{len(feat_cols)} features"
        )

        progress.progress(50, "Step 4/5 — Anomaly Detection …")

        # ── Step 4: Anomaly Detection ─────────────────────────────────────────
        with st.spinner("Running anomaly detection on incident window …"):
            result_df = detector.detect(incident_scaled, feat_cols)

            anomaly_scores: dict = {}
            anomaly_times:  dict = {}
            active_anomalies = []

            for col in feat_cols:
                score_col = f"{col}_score"
                flag_col  = f"{col}_is_anomaly"
                if score_col not in result_df.columns:
                    continue
                flagged = result_df[result_df[flag_col] == True]
                if not flagged.empty:
                    active_anomalies.append(col)
                    anomaly_scores[col] = float(result_df[score_col].max())
                    first_idx = flagged.index[0]
                    ts_val = incident_scaled.loc[first_idx, "timestamp"]
                    anomaly_times[col] = pd.Timestamp(ts_val)

        n_anomalies = len(active_anomalies)
        st.success(f"✅ Detected **{n_anomalies}** anomalous metric(s)")

        if n_anomalies == 0:
            st.warning(
                "No anomalies detected. Try increasing **Severity** or "
                "**Training Epochs** in the sidebar."
            )
            st.stop()

        progress.progress(70, "Step 5/5 — Causal Inference & Ranking …")

        # ── Step 5: Causal Inference ──────────────────────────────────────────
        if n_anomalies < 2:
            st.warning(
                "Only 1 anomalous metric — causal graph will have a single node. "
                "Increase severity for richer results."
            )

        with st.spinner("Running Granger causality & building causal graph …"):
            df_for_granger = incident_scaled.set_index("timestamp")[
                [m for m in active_anomalies if m in incident_scaled.columns]
            ]

            failure_start_time = pd.Timestamp(
                incident_df.iloc[failure_start_idx]["timestamp"]
            )
            events_df = pd.DataFrame([{
                "timestamp":   failure_start_time - pd.Timedelta(minutes=20),
                "description": "Code deployment preceding incident",
                "type":        "deployment",
            }])

            pipeline = CausalInferencePipeline(
                max_lag=max_granger_lag, significance_level=0.05
            )
            causal_results = pipeline.run(
                df=df_for_granger,
                anomalous_metrics=active_anomalies,
                anomaly_scores=anomaly_scores,
                anomaly_first_seen=anomaly_times,
                events_df=events_df,
            )

        causal_graph  = causal_results["causal_graph"]
        root_causes   = causal_results["root_causes"]
        event_corrs   = causal_results["event_correlations"]

        progress.progress(100, "✅ Pipeline complete!")

        # ── Build report ──────────────────────────────────────────────────────
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        report_gen  = ReportGenerator()

        ranked_tuples = [
            (
                rc["metric"],
                rc["composite_score"],
                {
                    "out_edges":  rc.get("downstream_effects", []),
                    "components": rc.get("scores_breakdown", {}),
                    "pagerank":   rc.get("pagerank", 0.0),
                }
            )
            for rc in root_causes
        ]

        md_report = report_gen.generate_report(
            incident_id=incident_id,
            ranked_candidates=ranked_tuples,
            causal_graph=causal_graph,
            anomaly_times=anomaly_times,
        )

        # ── Display results ───────────────────────────────────────────────────
        st.markdown("---")

        top_metric = root_causes[0]["metric"] if root_causes else "N/A"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Primary Root Cause", top_metric)
        kpi2.metric("Anomalous Metrics", n_anomalies)
        kpi3.metric(
            "Top Confidence",
            root_causes[0]["confidence"] if root_causes else "—"
        )
        kpi4.metric("Causal Edges", causal_graph.number_of_edges())

        st.markdown("---")

        # ── Tab layout ────────────────────────────────────────────────────────
        tab_rc, tab_graph, tab_timeline, tab_report, tab_json = st.tabs([
            "🏆 Root Causes",
            "🕸️ Causal Graph",
            "📊 Anomaly Timeline",
            "📄 MD Report",
            "🗂️ JSON Report",
        ])

        with tab_rc:
            st.subheader("Ranked Root Cause Candidates")
            if root_causes:
                rows = []
                for rc in root_causes:
                    sb = rc.get("scores_breakdown", {})
                    rows.append({
                        "Rank":            rc["rank"],
                        "Metric":          rc["metric"],
                        "Composite Score": f"{rc['composite_score']:.4f}",
                        "Confidence":      rc["confidence"],
                        "Causal Outflow":  f"{sb.get('causal_outflow', 0):.3f}",
                        "Temporal Prio":   f"{sb.get('temporal_priority', 0):.3f}",
                        "Severity":        f"{sb.get('anomaly_severity', 0):.3f}",
                        "Downstream":      ", ".join(rc.get("downstream_effects", [])[:3]) or "—",
                    })
                df_rc = pd.DataFrame(rows)
                st.dataframe(df_rc, use_container_width=True, hide_index=True)

                st.markdown("#### Ground Truth Comparison")
                chain_str = " -> ".join(causal_chain)
                top1_words  = set(top_metric.lower().replace("_", " ").split())
                chain_words = set(chain_str.lower().replace(",", " ").split())
                match = bool(top1_words & chain_words)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Ground Truth Root Cause", ground_truth)
                col_b.metric("Top-1 Prediction", top_metric)
                col_c.metric("Prediction Aligned", "✅ Yes" if match else "❌ No")
                st.caption(f"Causal chain: {chain_str or 'N/A'}")

                if event_corrs:
                    st.markdown("#### Event Correlations")
                    ec_df = pd.DataFrame(event_corrs[:5])[
                        ["metric", "event_description", "delta_hours", "correlation_score"]
                    ]
                    st.dataframe(ec_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No root causes ranked — insufficient causal graph data.")

        with tab_graph:
            st.subheader("Causal Graph")
            top_rc_metric = root_causes[0]["metric"] if root_causes else ""
            fig_graph = draw_causal_graph(causal_graph, top_rc_metric)
            st.plotly_chart(fig_graph, use_container_width=True)

            if causal_graph.edges:
                st.markdown("**Causal Edges (sorted by strength):**")
                edges_data = []
                for u, v, d in causal_graph.edges(data=True):
                    edges_data.append({
                        "Cause":    u,
                        "Effect":   v,
                        "Strength": round(d.get("strength", 0.0), 4),
                        "Lag":      d.get("lag", "?"),
                        "p-value":  round(d.get("p_value", 1.0), 6),
                    })
                edges_df = pd.DataFrame(edges_data).sort_values(
                    "Strength", ascending=False
                )
                st.dataframe(edges_df, use_container_width=True, hide_index=True)

        with tab_timeline:
            st.subheader("Anomaly Detection Timeline")
            top_5 = sorted(anomaly_scores, key=anomaly_scores.get, reverse=True)[:5]  # type: ignore[arg-type]
            if top_5:
                viz_cols = [c for c in top_5 if c in incident_scaled.columns]
                ts_df = incident_scaled.set_index("timestamp")[viz_cols].copy()

                # Annotate anomaly start times
                fig_ts = go.Figure()
                for col in viz_cols:
                    fig_ts.add_trace(go.Scatter(
                        x=ts_df.index, y=ts_df[col],
                        mode="lines", name=col,
                    ))
                    if col in anomaly_times:
                        # Convert to milliseconds since epoch to avoid Plotly math errors with Timestamps
                        x_val = anomaly_times[col].timestamp() * 1000
                        fig_ts.add_vline(
                            x=x_val,
                            line_dash="dash", line_color="red",
                            annotation_text=f"{col} anomaly",
                            annotation_position="top left",
                        )

                fig_ts.update_layout(
                    title="Top-5 Anomalous Metrics (scaled 0–1)",
                    xaxis_title="Time",
                    yaxis_title="Scaled Value",
                    height=420,
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            # Anomaly score bar chart
            if anomaly_scores:
                score_df = pd.DataFrame.from_dict(
                    anomaly_scores, orient="index", columns=["Max Score"]
                ).sort_values("Max Score", ascending=False)

                fig_bar = px.bar(
                    score_df, y="Max Score", title="Max Anomaly Score per Metric",
                    color="Max Score", color_continuous_scale="reds",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_report:
            st.subheader("Markdown Incident Report")
            st.markdown(md_report)

        with tab_json:
            st.subheader("JSON Report (machine-readable)")
            edges_serializable = [
                {
                    "cause":    u,
                    "effect":   v,
                    "strength": round(float(d.get("strength", 0.0)), 4),
                    "lag":      d.get("lag"),
                    "p_value":  round(float(d.get("p_value", 1.0)), 6),
                }
                for u, v, d in causal_graph.edges(data=True)
            ]

            json_report = {
                "incident_id":    incident_id,
                "timestamp":      datetime.now().isoformat() + "Z",
                "failure_type":   failure_type,
                "ground_truth": {
                    "root_cause":   metadata.get("root_cause"),
                    "causal_chain": metadata.get("causal_chain"),
                },
                "root_causes": [
                    {
                        "rank":              rc["rank"],
                        "metric":            rc["metric"],
                        "composite_score":   rc["composite_score"],
                        "confidence":        rc["confidence"],
                        "scores_breakdown":  rc.get("scores_breakdown", {}),
                        "downstream_effects": rc.get("downstream_effects", []),
                        "causal_chain":      rc.get("causal_chain", []),
                    }
                    for rc in root_causes
                ],
                "causal_graph": {
                    "nodes": list(causal_graph.nodes),
                    "edges": edges_serializable,
                },
                "event_correlations": event_corrs,
                "anomaly_detection_times": {
                    k: str(v) for k, v in anomaly_times.items()
                },
            }

            json_str = json.dumps(json_report, indent=2, default=str)
            st.code(json_str, language="json")
            st.download_button(
                label="⬇️ Download JSON Report",
                data=json_str,
                file_name=f"{incident_id}_report.json",
                mime="application/json",
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AI-Powered RCA System | LSTM Autoencoder + Granger Causality + "
    "Multi-factor Root Cause Ranking | College Major Project"
)
