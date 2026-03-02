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
import math
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# ── resolve import paths ───────────────────────────────────────────────────────
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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Aesthetic Overhaul
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Animated gradient header ────────────────────────────────── */
.hero-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    0% { filter: brightness(1); }
    100% { filter: brightness(1.15); }
}

.hero-subtitle {
    text-align: center;
    color: #a0aec0;
    font-size: 1.1rem;
    font-weight: 300;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}

/* ── Glass card ──────────────────────────────────────────────── */
.glass-card {
    background: rgba(30, 33, 48, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
}

/* ── KPI metric cards ────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.10));
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    transform: translateY(-3px);
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.kpi-label {
    font-size: 0.85rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Pipeline stage badge ────────────────────────────────────── */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* ── Confidence gauge ────────────────────────────────────────── */
.confidence-gauge {
    background: rgba(30, 33, 48, 0.6);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.gauge-bar {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    transition: width 1s ease;
    box-shadow: 0 0 12px rgba(102, 126, 234, 0.4);
}

.gauge-bg {
    height: 12px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.06);
    overflow: hidden;
}

/* ── Sidebar styling ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #1a1f3a 50%, #0f1628 100%);
    border-right: 1px solid rgba(102, 126, 234, 0.15);
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 500;
}

/* ── Scenario info box ───────────────────────────────────────── */
.scenario-info {
    background: rgba(102, 126, 234, 0.08);
    border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #c4cef5;
    margin-top: 0.3rem;
}

/* ── Fade-in animation for sections ──────────────────────────── */
.fade-in {
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Tab styling ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: rgba(102, 126, 234, 0.12);
    border-bottom: 2px solid #667eea;
}

/* ── Primary button glow ─────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.45) !important;
}

/* ── Divider ─────────────────────────────────────────────────── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    border: none;
    margin: 1.5rem 0;
}

/* ── About section ───────────────────────────────────────────── */
.about-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.about-item {
    background: rgba(30, 33, 48, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.3s ease;
}

.about-item:hover {
    border-color: rgba(102, 126, 234, 0.3);
}

.about-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.about-title {
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
    font-size: 0.95rem;
}

.about-desc {
    font-size: 0.8rem;
    color: #718096;
    line-height: 1.4;
}

/* ── Footer ──────────────────────────────────────────────────── */
.footer-text {
    text-align: center;
    color: #4a5568;
    font-size: 0.82rem;
    padding-top: 1rem;
    letter-spacing: 0.03em;
}

/* ── Accuracy table styling ──────────────────────────────────── */
.accuracy-highlight {
    background: linear-gradient(135deg, rgba(72, 187, 120, 0.12), rgba(56, 178, 172, 0.08));
    border: 1px solid rgba(72, 187, 120, 0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.accuracy-number {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #48bb78, #38b2ac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.accuracy-label {
    font-size: 0.85rem;
    color: #68d391;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero Title
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="fade-in">
    <div class="hero-title">🔍 AI-Powered Root Cause Analysis</div>
    <div class="hero-subtitle">
        Diagnose production failures using <b>LSTM Autoencoders</b>, 
        <b>Granger Causality</b>, and <b>Multi-factor Root Cause Scoring</b>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Welcome / About Section
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️  About This System", expanded=False):
    st.markdown("""
<div class="about-grid">
    <div class="about-item">
        <div class="about-icon">🧠</div>
        <div class="about-title">LSTM Autoencoder</div>
        <div class="about-desc">Deep learning model trained on normal baselines. Flags anomalies via reconstruction error exceeding the 99th-percentile threshold.</div>
    </div>
    <div class="about-item">
        <div class="about-icon">🔗</div>
        <div class="about-title">Granger Causality</div>
        <div class="about-desc">Statistical tests discover causal relationships between metrics. Combined with PC algorithm for constraint-based structure learning.</div>
    </div>
    <div class="about-item">
        <div class="about-icon">🏆</div>
        <div class="about-title">Multi-Factor Scoring</div>
        <div class="about-desc">Ranks root causes using causal outflow, temporal priority, anomaly severity, and PageRank-augmented graph influence.</div>
    </div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Pipeline Flow:**  `Data Generation` → `Preprocessing` → `LSTM Training` → `Anomaly Detection` → `Causal Inference & Ranking`
    
    Use **Stage 1** to generate baseline data and train the model, then switch to **Stage 2** to simulate an incident and run the full RCA pipeline.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario descriptions dictionary (Feature 4)
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_DESCRIPTIONS = {
    "database_slow_query":      "🗄️ Simulates slow DB queries causing cascading latency and throughput drops",
    "memory_leak":              "💾 Gradual memory exhaustion leading to OOM errors and degraded performance",
    "cpu_spike":                "🔥 CPU saturation from a runaway process, causing request queuing and timeouts",
    "network_partition":        "🌐 Network failures causing error rate spikes and intermittent connectivity",
    "thread_pool_exhaustion":   "🧵 Thread pool starvation reducing throughput and increasing response times",
    "disk_io_spike":            "💿 Disk I/O bottleneck from excessive logging, impacting DB connections",
}

SCENARIO_ROOT_CAUSES = {
    "database_slow_query":    "db_active_connections",
    "memory_leak":            "memory_usage_percent",
    "cpu_spike":              "cpu_usage_percent",
    "network_partition":      "error_rate_percent",
    "thread_pool_exhaustion": "throughput_rps",
    "disk_io_spike":          "disk_io_bytes",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Control Panel (Feature 3: Tooltips, Feature 4: Descriptions)
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:1rem;">
    <span style="font-size:1.5rem;">⚙️</span>
    <span style="font-size:1.2rem; font-weight:700; 
          background: linear-gradient(135deg, #667eea, #f093fb);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text;">Control Panel</span>
</div>
""", unsafe_allow_html=True)

pipeline_state = st.sidebar.radio(
    "Select Pipeline Stage",
    ["1 — Data Generation & Training", "2 — Run RCA Inference"],
    index=0,
    help="Stage 1 generates synthetic data and trains the model. Stage 2 runs the full RCA pipeline on a simulated incident.",
)

st.sidebar.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# Training parameters with tooltips (Feature 3)
baseline_days = st.sidebar.slider(
    "Baseline Training Days", 10, 60, 30,
    help="Number of days of normal operational data to generate. More data = better baseline but slower training. 30 days is a good default."
)
training_epochs = st.sidebar.slider(
    "LSTM Training Epochs", 1, 30, 5,
    help="Training iterations over the full dataset. More epochs = lower reconstruction error but risk of overfitting. 5-10 is optimal."
)
window_size = st.sidebar.slider(
    "LSTM Window Size (timesteps)", 6, 60, 12,
    help="Number of consecutive timesteps the model looks at in each sliding window. Larger windows capture longer temporal patterns but need more data."
)
seed = st.sidebar.number_input(
    "Random Seed", value=42, step=1,
    help="Controls random number generation for reproducibility. Same seed = same synthetic data every time.",
)

# Failure injection parameters
st.sidebar.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("**🎯 Failure Injection**")

failure_type = st.sidebar.selectbox(
    "Failure Scenario",
    list(SCENARIO_DESCRIPTIONS.keys()),
    help="Choose which type of infrastructure failure to simulate. Each scenario has a known root cause for validation.",
)

# Feature 4: Show scenario description
st.sidebar.markdown(
    f'<div class="scenario-info">{SCENARIO_DESCRIPTIONS.get(failure_type, "")}</div>',
    unsafe_allow_html=True,
)

severity = st.sidebar.slider(
    "Severity (0.1 – 1.0)", 0.1, 1.0, 0.8, 0.05,
    help="Failure intensity multiplier. Higher severity = more pronounced anomalies and clearer causal chains. 0.8 is a good starting point.",
)
max_granger_lag = st.sidebar.slider(
    "Granger Max Lag", 2, 10, 5,
    help="Maximum number of time lags tested in Granger causality. Higher values detect longer-range causal effects but increase computation time.",
)


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
# Utility: Enhanced Plotly causal graph (Feature 5)
# ─────────────────────────────────────────────────────────────────────────────

def draw_causal_graph(G: nx.DiGraph, root_cause_metric: str) -> go.Figure:
    """Return a Plotly figure for the causal graph with arrows and legend."""
    if len(G.nodes) == 0:
        return go.Figure().update_layout(title="No causal edges identified")

    # Use kamada_kawai for better layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=1, k=1.5)

    fig = go.Figure()

    # ── Draw edges with arrows ──
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        lag = d.get("lag", "?")

        # Edge line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=1.5, color="rgba(150, 160, 200, 0.45)"),
            hoverinfo="text",
            hovertext=f"{u} → {v} (lag={lag})",
            showlegend=False,
        ))

        # Arrowhead via annotation
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
            arrowcolor="rgba(150, 160, 200, 0.6)",
        )

        # Edge label (lag)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_annotation(
            x=mx, y=my, text=f"lag={lag}",
            showarrow=False,
            font=dict(size=9, color="rgba(180, 190, 220, 0.7)"),
        )

    # ── Classify nodes ──
    node_categories = {}
    for n in G.nodes:
        if n == root_cause_metric:
            node_categories[n] = "root_cause"
        elif G.in_degree(n) == 0:
            node_categories[n] = "source"
        else:
            node_categories[n] = "intermediate"

    color_map = {
        "root_cause":    "#ff4757",   # vibrant red
        "source":        "#ffa502",   # warm orange
        "intermediate":  "#70a1ff",   # soft blue
    }

    size_map = {
        "root_cause":    34,
        "source":        26,
        "intermediate":  22,
    }

    # ── Draw nodes ──
    for cat, cat_label in [("root_cause", "🔴 Root Cause"), ("source", "🟠 Source Node"), ("intermediate", "🔵 Intermediate")]:
        cat_nodes = [n for n in G.nodes if node_categories[n] == cat]
        if not cat_nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in cat_nodes],
            y=[pos[n][1] for n in cat_nodes],
            mode="markers+text",
            text=cat_nodes,
            textposition="top center",
            textfont=dict(size=11, color="#e2e8f0"),
            hovertext=[
                f"<b>{n}</b><br>Out-degree: {G.out_degree(n)}<br>In-degree: {G.in_degree(n)}"
                for n in cat_nodes
            ],
            hoverinfo="text",
            name=cat_label,
            marker=dict(
                size=[size_map[cat]] * len(cat_nodes),
                color=color_map[cat],
                line=dict(width=3, color="white"),
                symbol="circle",
            ),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(text="Causal Dependency Graph", font=dict(size=16)),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Run a single scenario (for accuracy benchmark)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_single_scenario(_normal_df, feat_cols_list, scenario: str,
                        sev: float, ws: int, epochs: int, _seed: int,
                        mgl: int):
    """Run the full RCA pipeline for a single scenario and return results."""
    from sklearn.preprocessing import MinMaxScaler

    # Get trained model
    detector, _fc, _sc = get_trained_model(
        _df_hash(_normal_df), _normal_df,
        len(feat_cols_list), ws, epochs
    )

    # Generate incident data
    gen2 = SyntheticMetricsGenerator(seed=_seed + 1)
    incident_base = gen2.generate_normal_behavior(duration_days=3)
    failure_start_idx = len(incident_base) - 200
    incident_df, metadata = gen2.inject_failure_scenario(
        incident_base, failure_type=scenario,
        start_idx=failure_start_idx, duration_samples=200, severity=sev,
    )

    # Preprocess
    scaler = MinMaxScaler()
    scaler.fit(_normal_df[feat_cols_list].values)
    incident_clean = incident_df[feat_cols_list].ffill().bfill()
    incident_scaled_vals = np.clip(scaler.transform(incident_clean.values), 0.0, 1.0)
    incident_scaled = pd.DataFrame(incident_scaled_vals, columns=feat_cols_list)
    incident_scaled.insert(0, "timestamp", incident_df["timestamp"].values)

    # Detect anomalies
    result_df = detector.detect(incident_scaled, feat_cols_list)
    anomaly_scores = {}
    anomaly_times = {}
    active_anomalies = []

    for col in feat_cols_list:
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

    if len(active_anomalies) < 1:
        return {
            "scenario": scenario,
            "ground_truth": metadata.get("root_cause", "unknown"),
            "top1": "No anomalies",
            "top3": [],
            "top1_match": False,
            "top3_match": False,
            "score": 0.0,
        }

    # Causal inference
    df_for_granger = incident_scaled.set_index("timestamp")[
        [m for m in active_anomalies if m in incident_scaled.columns]
    ]
    failure_start_time = pd.Timestamp(incident_df.iloc[failure_start_idx]["timestamp"])
    events_df = pd.DataFrame([{
        "timestamp":   failure_start_time - pd.Timedelta(minutes=20),
        "description": "Code deployment preceding incident",
        "type":        "deployment",
    }])

    pipeline = CausalInferencePipeline(max_lag=mgl, significance_level=0.05)
    causal_results = pipeline.run(
        df=df_for_granger,
        anomalous_metrics=active_anomalies,
        anomaly_scores=anomaly_scores,
        anomaly_first_seen=anomaly_times,
        events_df=events_df,
    )

    root_causes = causal_results["root_causes"]

    if not root_causes:
        return {
            "scenario": scenario,
            "ground_truth": metadata.get("root_cause", "unknown"),
            "top1": "No root causes",
            "top3": [],
            "top1_match": False,
            "top3_match": False,
            "score": 0.0,
        }

    top1_metric = root_causes[0]["metric"]
    top3_metrics = [rc["metric"] for rc in root_causes[:3]]
    gt = metadata.get("root_cause", "")
    causal_chain = metadata.get("causal_chain", [])

    # Match logic: check if prediction overlaps with ground truth chain
    gt_words = set(gt.lower().replace(",", " ").split())
    chain_words = set(" ".join(causal_chain).lower().replace(",", " ").split())
    all_gt_words = gt_words | chain_words

    top1_words = set(top1_metric.lower().replace("_", " ").split())
    top1_match = bool(top1_words & all_gt_words)

    top3_match = False
    for m in top3_metrics:
        m_words = set(m.lower().replace("_", " ").split())
        if m_words & all_gt_words:
            top3_match = True
            break

    return {
        "scenario": scenario,
        "ground_truth": gt,
        "top1": top1_metric,
        "top3": top3_metrics,
        "top1_match": top1_match,
        "top3_match": top3_match,
        "score": root_causes[0]["composite_score"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Data Generation & Training
# ─────────────────────────────────────────────────────────────────────────────

if pipeline_state == "1 — Data Generation & Training":
    st.markdown('<div class="stage-badge">🧪 Stage 1 — Data Generation & LSTM Training</div>', unsafe_allow_html=True)

    # Feature 2: KPI Preview Cards
    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin-bottom:1.5rem;" class="fade-in">
        <div class="kpi-card">
            <div class="kpi-value">{baseline_days * 288:,}</div>
            <div class="kpi-label">Expected Samples</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">10</div>
            <div class="kpi-label">Metrics Tracked</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">~35K</div>
            <div class="kpi-label">Model Parameters</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{training_epochs}</div>
            <div class="kpi-label">Training Epochs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
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
                "error on the validation split. Proceed to **Stage 2** to run RCA."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Full RCA Inference
# ─────────────────────────────────────────────────────────────────────────────

elif pipeline_state == "2 — Run RCA Inference":
    st.markdown('<div class="stage-badge">🔬 Stage 2 — RCA Inference Pipeline</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card fade-in">
        <div style="display:flex; gap:2rem; flex-wrap:wrap;">
            <div><span style="color:#8892b0;">Failure:</span> <b style="color:#ff6b6b;">{failure_type}</b></div>
            <div><span style="color:#8892b0;">Severity:</span> <b style="color:#ffa502;">{severity}</b></div>
            <div><span style="color:#8892b0;">Granger Lag:</span> <b style="color:#70a1ff;">{max_granger_lag}</b></div>
            <div><span style="color:#8892b0;">Expected Root Cause:</span> <b style="color:#7bed9f;">{SCENARIO_ROOT_CAUSES.get(failure_type, 'unknown')}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        top_metric = root_causes[0]["metric"] if root_causes else "N/A"
        top_score  = root_causes[0]["composite_score"] if root_causes else 0.0
        top_conf   = root_causes[0]["confidence"] if root_causes else "—"

        # KPI Row
        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin-bottom:1rem;" class="fade-in">
            <div class="kpi-card">
                <div class="kpi-value" style="font-size:1.3rem;">{top_metric}</div>
                <div class="kpi-label">Primary Root Cause</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{n_anomalies}</div>
                <div class="kpi-label">Anomalous Metrics</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{top_conf}</div>
                <div class="kpi-label">Confidence Level</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{causal_graph.number_of_edges()}</div>
                <div class="kpi-label">Causal Edges</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature 6: Confidence Gauge
        pct = max(0, min(top_score * 100, 100))
        gauge_color = "#48bb78" if pct >= 70 else "#ffa502" if pct >= 40 else "#ff4757"
        st.markdown(f"""
        <div class="confidence-gauge fade-in">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#8892b0; font-size:0.85rem; font-weight:500;">Root Cause Confidence</span>
                <span style="color:{gauge_color}; font-weight:700; font-size:0.95rem;">{pct:.1f}%</span>
            </div>
            <div class="gauge-bg">
                <div class="gauge-bar" style="width:{pct}%; background: linear-gradient(90deg, #667eea, {gauge_color});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # ── Tab layout ────────────────────────────────────────────────────────
        tab_rc, tab_graph, tab_timeline, tab_report, tab_json, tab_accuracy = st.tabs([
            "🏆 Root Causes",
            "🕸️ Causal Graph",
            "📊 Anomaly Timeline",
            "📄 MD Report",
            "🗂️ JSON Report",
            "📈 Accuracy Benchmark",
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

                # Feature 7: Download CSV
                csv_data = df_rc.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Rankings as CSV",
                    data=csv_data,
                    file_name=f"{incident_id}_rankings.csv",
                    mime="text/csv",
                )

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

                fig_ts = go.Figure()
                for col in viz_cols:
                    fig_ts.add_trace(go.Scatter(
                        x=ts_df.index, y=ts_df[col],
                        mode="lines", name=col,
                    ))
                    if col in anomaly_times:
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
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            if anomaly_scores:
                score_df = pd.DataFrame.from_dict(
                    anomaly_scores, orient="index", columns=["Max Score"]
                ).sort_values("Max Score", ascending=False)

                fig_bar = px.bar(
                    score_df, y="Max Score", title="Max Anomaly Score per Metric",
                    color="Max Score", color_continuous_scale="reds",
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_report:
            st.subheader("Markdown Incident Report")
            st.markdown(md_report)
            # Feature 7: Download MD report
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=md_report,
                file_name=f"{incident_id}_report.md",
                mime="text/markdown",
            )

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

        # ── Feature 8: Accuracy Benchmark Tab ─────────────────────────────────
        with tab_accuracy:
            st.subheader("📈 Accuracy Benchmark — All 6 Scenarios")
            st.markdown(
                "Automatically runs the RCA pipeline across all failure scenarios "
                "to measure Top-1 and Top-3 identification accuracy."
            )

            run_benchmark = st.button("🧪 Run Full Benchmark", type="primary", key="benchmark_btn")

            if run_benchmark:
                all_scenarios = list(SCENARIO_DESCRIPTIONS.keys())
                benchmark_results = []

                bench_progress = st.progress(0, "Running benchmark …")

                for i, scenario in enumerate(all_scenarios):
                    bench_progress.progress(
                        int((i / len(all_scenarios)) * 100),
                        f"Running scenario {i+1}/{len(all_scenarios)}: {scenario} …"
                    )
                    try:
                        result = run_single_scenario(
                            normal_df, feat_cols, scenario,
                            severity, window_size, training_epochs,
                            int(seed), max_granger_lag,
                        )
                        benchmark_results.append(result)
                    except Exception as e:
                        benchmark_results.append({
                            "scenario": scenario,
                            "ground_truth": SCENARIO_ROOT_CAUSES.get(scenario, "?"),
                            "top1": f"Error: {str(e)[:30]}",
                            "top3": [],
                            "top1_match": False,
                            "top3_match": False,
                            "score": 0.0,
                        })

                bench_progress.progress(100, "✅ Benchmark complete!")

                # Display results
                bench_rows = []
                for r in benchmark_results:
                    bench_rows.append({
                        "Scenario":          r["scenario"],
                        "Ground Truth":      r["ground_truth"],
                        "Top-1 Prediction":  r["top1"],
                        "Score":             f"{r['score']:.4f}" if r['score'] else "—",
                        "Top-1 Match":       "✅" if r["top1_match"] else "❌",
                        "Top-3 Match":       "✅" if r["top3_match"] else "❌",
                    })

                bench_df = pd.DataFrame(bench_rows)
                st.dataframe(bench_df, use_container_width=True, hide_index=True)

                # Accuracy summary
                top1_acc = sum(1 for r in benchmark_results if r["top1_match"]) / len(benchmark_results) * 100
                top3_acc = sum(1 for r in benchmark_results if r["top3_match"]) / len(benchmark_results) * 100

                st.markdown(f"""
                <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:1.5rem; margin-top:1rem;" class="fade-in">
                    <div class="accuracy-highlight">
                        <div class="accuracy-number">{top1_acc:.0f}%</div>
                        <div class="accuracy-label">Top-1 Accuracy</div>
                    </div>
                    <div class="accuracy-highlight">
                        <div class="accuracy-number">{top3_acc:.0f}%</div>
                        <div class="accuracy-label">Top-3 Accuracy</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Download benchmark
                bench_csv = bench_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Benchmark Results",
                    data=bench_csv,
                    file_name="rca_benchmark_results.csv",
                    mime="text/csv",
                    key="bench_download",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    <b>AI-Powered RCA System</b> &nbsp;•&nbsp; LSTM Autoencoder + Granger Causality + Multi-factor Root Cause Ranking
    <br>College Major Project &nbsp;•&nbsp; Built with PyTorch, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
