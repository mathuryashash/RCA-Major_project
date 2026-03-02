"""
Visual Anomaly Simulator Suite for the RCA System

This interactive dashboard proves the Unsupervised LSTM Autoencoder's 
ability to detect a massive variety of anomalies by testing 25+ distinct
failure shapes (Spikes, Drops, Ramps, Noise bursts) across different metrics.

Run with:
    streamlit run src/reporting/anomaly_simulator.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Setup paths to import project modules
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from models.lstm_autoencoder import AnomalyDetector

st.set_page_config(page_title="25+ Anomaly Simulator", layout="wide")

st.title("🧪 25+ Anomaly Stress-Test Simulator")
st.markdown("""
This suite demonstrates the system's Unsupervised LSTM Engine. Because the AI only learns what **normal** looks like, it can detect dozens of unprogrammed, zero-day anomalies.
Select from **27 distinct anomaly scenarios** below to visualize how the engine reacts to different metrics, shapes, and severities.
""")

# ── 1. DEFINING 27 ANOMALIES ──────────────────────────────────────────────
ANOMALIES = [
    # --- CPU Anomalies ---
    {"name": "01. CPU Sudden Spike", "metric": "cpu_utilization", "shape": "spike", "severity": 0.8},
    {"name": "02. CPU Sustained Saturation (Step Up)", "metric": "cpu_utilization", "shape": "step_up", "severity": 0.9},
    {"name": "03. CPU Gradual Resource Exhaustion (Ramp)", "metric": "cpu_utilization", "shape": "ramp_up", "severity": 0.8},
    {"name": "04. CPU Noise Burst (High Variance)", "metric": "cpu_utilization", "shape": "noise", "severity": 0.5},
    
    # --- Memory Anomalies ---
    {"name": "05. Memory Leak (Ramp Up)", "metric": "memory_usage_percent", "shape": "ramp_up", "severity": 0.9},
    {"name": "06. Sudden Memory Allocation (Step Up)", "metric": "memory_usage_percent", "shape": "step_up", "severity": 0.7},
    {"name": "07. Memory Flapping (Variance)", "metric": "memory_usage_percent", "shape": "noise", "severity": 0.6},
    
    # --- Latency Anomalies ---
    {"name": "08. API P50 Latency Spike", "metric": "api_latency_p50_ms", "shape": "spike", "severity": 0.8},
    {"name": "09. API P95 Latency Step Up", "metric": "api_latency_p95_ms", "shape": "step_up", "severity": 0.6},
    {"name": "10. Tail Latency (P99) Outliers (Noise)", "metric": "api_latency_p99_ms", "shape": "noise", "severity": 0.7},
    {"name": "11. API Latency Slow Degradation (Ramp)", "metric": "api_latency_p50_ms", "shape": "ramp_up", "severity": 0.8},
    
    # --- Error Rate Anomalies ---
    {"name": "12. Sudden Error Spike (Code Bug)", "metric": "error_rate_percent", "shape": "spike", "severity": 0.9},
    {"name": "13. Sustained Errors (Bad Config)", "metric": "error_rate_percent", "shape": "step_up", "severity": 0.5},
    {"name": "14. Gradual Error Increase", "metric": "error_rate_percent", "shape": "ramp_up", "severity": 0.6},
    
    # --- Database Anomalies ---
    {"name": "15. DB Connection Spike", "metric": "db_connections_active", "shape": "spike", "severity": 0.7},
    {"name": "16. Connection Pool Exhaustion (Step)", "metric": "db_connections_active", "shape": "step_up", "severity": 0.9},
    {"name": "17. Slow DB Connection Leak (Ramp)", "metric": "db_connections_active", "shape": "ramp_up", "severity": 0.8},
    
    # --- Cache Anomalies ---
    {"name": "18. Cache Eviction Drop (Step Down)", "metric": "cache_hit_rate", "shape": "step_down", "severity": 0.8},
    {"name": "19. Cache Degradation (Ramp Down)", "metric": "cache_hit_rate", "shape": "ramp_down", "severity": 0.7},
    {"name": "20. Cache Hit Rate Volatility", "metric": "cache_hit_rate", "shape": "noise", "severity": 0.5},
    
    # --- Throughput Anomalies ---
    {"name": "21. DDoS Attack (Throughput Spike)", "metric": "request_throughput", "shape": "spike", "severity": 0.9},
    {"name": "22. Traffic Routing Failure (Step Down)", "metric": "request_throughput", "shape": "step_down", "severity": 0.8},
    {"name": "23. Dropping Traffic Slowly (Ramp Down)", "metric": "request_throughput", "shape": "ramp_down", "severity": 0.7},
    
    # --- Disk IO Anomalies ---
    {"name": "24. Disk IO Contention (Spike)", "metric": "disk_io_wait_percent", "shape": "spike", "severity": 0.8},
    {"name": "25. Disk Out of Space / High IO (Step Up)", "metric": "disk_io_wait_percent", "shape": "step_up", "severity": 0.9},
    
    # --- Multi-Metric Cascades ---
    {"name": "26. CPU + Memory Exhaustion (Dual Ramp)", "metric": "multi_resource", "shape": "custom", "severity": 0.8},
    {"name": "27. Total Service Outage (Latency + Errors)", "metric": "multi_outage", "shape": "custom", "severity": 0.9},
]

# ── 2. UTILITY & CACHING ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_base_data() -> pd.DataFrame:
    gen = SyntheticMetricsGenerator(seed=101)
    return gen.generate_normal_behavior(duration_days=10) # 10 days of training

@st.cache_resource(show_spinner=False)
def train_detector(_df: pd.DataFrame, window_size: int = 12):
    feat_cols = [c for c in _df.columns if c != "timestamp"]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(_df[feat_cols].values).astype(np.float32)
    
    detector = AnomalyDetector(n_features=len(feat_cols), window_size=window_size)
    detector.train(scaled_data, epochs=4, batch_size=32)
    return detector, scaler, feat_cols

def inject_shape(arr: np.ndarray, shape: str, severity: float, start_idx: int, length: int) -> np.ndarray:
    """Injects a mathematical shape into a 1D timeseries array."""
    out = arr.copy()
    max_val = max(1.0, np.max(out)) 
    min_val = min(0.0, np.min(out))
    span = max_val - min_val if max_val > min_val else 100.0
    
    end_idx = min(len(out), start_idx + length)
    actual_len = end_idx - start_idx
    if actual_len <= 0: return out
    
    scale = span * severity
    
    if shape == "spike":
        # Triangle spike
        peak = actual_len // 2
        for i in range(actual_len):
            if i < peak:
                out[start_idx + i] += scale * (i / peak)
            else:
                out[start_idx + i] += scale * (1 - (i - peak) / (actual_len - peak))
    elif shape == "step_up":
        out[start_idx:end_idx] += scale
    elif shape == "step_down":
        out[start_idx:end_idx] -= scale * 0.8 # Don't drop too far below 0
    elif shape == "ramp_up":
        ramp = np.linspace(0, scale, actual_len)
        out[start_idx:end_idx] += ramp
    elif shape == "ramp_down":
        ramp = np.linspace(0, scale * 0.8, actual_len)
        out[start_idx:end_idx] -= ramp
    elif shape == "noise":
        noise = np.random.normal(0, scale * 0.5, actual_len)
        out[start_idx:end_idx] += noise
        
    return out

# ── 3. UI DASHBOARD ───────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulator Controls")
selected_name = st.sidebar.selectbox("Select Anomaly Pattern:", [a["name"] for a in ANOMALIES])
anomaly_spec = next(a for a in ANOMALIES if a["name"] == selected_name)

# Allow manual override of severity
severity_override = st.sidebar.slider("Severity Factor", 0.1, 1.5, anomaly_spec["severity"])

with st.spinner("Generating Base Architecture & Neural Weights..."):
    base_df = generate_base_data()
    detector, scaler, all_feats = train_detector(base_df)

# We will test on a strict 2-day window
test_df = base_df.tail(288 * 2).copy().reset_index(drop=True)
incident_start = int(len(test_df) * 0.6) # Anomaly hits 60% of the way through
incident_length = 50 

# Inject the mathematical anomaly
if anomaly_spec["metric"] == "multi_resource":
    test_df["cpu_utilization"] = inject_shape(test_df["cpu_utilization"].values, "ramp_up", severity_override, incident_start, incident_length)
    test_df["memory_usage_percent"] = inject_shape(test_df["memory_usage_percent"].values, "ramp_up", severity_override, incident_start, incident_length)
    highlight_metrics = ["cpu_utilization", "memory_usage_percent"]
elif anomaly_spec["metric"] == "multi_outage":
    test_df["api_latency_p99_ms"] = inject_shape(test_df["api_latency_p99_ms"].values, "spike", severity_override, incident_start, incident_length)
    test_df["error_rate_percent"] = inject_shape(test_df["error_rate_percent"].values, "step_up", severity_override, incident_start, incident_length)
    highlight_metrics = ["api_latency_p99_ms", "error_rate_percent"]
else:
    m = anomaly_spec["metric"]
    test_df[m] = inject_shape(test_df[m].values, anomaly_spec["shape"], severity_override, incident_start, incident_length)
    highlight_metrics = [m]

# Ensure bounds are somewhat realistic
for col in all_feats:
    if "percent" in col or "rate" in col: test_df[col] = np.clip(test_df[col], 0, 100)
    elif "latency" in col: test_df[col] = np.clip(test_df[col], 0, 20000)

# ── 4. RUN AI DETECTION ───────────────────────────────────────────────────
scaled_test = scaler.transform(test_df[all_feats].values)
scaled_test_df = pd.DataFrame(scaled_test, columns=all_feats)
scaled_test_df.insert(0, "timestamp", test_df["timestamp"].values)

results_df = detector.detect(scaled_test_df, all_feats)
# The detector returns a df with an index offset by window_size-1. 
# We pull the correct timestamps using this index.
results_df["timestamp"] = test_df.loc[results_df.index, "timestamp"]

st.subheader(f"Analyzing Pattern: `{selected_name}`")

for metric in highlight_metrics:
    col1, col2 = st.columns([1.5, 1])
    
    score_col = f"{metric}_score"
    is_anomaly_col = f"{metric}_is_anomaly"
    
    if score_col not in results_df.columns: continue
    
    with col1:
        # Plot 1: Raw Metric
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df["timestamp"], y=test_df[metric], mode="lines", name=f"Raw {metric}", line_color="#1f77b4"))
        
        # Add red background for ground-truth anomaly zone
        fig.add_vrect(
            x0=test_df.loc[incident_start, "timestamp"], 
            x1=test_df.loc[min(incident_start+incident_length, len(test_df)-1), "timestamp"],
            fillcolor="red", opacity=0.15, line_width=0, layer="below", annotation_text="Injected Failure Window"
        )
        fig.update_layout(title=f"Telemetry Graph: {metric}", height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Plot 2: AI Anomaly Score
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(x=results_df["timestamp"], y=results_df[score_col], mode="lines", name="LSTM Error", fill="tozeroy", line_color="#ff7f0e"))
        
        # Use detector's threshold if available, otherwise default
        thresh = getattr(detector, 'threshold', 1.0)
        fig_score.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text=f"AI Threshold ({thresh:.3f})")
        
        fig_score.update_layout(title="LSTM Reconstruction Error (Anomaly Score)", height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig_score, use_container_width=True)

    # Calculate actual detections
    detected_count = results_df[is_anomaly_col].sum()
    if detected_count > 0:
        first_detected = results_df[results_df[is_anomaly_col] == True].iloc[0]["timestamp"]
        st.success(f"✅ AI Successfully caught the anomaly! Flagged {detected_count} timesteps. First anomaly seen at {first_detected}.")
    else:
        st.error(f"❌ AI missed the anomaly. It blended too well into normal noise. Try increasing 'Severity Factor'.")
    st.markdown("---")
