<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔍 AI-Powered Root Cause Analysis (RCA) System</h1>

<p align="center">
  <b>Automated incident diagnosis using LSTM Autoencoders, Granger Causality, and Multi-Factor Root Cause Scoring</b>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#%EF%B8%8F-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-dashboard">Dashboard</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

---

## 📌 Overview

Modern distributed systems generate thousands of metrics — when something breaks, manually sifting through dashboards to find the root cause is slow and error-prone. This project automates that process.

The **AI-Powered RCA System** takes multivariate time-series metrics, detects anomalies using deep learning, constructs causal graphs using statistical and constraint-based methods, and produces a **ranked list of root causes** with confidence scores and evidence chains.

> **Built as a Major Project** — designed to demonstrate end-to-end ML pipeline engineering, from data generation to interactive deployment.

---

## ✨ Features

| Category | Capability |
|----------|-----------|
| 🧠 **Deep Learning** | LSTM Autoencoder trained on normal baselines for reconstruction-error–based anomaly detection |
| 📊 **Ensemble Detection** | Combines LSTM (40%), Statistical (35%), and Temporal (25%) detectors to reduce false positives |
| 🔗 **Causal Inference** | Granger Causality tests + Peter-Clark (PC) algorithm for structure learning |
| 🏆 **Root Cause Ranking** | Multi-factor composite scoring with PageRank-augmented graph influence |
| 🎯 **Failure Simulation** | 6 realistic failure scenarios with configurable severity for validation |
| 📉 **Dimensionality Reduction** | Flatline filtering + hierarchical correlation grouping for high-cardinality metrics |
| 🔄 **Concept Drift Handling** | Automated model fine-tuning after deployments (soak period detection) |
| 🖥️ **Interactive Dashboard** | Streamlit UI with live training, causal graph visualization, and downloadable reports |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RCA Pipeline                                 │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │    Data       │    │   Anomaly    │    │   Causal Inference    │  │
│  │  Ingestion    │───▶│  Detection   │───▶│                       │  │
│  │              │    │              │    │  • Granger Causality  │  │
│  │ • Synthetic  │    │ • LSTM AE    │    │  • PC Algorithm       │  │
│  │ • Prometheus │    │ • Statistical│    │  • Event Correlation  │  │
│  │ • CloudWatch │    │ • Temporal   │    │  • Graph Builder      │  │
│  │ • Logs       │    │ • Ensemble   │    │                       │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│                                          ┌───────────▼───────────┐  │
│                                          │   Root Cause Ranker   │  │
│                                          │                       │  │
│                                          │  • Causal Outflow     │  │
│                                          │  • Temporal Priority  │  │
│                                          │  • Anomaly Severity   │  │
│                                          │  • PageRank Score     │  │
│                                          │  • Event Correlation  │  │
│                                          └───────────┬───────────┘  │
│                                                       │             │
│                                          ┌───────────▼───────────┐  │
│                                          │     Reporting         │  │
│                                          │                       │  │
│                                          │  • Streamlit UI       │  │
│                                          │  • MD/JSON Reports    │  │
│                                          │  • Causal Graph Viz   │  │
│                                          └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/mathuryashash/RCA-Major_project.git
cd RCA-Major_project

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
cd src
streamlit run reporting/dashboard.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🖥️ Dashboard

The interactive Streamlit dashboard provides a two-stage workflow:

### Stage 1 — Data Generation & Training
- Generate synthetic normal-behavior metrics (configurable duration)
- Train the LSTM Autoencoder with real-time progress
- Visualize baseline metric patterns

### Stage 2 — RCA Inference
- Inject one of **6 failure scenarios** with adjustable severity
- Run the full 5-step pipeline automatically:
  1. Data Generation (normal + failure)
  2. Preprocessing & Scaling
  3. LSTM Autoencoder Training
  4. Anomaly Detection
  5. Causal Inference & Root Cause Ranking
- View results across 5 tabs:
  - 🏆 **Root Causes** — Ranked table with ground truth comparison
  - 🕸️ **Causal Graph** — Interactive Plotly visualization
  - 📊 **Anomaly Timeline** — Top-5 anomalous metric trends
  - 📄 **Markdown Report** — Human-readable incident summary
  - 🗂️ **JSON Report** — Machine-readable output (downloadable)

---

## 🧪 Methodology

### 1. LSTM Autoencoder (Anomaly Detection)

The system uses a **sequence-to-sequence LSTM Autoencoder** trained exclusively on normal operational data. During inference, high reconstruction error indicates anomalous behavior.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 64 | LSTM hidden units |
| `n_layers` | 2 | Stacked LSTM depth |
| `latent_size` | 32 | Bottleneck dimension |
| `dropout` | 0.2 | Regularization |
| `window_size` | 12 | Sliding window length |

**Threshold calibration** uses the 99th percentile of reconstruction error on a held-out validation split.

### 2. Ensemble Detection

Three complementary detectors are combined to reduce false positives:

| Detector | Weight | Method |
|----------|--------|--------|
| LSTM Autoencoder | 40% | Deep learning reconstruction error |
| Statistical | 35% | Z-Score + IQR + Median Absolute Deviation |
| Temporal | 25% | Spike detection + oscillation + trend analysis |

### 3. Causal Inference

Two approaches are used for causal structure learning:

- **Granger Causality** — Pairwise statistical tests with ADF stationarity preprocessing, optimal lag selection, and cycle-breaking
- **PC Algorithm** — Constraint-based structure learning using Fisher's Z conditional independence test (via `causal-learn`)

The resulting edges are assembled into a **directed causal graph** with temporal precedence filtering.

### 4. Root Cause Scoring

Each anomalous metric is scored using a weighted composite:

| Factor | Weight | Description |
|--------|--------|-------------|
| Causal Outflow | 30% | Number of downstream effects |
| Temporal Priority | 25% | How early the anomaly appeared |
| Causal Inflow Penalty | 15% | True root causes have 0 incoming edges |
| Anomaly Severity | 15% | Magnitude of deviation from normal |
| Event Correlation | 15% | Proximity to deployments/changes |

The heuristic score is blended with **PageRank** on the reversed causal graph (70% heuristic / 30% PageRank) for topology-aware ranking.

### 5. Simulated Failure Scenarios

| Scenario | Root Cause Metric | Causal Chain |
|----------|-------------------|-------------|
| `database_slow_query` | db_active_connections | db → latency → throughput → errors |
| `memory_leak` | memory_usage_percent | memory → latency → errors |
| `cpu_spike` | cpu_usage_percent | cpu → latency → throughput |
| `network_partition` | error_rate | errors → latency → throughput |
| `thread_pool_exhaustion` | throughput_rps | throughput → latency → errors |
| `disk_io_spike` | disk_io_bytes | disk → latency → db_connections |

---

## 📂 Project Structure

```
RCA-Major_project/
├── src/
│   ├── data_ingestion/
│   │   ├── synthetic_generator.py     # Generates normal + failure metrics
│   │   ├── prometheus_connector.py    # Prometheus integration
│   │   ├── cloudwatch_connector.py    # AWS CloudWatch integration
│   │   ├── log_integrator.py          # Log data ingestion
│   │   └── imputer.py                 # Missing data imputation
│   │
│   ├── models/
│   │   ├── lstm_autoencoder.py        # LSTM Autoencoder + AnomalyDetector
│   │   └── concept_drift_handler.py   # Post-deployment model fine-tuning
│   │
│   ├── anomaly_detection/
│   │   ├── anomaly_scorer.py          # LSTM-based anomaly scoring
│   │   ├── ensemble_detector.py       # Ensemble (LSTM + Stat + Temporal)
│   │   ├── dimensionality_reduction.py# Flatline filter + correlation grouping
│   │   └── alert_dampener.py          # Alert fatigue reduction
│   │
│   ├── causal_inference/
│   │   ├── causal_engine.py           # Full causal pipeline orchestrator
│   │   ├── granger_causality.py       # Pairwise Granger tests
│   │   ├── pc_algorithm.py            # PC algorithm (causal-learn)
│   │   ├── dynamic_graph.py           # Dynamic graph updates
│   │   ├── deployment_listener.py     # Deployment event tracking
│   │   └── jaeger_connector.py        # Distributed tracing integration
│   │
│   ├── root_cause_ranking/
│   │   └── scorer.py                  # Multi-factor composite scorer
│   │
│   ├── reporting/
│   │   ├── dashboard.py               # Streamlit interactive dashboard
│   │   ├── report_generator.py        # Markdown report builder
│   │   └── anomaly_simulator.py       # Simulation utilities
│   │
│   └── train_and_run.py               # CLI training + inference script
│
├── rca-system/                         # Alternate modular implementation
│   ├── src/                            # Preprocessing pipeline, configs
│   └── tests/                          # Unit & integration tests
│
├── .streamlit/
│   └── config.toml                     # Streamlit deployment config
│
├── Dockerfile                          # Container deployment
├── requirements.txt                    # Python dependencies (CPU-optimized)
├── packages.txt                        # System dependencies (Streamlit Cloud)
└── best_autoencoder_model.pt           # Pre-trained model weights
```

---

## ⚙️ Configuration

### Sidebar Controls (Dashboard)

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Baseline Training Days | 10–60 | 30 | Amount of normal data for training |
| LSTM Training Epochs | 1–30 | 5 | Training iterations |
| LSTM Window Size | 6–60 | 12 | Lookback timesteps per window |
| Failure Scenario | 6 types | db_slow_query | Type of injected failure |
| Severity | 0.1–1.0 | 0.8 | Failure intensity multiplier |
| Granger Max Lag | 2–10 | 5 | Maximum causality lag tested |

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Branch: `main` → File: `src/reporting/dashboard.py`
4. Deploy!

### Docker

```bash
docker build -t rca-system .
docker run -p 8501:8501 rca-system
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch (LSTM Autoencoder) |
| Causal Inference | statsmodels (Granger), causal-learn (PC Algorithm) |
| Graph Analysis | NetworkX + PageRank |
| Dashboard | Streamlit + Plotly |
| Data Processing | Pandas, NumPy, scikit-learn |
| Monitoring Connectors | Prometheus, AWS CloudWatch, Jaeger |

---

## 📄 License

This project is developed as part of a college major project.

---

<p align="center">
  <b>Built with ❤️ using PyTorch, Streamlit, and Causal Inference</b>
</p>
