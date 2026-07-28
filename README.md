<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PySide6-Qt%206-41CD52?style=for-the-badge&logo=qt&logoColor=white" />
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
  <a href="#%EF%B8%8F-desktop-app-pyside6">Desktop App</a> •
  <a href="#-methodology">Methodology</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

---

## 📌 Overview

When a machine slows down, stalls, or crashes, finding out why means manually correlating resource graphs against event logs — slow and error-prone. This project automates that, using telemetry collected from the machine it runs on.

The **AI-Powered RCA System** takes multivariate time-series metrics, detects anomalies using deep learning, constructs causal graphs using statistical and constraint-based methods, and produces a **ranked list of root causes** with confidence scores and evidence chains.

> **Built as a Major Project** — demonstrates end-to-end ML pipeline engineering, from real telemetry collection to a packaged desktop application.

---

## ✨ Features

| Category | Capability |
|----------|-----------|
| 🧠 **Deep Learning** | LSTM Autoencoder trained on normal baselines for reconstruction-error–based anomaly detection |
| 📊 **Ensemble Detection** | Combines LSTM (40%), Statistical (35%), and Temporal (25%) detectors to reduce false positives |
| 🔗 **Causal Inference** | Granger Causality tests + Peter-Clark (PC) algorithm for structure learning |
| 🏆 **Root Cause Ranking** | Multi-factor composite scoring with PageRank-augmented graph influence |
| 🎯 **Incident Discovery** | Incidents found from the detector or triggered by Event Log faults — never injected |
| 📉 **Dimensionality Reduction** | Flatline filtering + hierarchical correlation grouping for high-cardinality metrics |
| 🔄 **Model Staleness** | Warns and offers retraining when reconstruction error drifts from its training-time reference |
| 🖥️ **Desktop App** | PySide6 UI with live training, causal graph visualization, and downloadable reports |

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
│  │ • psutil     │    │ • LSTM AE    │    │  • PC Algorithm       │  │
│  │ • Event Log  │    │ • Statistical│    │  • Event Correlation  │  │
│  │ • SQLite     │    │ • Temporal   │    │  • Topology Prior     │  │
│  │   store      │    │ • Ensemble   │    │  • Graph Builder      │  │
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
│                                          │  • PySide6 Desktop UI │  │
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

### Run the desktop application

```bash
python src/desktop/main.py
```

Start the opt-in collector first, then train from three clean days of local
telemetry in Stage 1 and run RCA over an observed time window in Stage 2.

---

## 🖥️ Desktop App (PySide6)

A native desktop application built with PySide6 (Qt 6), sharing the pipeline
engine with the CLI via `src/pipeline/engine.py`. Native widgets for controls
and tables, with an embedded fully offline Plotly view for the causal graph and
anomaly timeline.

### Stage 1 — Baseline & Training
- Shows collection status: clean days available, days remaining
- Trains the LSTM Autoencoder once a clean 3-day baseline exists
- Writes a versioned model artifact (feature order, scaler, thresholds)

### Stage 2 — RCA Inference
- Analyses an observed window of real telemetry
- Runs the pipeline: preprocessing → anomaly detection → Granger causality
  (BH-FDR corrected) → topology pruning → root-cause ranking → attribution
- Results: ranked root causes, causal graph, anomaly timeline, and Markdown
  and JSON reports

> Reports state evidence honestly — there is no ground truth for a real
> incident. Confidence comes from the composite score, attribution coverage,
> and how many causal edges survived correction.

### Run from source

```bash
pip install -r requirements.txt
cd src
python -m desktop.main
```

### Build a standalone .exe

```powershell
.\packaging\build.ps1
```

Output: `dist\RCA-Desktop\RCA-Desktop.exe`

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
│   ├── telemetry/                      # Real telemetry collection (no synthetic data)
│   │   ├── collector.py               # Sampling loop, burst logic, consent gate
│   │   ├── sampler.py                 # psutil system + per-process snapshots
│   │   ├── eventlog.py                # Windows Event Log, per-channel watermark
│   │   ├── store.py                   # SQLite schema and read/write APIs
│   │   ├── analysis.py                # Baseline selection, gaps, clean windows
│   │   ├── rates.py                   # Monotonic counter differencing
│   │   ├── redaction.py               # Best-effort text redaction
│   │   ├── schedule.py                # Startup-folder autostart registration
│   │   └── __main__.py                # CLI: consent, install, run, status, delete
│   │
│   ├── data_ingestion/
│   │   ├── log_integrator.py          # Log data ingestion (unused, see note)
│   │   └── imputer.py                 # Missing data imputation (unused, see note)
│   │
│   ├── models/
│   │   └── lstm_autoencoder.py        # LSTM Autoencoder + AnomalyDetector
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
│   │   └── dynamic_graph.py           # Subsystem topology prior
│   │
│   ├── root_cause_ranking/
│   │   └── scorer.py                  # Multi-factor composite scorer
│   │
│   ├── reporting/
│   │   └── report_generator.py        # Markdown report builder
│   │
│   ├── pipeline/
│   │   ├── engine.py                  # Shared GUI-agnostic pipeline
│   │   └── visualizations.py          # Plotly figure builders
│   │
│   ├── desktop/                        # PySide6 desktop app
│   │   ├── main_window.py             # Two-stage window shell
│   │   ├── workers.py                 # QThread training / RCA workers
│   │   ├── views/                     # Stage 1 and Stage 2 panels
│   │   └── theme.py                   # Qt stylesheet
│   │
│   └── train_and_run.py               # CLI training + inference script
│
├── packaging/
│   ├── rca_desktop.spec               # PyInstaller spec
│   ├── excludes.txt                   # Generated module exclude list
│   ├── hooks/hook-torch.py            # torch 2.12 Windows build workaround
│   └── build.ps1                      # Build script
│
├── docs/superpowers/
│   ├── specs/                          # Approved design documents
│   └── plans/                          # Implementation plans
│
├── Dockerfile                          # Container deployment
├── requirements.txt                    # Python dependencies (CPU-optimized)
└── best_autoencoder_model.pt           # Legacy weights, superseded by the
                                        # versioned artifact in %LOCALAPPDATA%\RCA
```

> **Note:** `data_ingestion/log_integrator.py` and `data_ingestion/imputer.py`
> are currently unused — the telemetry collector reads the Event Log directly.
> They are kept rather than deleted because nothing in the current design
> replaces their intent.

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

## Telemetry collector

The RCA pipeline can collect real laptop telemetry locally. Collection is opt-in;
window titles are never captured and data is never sent over the network.

```powershell
python -m telemetry accept-consent
python -m telemetry install
python -m telemetry status
python -m telemetry uninstall
python -m telemetry delete-all-data
```

Training requires about three days of collected samples. Event message text is
off by default; use `python -m telemetry run --capture-messages` only when you
explicitly want redacted EventData values retained locally.

## 🌐 Deployment

> **Note:** this application analyses telemetry from the machine it runs on,
> so it is distributed as a desktop build rather than hosted. See
> `packaging/build.ps1` for the Windows executable.

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
| Desktop UI | PySide6 (Qt 6) + Plotly |
| Data Processing | Pandas, NumPy, scikit-learn |
| Telemetry Sources | psutil, Windows Event Log (pywin32), SQLite |

---

## 📄 License

This project is developed as part of a college major project.

---

<p align="center">
  <b>Built with ❤️ using PyTorch, PySide6, and Causal Inference</b>
</p>
