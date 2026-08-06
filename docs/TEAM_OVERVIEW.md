# AI-Powered Root Cause Analysis (RCA) System — Team Overview

> Everything you need to know about the project: what it does, how it works, exact numbers, and how to run it.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Collection (Telemetry)](#3-data-collection-telemetry)
4. [Stage 1 — Training the Model](#4-stage-1--training-the-model)
5. [Stage 2 — Running RCA (Root Cause Analysis)](#5-stage-2--running-rca-root-cause-analysis)
6. [The Causal Inference Engine](#6-the-causal-inference-engine)
7. [Root Cause Ranking](#7-root-cause-ranking)
8. [Reports & Visualization](#8-reports--visualization)
9. [Desktop Application](#9-desktop-application)
10. [Project Structure](#10-project-structure)
11. [How to Run Everything](#11-how-to-run-everything)
12. [Key Numbers at a Glance](#12-key-numbers-at-a-glance)
13. [FAQ for Teammates](#13-faq-for-teammates)

---

## 1. What This Project Does

When a Windows laptop or desktop slows down, stalls, or crashes, figuring out *why* usually means manually lining up resource graphs (CPU, memory, disk) against the Windows Event Log and guessing. This tool does that automatically.

**The system:**

1. **Collects** system metrics (CPU, memory, disk, network, GPU, power) every 30 seconds, and Windows Event Log entries every 5 minutes — all stored in a local SQLite database.
2. **Learns** what "normal" looks like for *this specific machine* using an LSTM autoencoder (a type of neural network trained on clean data only).
3. **Detects** when something goes wrong by scoring new data against the learned normal.
4. **Figures out causality** using Granger causality tests — which metric caused which other metric to go wrong.
5. **Ranks root causes** using a multi-factor scoring system (causal outflow, temporal priority, anomaly severity, etc.).
6. **Generates reports** in Markdown and JSON formats.

**Key principle:** No synthetic data. Every number the model sees was actually measured on the machine. No data leaves the device.

---

## 2. High-Level Architecture

```
┌────────────────┐      ┌──────────────┐      ┌──────────────────────┐
│   Collector     │─────▶│ telemetry.db │◀─────│   Desktop App        │
│   (headless)    │      │   SQLite     │      │   PySide6 (Qt 6)     │
└────────────────┘      └──────────────┘      └──────────────────────┘
   psutil   30s              WAL                  Train / Analyse
   processes 5min
   Event Log 5min
```

**The collector never calls the detector.** The app is only a reader. This means always-on monitoring can be added later without changing how data is produced.

### The Pipeline (4 Stages)

| Stage | What happens | Exact values |
|-------|-------------|--------------|
| **Data Ingestion** | Collect CPU, memory, disk, network, GPU, power metrics + Windows events | 30s cadence, 25 channels, 18 modelled |
| **Stage 1 — Baseline** | Exclude gaps and crash-adjacent samples, train LSTM autoencoder | Needs 250 training windows across clean segments |
| **Stage 2 — Detection** | LSTM autoencoder scores incoming data; >99th percentile = anomaly | 99th percentile threshold per metric |
| **Causal Inference** | Granger tests + Benjamini-Hochberg FDR + effect-size floor + topology pruning | Max lag=5, sig level=0.05, min effect=0.10 |
| **Root Cause Ranking** | Multi-factor scoring + PageRank on reversed causal graph | 40% outflow, 30% temporal, 20% inflow, 5% severity, 5% event |
| **Reports** | Markdown + JSON with causal graph, evidence, process attribution | Saved to `%LOCALAPPDATA%/RCA/reports/` |

---

## 3. Data Collection (Telemetry)

### What Gets Collected

| Group | Channels | Cadence |
|-------|----------|---------|
| **CPU** | utilisation, busiest core, frequency, frequency ratio, busy time | 30s |
| **GPU** | utilisation, memory used, temperature (via NVML) | 30s |
| **Memory** | used %, available, swap %, swap in use, swap change | 30s |
| **Disk** | read/write rate, busy %, free space | 30s |
| **Network** | sent, received | 30s |
| **Load** | process count | 30s |
| **Power** | charge, drain rate, on mains | 30s |
| **Context** | idle time, foreground executable name | 30s |
| **Processes** | top 15 by CPU ∪ top 15 by RSS | 5 min |
| **Events** | Kernel-Power 41, app crashes/hangs, disk faults, WHEA, resource exhaustion, Windows Update, MSI installs | 5 min |

### Modelled Columns (18 total, used by the ML model)

```
cpu_pct, cpu_pct_max_core, cpu_freq_mhz, cpu_freq_ratio,
mem_pct, mem_available_mb, swap_pct, swap_used_bytes, swap_used_delta,
disk_read_bps, disk_write_bps, disk_busy_pct, disk_free_pct,
net_sent_bps, net_recv_bps, process_count, battery_pct,
battery_drain_rate, power_plugged
```

### Storage

| Item | Value |
|------|-------|
| Database format | SQLite with WAL mode |
| Database location | `%LOCALAPPDATA%/RCA/telemetry.db` |
| Process snapshot retention | 30 days |
| Event retention | 365 days |
| GPU temperature source | NVML (only temperature source on Windows) |

### Privacy

- **Window titles are never captured.** Only the foreground executable name.
- **Event message text is not stored** unless you pass `--capture-messages`. Provider, ID, level, and time carry everything the analysis needs.
- **The collector opens no sockets.** Everything stays on the machine.
- `python -m telemetry delete-all-data` stops collection, removes the startup
  entry, and erases the entire data directory: the database, the trained model,
  every generated report, and the collector log — which records exception
  tracebacks containing your profile path.

### Allowed Windows Events

| Provider | Event IDs | Notes |
|----------|-----------|-------|
| Microsoft-Windows-Kernel-Power | 41 | Unexpected shutdown |
| Application Error | 1000 | App crashes |
| Application Hang | 1002 | App hangs |
| disk | 7, 51, 153 | Disk faults |
| Microsoft-Windows-WHEA-Logger | *all* | Hardware errors |
| Microsoft-Windows-Resource-Exhaustion-Detector | 2004 | Resource exhaustion |
| Microsoft-Windows-WindowsUpdateClient | *all* | Change events (not failures) |
| MsiInstaller | *all* | Change events (not failures) |

---

## 4. Stage 1 — Training the Model

### What the Model Is

An **LSTM autoencoder** — a neural network that learns to compress and reconstruct normal system metrics. It has:

| Parameter | Value |
|-----------|-------|
| Input features | Up to 18 modelled columns (varies per machine) |
| LSTM hidden size | 64 |
| Number of LSTM layers | 2 |
| Latent (bottleneck) size | 32 |
| Dropout | 0.2 |
| Total parameters | ~129,000 |
| Model size on disk | 0.52 MB |
| Default window size | 12 samples (= 6 minutes at 30s cadence) |
| Default epochs | 5 (configurable 1–30) |
| Learning rate | 0.001 |
| Batch size | 32 |
| Validation split | 20% |
| Default stride (training) | 5 samples |
| Threshold percentile | 99th per metric |
| Training time | ~24 seconds on 20-core laptop |

### Training Requirements

| Requirement | Value |
|-------------|-------|
| Minimum training windows | 250 |
| Training stride | 5 |
| Formula for required samples | `window_size + 250 * 5` = **1,262 uninterrupted samples** (at default 12 window) |
| Equivalent collection time | ~10.5 hours at 30s cadence (if uninterrupted) |
| Actual needed (with gaps) | ~21 hours typical due to sleep/restart gaps |
| Bool gate | `total_windows >= 250` across ALL clean segments |

### How Training Works

1. The collector database is read for system samples and Windows events.
2. `clean_baseline()` filters out:
   - Samples within gaps (missed cadence > 45s)
   - Samples within 60 minutes before and 15 minutes after a bad event (crash, disk fault, WHEA)
   - Rows where modelled columns are null
3. `contiguous_windows()` splits the clean history at each collector gap. **No model window ever spans a gap.**
4. Windows (sequences of 12 samples) are built inside each clean segment with a stride of 5.
5. The LSTM autoencoder is trained on these windows.
6. Thresholds are calibrated at the 99th percentile of reconstruction error on the validation set.
7. The model artifact is saved as a `.pt` file containing:
   - State dictionary (model weights)
   - Feature columns list
   - Per-metric thresholds
   - MinMaxScaler parameters (min, scale, data_min, data_max, data_range)
   - Training timestamp
   - Reference reconstruction error (for staleness detection)

### Model Staleness

| Parameter | Value |
|-----------|-------|
| Staleness ratio threshold | 2.0x |
| What it means | If current median reconstruction error > 2x the reference error at training time, the model is stale |
| What happens | RCA is blocked until retraining; report still labels it clearly |

---

## 5. Stage 2 — Running RCA (Root Cause Analysis)

### Detection Process

1. A time window is selected (detected incident or custom range).
2. The largest clean segment within that window is used.
3. Metrics are scaled using the same MinMaxScaler from training.
4. The LSTM autoencoder scores each metric against its 99th percentile threshold.
5. Any metric exceeding its threshold is flagged as "anomalous."
6. A metric is anomalous if: `reconstruction_error > threshold_per_metric[i]`

### Incident Discovery

Two independent triggers find incidents:

| Trigger | How it works | Parameters |
|---------|-------------|------------|
| **Detector** | Contiguous runs of detector-flagged rows (at least 3 consecutive) | `min_consecutive=3` |
| **Event** | Windows Event Log fault (crash, WHEA, disk fault) defines a window | 30 min lead, 5 min tail |
| **Merge** | Overlapping or near-adjacent windows merged | Merge gap = 5 min |

---

## 6. The Causal Inference Engine

### Granger Causality

For every pair of anomalous metrics (A, B), we ask: *Does the past of A help predict B better than B's own past alone?*

| Parameter | Value |
|-----------|-------|
| Maximum lag | 5 samples (= 2.5 minutes at 30s cadence) |
| Significance level | 0.05 |
| Minimum effect size | 0.10 (F-test based) |
| Multiple testing correction | Benjamini-Hochberg FDR |
| Stationarity check | ADF test; first-order differencing if needed (max 2 rounds) |
| Minimum data requirement | `len(aligned) > max_lag * 3` |

**Only pairs where `p < 0.05` AND `effect_size >= 0.10` AND survive FDR correction are kept.**

### Causal Graph Construction

1. Start with all significant Granger pairs as directed edges.
2. Add nodes with anomaly scores as metadata.
3. Break cycles:
   - Prefer removing edges that violate temporal precedence (effect appeared before cause).
   - Fallback: remove the weakest edge (lowest Granger strength).

### Topology Pruning (Dynamic Graph)

Because Granger tests can produce impossible relationships (e.g., network packet rate causing CPU frequency), the graph is pruned against a static laptop subsystem topology:

```
Subsystems:
  cpu:      cpu_pct, cpu_pct_max_core, cpu_freq_mhz, cpu_freq_ratio
  memory:   mem_pct, mem_available_mb, swap_pct, swap_used_bytes, swap_used_delta
  disk:     disk_read_bps, disk_write_bps, disk_busy_pct, disk_free_pct
  network:  net_sent_bps, net_recv_bps
  power:    battery_pct, battery_drain_rate, power_plugged
  process:  process_count

Allowed dependencies (directed):
  power → cpu, power → memory, power → disk
  cpu → memory, cpu → disk, cpu → network
  memory → disk, memory → network
  disk → network
  process → cpu, process → memory, process → disk, process → network
```

An edge is kept **only if** the source subsystem can reach the target subsystem in this directed graph (or they are in the same subsystem).

### Event Correlation

Windows events (crashes, WHEA, resource exhaustion) within 24 hours before an anomaly are correlated. Scoring: `1.0 / (1.0 + delta_hours)`. Closer events get higher scores.

---

## 7. Root Cause Ranking

### Scoring Factors

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| **Causal Outflow** | 40% | How many downstream metrics does this cause? More = likely root cause |
| **Temporal Priority** | 30% | How early did this anomaly appear? Earlier = likely root cause |
| **Causal Inflow (penalty)** | 20% | Fewer upstream causes = likely root cause |
| **Anomaly Severity** | 5% | How far from normal is this metric? |
| **Event Correlation** | 5% | Did a Windows event precede this anomaly? |

**Final formula:** `0.70 * weighted_sum + 0.30 * PageRank_on_reversed_graph`

### Confidence Levels

| Score Range | Label |
|-------------|-------|
| 95–100% | Critical |
| 85–95% | High |
| 70–85% | Medium |
| 50–70% | Low |
| < 50% | Very Low |

### Process Attribution

Process snapshots (top 15 by CPU ∪ top 15 by RSS, captured every 5 minutes) are aggregated over the incident window. The top 10 processes by CPU usage + I/O are reported. System Idle Process is excluded (it always ranks highest and would name "doing nothing" as the cause).

---

## 8. Reports & Visualization

### Markdown Report

Generated with: incident ID, timestamp, executive summary (primary root cause + confidence), causal chain, detailed evidence scoring (temporal priority, severity, causal outflow, PageRank), alternative root causes, process attribution table.

### JSON Report

Structured with: incident metadata, root causes (rank, metric, composite_score, confidence, scores_breakdown, downstream_effects, causal_chain), causal graph (nodes + edges with strength/lag/p_value), event correlations, process attribution, anomaly detection times.

### Plotly Figures (Desktop App)

| Figure | Description |
|--------|-------------|
| **Causal Graph** | Interactive directed graph with node colors (red=root cause, orange=source, blue=intermediate), edge width encoding strength, hover for details |
| **Anomaly Timeline** | Top-5 anomalous metrics over time with vertical markers at first anomaly detection time |

---

## 9. Desktop Application

### Tech Stack

| Component | Technology |
|-----------|-----------|
| UI framework | PySide6 (Qt 6.7+) |
| Charts | Plotly via QWebEngineView |
| Background workers | QThread (TrainWorker, DetectIncidentsWorker, InferenceWorker) |
| Window size | 1400 × 900 pixels |
| Theme | Dark (navy `#0f1628` background, indigo `#667eea` accent, `#e2e8f0` text) |
| Font | "Segoe UI", "Inter", sans-serif (10.5pt base) |
| Console | Monospace Consolas, green text on dark background |

### App Layout

```
┌──────────────────────────────────────────────────────────────┐
│  🔍 AI-Powered Root Cause Analysis                           │
│  Diagnose slowdowns, stalls and crashes on this machine...   │
├──────────────┬───────────────────────────────────────────────┤
│              │                                               │
│  Captured    │  Shows collected store stats:                 │
│  Data        │  - Sample counts, database size, span        │
│              │  - Channel table (group, value, unit, model)  │
│              │  - Auto-refresh every 30 seconds              │
├──────────────┼───────────────────────────────────────────────┤
│  1 —         │  Collection status + training params:         │
│  Baseline    │  - Epochs slider (1–30, default 5)           │
│  & Training  │  - Window size slider (6–60, default 12)     │
│              │  - Progress bar + log console                 │
├──────────────┼───────────────────────────────────────────────┤
│  2 —         │  Incident selection + RCA controls:           │
│  Run RCA     │  - Incident combo box / custom range          │
│  Inference   │  - Granger max lag spinner (2–10, default 5)  │
│              │  - Results: table, graph, timeline, report    │
│              │  - Export: Markdown / JSON buttons            │
└──────────────┴───────────────────────────────────────────────┘
```

### Tab Details

#### Captured Data Tab
- "Collected Store" group box: samples count, process samples, events, gaps, collection span, database size, database path
- "Captured Channels" table: 26 channels across 9 groups, showing latest value and whether the channel is used by the model
- Refresh button + auto-refresh every 30s

#### Stage 1 Tab
- "Collection Status": clean samples collected, longest uninterrupted run, remaining until trainable, current model age
- "Training Parameters": LSTM epochs slider (1–30, default 5), window size slider (6–60, default 12)
- Primary action button (gradient styled): "Train from Clean Collected Telemetry"
- Progress bar + status label + log console (green terminal-style)
- Auto-refresh collection status every 30s

#### Stage 2 Tab
- **Disabled until Stage 1 completes training.**
- "Observed Incident Window" group: incident combo box (detected incidents + custom range), datetime range pickers, Granger max lag spinner (2–10, default 5)
- "Find Incidents" button scans last 168 hours
- "Run RCA on Collected Telemetry" primary button
- Results tab widget with 4 tabs: Root Causes (table), Causal Graph (Plotly), Anomaly Timeline (Plotly), Report (text)
- Export: Markdown + JSON buttons
- Model staleness warning when drift > 2x reference error

---

## 10. Project Structure

```
majorprojectt/
├── src/
│   ├── telemetry/              # Collection: no analysis, no ML
│   │   ├── collector.py        # Collection loop, consent gate
│   │   ├── sampler.py          # psutil + NVML snapshots
│   │   ├── eventlog.py         # Windows Event Log, per-channel watermark
│   │   ├── store.py            # SQLite schema + migrations
│   │   ├── analysis.py         # Baseline selection, gaps, incidents
│   │   ├── config.py           # All collection constants
│   │   ├── schedule.py         # Autostart registration
│   │   └── __main__.py         # CLI entry point
│   ├── models/
│   │   └── lstm_autoencoder.py # LSTM autoencoder + training
│   ├── anomaly_detection/
│   │   ├── ensemble_detector.py # LSTM + statistical + temporal ensemble
│   │   ├── anomaly_scorer.py
│   │   ├── alert_dampener.py
│   │   └── dimensionality_reduction.py
│   ├── causal_inference/
│   │   ├── causal_engine.py    # Granger + graph builder + ranker
│   │   ├── dynamic_graph.py    # Subsystem topology pruning
│   │   ├── granger_causality.py
│   │   └── pc_algorithm.py
│   ├── root_cause_ranking/
│   │   └── scorer.py
│   ├── pipeline/
│   │   ├── engine.py           # Shared pipeline (GUI-agnostic)
│   │   └── visualizations.py   # Plotly figure builders
│   ├── reporting/
│   │   └── report_generator.py # Markdown + JSON reports
│   └── desktop/
│       ├── main.py             # App entry point
│       ├── main_window.py      # Main window with tabs
│       ├── theme.py            # Dark QSS stylesheet
│       ├── state.py            # Shared application state
│       ├── workers.py          # QThread workers (train, inference)
│       └── views/
│           ├── data_view.py    # Captured Data tab
│           ├── stage1_view.py  # Stage 1 training tab
│           ├── stage2_view.py  # Stage 2 RCA tab
│           └── graph_panel.py  # Plotly QWebEngineView widget
├── tests/
│   ├── conftest.py
│   ├── test_desktop_smoke.py
│   ├── test_engine_lifecycle.py
│   ├── test_pipeline_engine.py
│   ├── telemetry/
│   │   ├── test_analysis.py
│   │   ├── test_collector.py
│   │   ├── test_eventlog_schedule.py
│   │   ├── test_incidents.py
│   │   └── test_store_rates_redaction.py
├── packaging/
│   ├── build.ps1               # Builds both RCA-Desktop and RCA-Collector
│   ├── rca_desktop.spec        # PyInstaller spec
│   ├── runtime_hook.py         # Survives torch's missing-source import crash
│   ├── excludes.txt
│   └── hooks/hook-torch.py
├── docs/
│   ├── screenshots/            # App screenshots (3 files)
│   ├── UI_overview.md
│   └── Repository_Overview.md
├── README.md
├── requirements.txt
├── PRD.md                      # Product Requirements Document
└── PRD1.md                     # Additional PRD material
```

---

## 11. How to Run Everything

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs: torch (CPU), pandas, numpy, scikit-learn, statsmodels, networkx, plotly, PySide6, pytest-qt, pyinstaller, psutil, pywin32, nvidia-ml-py

### 2. Start Collecting

```bash
cd src
python -m telemetry accept-consent   # States exactly what will be recorded
python -m telemetry install          # Registers to run at every logon
python -m telemetry status           # Shows consent, schedule, sample count
```

Training needs **at least 250 clean windows** (roughly 21 hours of real collection). Check readiness with `python -m telemetry status`.

### 3. Launch the Desktop App

```bash
python -m desktop.main
```

### 4. Build Standalone Executable

```powershell
.\packaging\build.ps1
# → dist/RCA-Desktop/RCA-Desktop.exe
# → dist/RCA-Collector/RCA-Collector.exe   (runs at logon, GUI closed)
```

### 5. Run Tests (54 tests)

```bash
python -m pytest tests/ -q
```

### 6. Verify No Synthetic Data

```bash
grep -rn "SyntheticMetricsGenerator\|generate_data\|inject_failure" src/
grep -rn "np\.random\|torch\.randn" src/
```

Both must return nothing.

---

## 12. Key Numbers at a Glance

| What | Value |
|------|-------|
| Collection cadence | 30 seconds |
| Process snapshot cadence | 5 minutes |
| Event poll cadence | 5 minutes |
| Total metric channels | 25 (18 modelled) |
| LSTM hidden size | 64 |
| LSTM layers | 2 |
| Bottleneck size | 32 |
| Model parameters | ~129,000 |
| Model size | 0.52 MB |
| Default window size | 12 samples (6 min) |
| Training stride | 5 samples |
| Min training windows | 250 |
| Min uninterrupted samples | 1,262 |
| Typical training data needed | ~21 hours |
| Training time | ~24 seconds |
| Epochs default | 5 |
| Epochs range | 1–30 |
| Window size range | 6–60 |
| Threshold percentile | 99th |
| Staleness ratio | 2.0x |
| Granger max lag default | 5 samples |
| Granger max lag range | 2–10 |
| Significance level | 0.05 |
| Min effect size | 0.10 |
| FDR method | Benjamini-Hochberg |
| Event correlation window | 24 hours |
| Scoring weights | 40/30/20/5/5 |
| PageRank blend | 30% |
| Desktop window | 1400 × 900 |
| App memory | ~455 MB |
| Packaged build size | ~1.5 GB (torch = 628 MB) |
| Process snapshot retention | 30 days |
| Event retention | 365 days |
| Number of tests | 54 |

---

## 13. FAQ for Teammates

### Q: Why not use synthetic data?
Real telemetry is the project's core integrity guarantee. Every number the model sees was measured on the actual machine. This is verifiable with two grep commands.

### Q: Why does training need 21 hours?
The formula is `window_size + 250 * stride` = 1,262 uninterrupted samples at default settings. At 30s cadence that's ~10.5 hours straight, but laptops sleep/go to standby, so in practice ~21 hours of wall-clock time is typical.

### Q: Can I change the training parameters?
Yes. The desktop app has sliders for epochs (1–30, default 5) and window size (6–60, default 12). The CLI accepts `--epochs` and `--window-size` flags.

### Q: What happens if there are no anomalies?
If no metric exceeds the 99th percentile threshold, the report says "No supported causal chain" and makes no causal claim. This is by design — the system is honest about uncertainty.

### Q: Why is Granger max lag default 5?
At 30s cadence, 5 samples = 2.5 minutes. Most resource contention effects on a single machine propagate within this window. Longer lags increase the risk of spurious correlations.

### Q: Can the model detect slow drift (disk filling over weeks)?
No. Slow drift is explicitly out of scope. The LSTM autoencoder detects acute resource exhaustion. A trend detector would be needed for gradual changes.

### Q: Why limit torch to 4 threads?
The per-operation work is tiny (60-step LSTM, hidden=64). Thread dispatch overhead costs more than parallelism gains at higher thread counts. Measured: 3.9× slower at 20 threads than at 4.

### Q: No GPU for training?
Correct. The model is 0.52 MB. A GPU sits idle between tiny kernel launches, and the CUDA wheel adds ~2 GB to the build. CPU capped at 4 threads is faster.

### Q: Where is the database?
`%LOCALAPPDATA%/RCA/telemetry.db`

### Q: How do I delete all collected data?
```bash
python -m telemetry delete-all-data
```
This erases the whole `%LOCALAPPDATA%/RCA` directory — the database, the
trained model, and every generated report — and removes the startup entry, so
collection does not resume at the next logon until you run `install` again.
Retraining afterwards needs a fresh baseline. If the collector does not release
the database within 35 seconds the command deletes nothing and exits non-zero.

### Q: Does anything leave the machine?
No. Exported reports contain process names, so if you share a report file, that's the only thing that leaves.
