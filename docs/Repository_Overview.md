# Repository Overview – AI‑Powered Root‑Cause Analysis (RCA) System

## High‑Level Architecture
The project implements an end‑to‑end pipeline that:
1. **Generates synthetic metric data** (or ingests real telemetry) – `src/data_ingestion/synthetic_generator.py`.
2. **Trains a multivariate LSTM auto‑encoder** on normal data – `src/models/lstm_autoencoder.py`.
3. **Detects anomalies** by measuring reconstruction error – `src/models/lstm_autoencoder.py` (via `AnomalyDetector.detect`).
4. **Runs a causal‑inference pipeline** that:
   * Performs pair‑wise **Granger causality** tests on the anomalous metrics – `src/causal_inference/causal_engine.py` → `GrangerAnalyzer`.
   * Builds a **directed causal graph** (DAG) from significant test results – `CausalGraphBuilder`.
   * **Correlates** metric anomalies with external events (deployments, config changes) – `EventCorrelator`.
   * **Ranks root‑cause candidates** using a weighted composite score plus PageRank – `RootCauseRanker`.
5. **Generates a Markdown report** – `src/reporting/report_generator.py`.
6. **Provides an interactive UI** built with **Streamlit** that lets a user:
   * Generate synthetic data and train the model.
   * Inject a chosen failure scenario (27 predefined anomaly shapes).
   * Visualise raw metrics, anomaly scores, and the AI‑detected windows.
   * See a concise success/failure banner.
   * Export the generated RCA report.

The UI glue lives in `src/reporting/dashboard.py` (the main Streamlit entry point) and `src/reporting/anomaly_simulator.py` (the simulator helper).

---

## Core Packages & Files
| Package | Key Modules | Purpose |
|---------|--------------|---------|
| `src/data_ingestion` | `synthetic_generator.py` | Generates realistic time‑series metrics and can inject failure scenarios (spike, step, ramp, noise, etc.). |
| `src/models` | `lstm_autoencoder.py` | Defines the LSTM‑based auto‑encoder, training utilities, and per‑metric anomaly‑score computation. |
| `src/causal_inference` | `causal_engine.py` | Implements Granger causality, causal graph construction, event correlation, and root‑cause ranking. |
| `src/reporting` | `dashboard.py`, `anomaly_simulator.py`, `report_generator.py` | Streamlit UI, simulator UI, and Markdown report generation. |
| `rca-system` (removed) | Duplicate older copy of the above modules – **no longer used** after cleanup. |

### Detailed File Descriptions
- **`src/data_ingestion/synthetic_generator.py`** – `SyntheticMetricsGenerator` creates normal metric series (CPU, memory, latency, etc.) and provides `inject_failure_scenario` to embed known anomalies for testing.
- **`src/models/lstm_autoencoder.py`** – `LSTMAutoencoder` builds the encoder/decoder; `AnomalyDetector` wraps training, validation, threshold calibration and anomaly detection.
- **`src/causal_inference/causal_engine.py`** – Contains five classes:
  * `GrangerAnalyzer` – pairwise Granger tests on anomalous metrics.
  * `CausalGraphBuilder` – builds a directed graph and removes cycles respecting temporal order.
  * `EventCorrelator` – looks back up to 24 h to match anomalies with events.
  * `RootCauseRanker` – computes a composite score (weights: outflow 40 %, temporal 30 %, inflow 20 %, severity 5 %, event 5 %) and adds PageRank influence.
  * `CausalInferencePipeline` – orchestrates the whole flow end‑to‑end.
- **`src/reporting/report_generator.py`** – Formats the ranked root‑cause list, causal chain, and scoring breakdown into a human‑readable Markdown report.
- **`src/reporting/dashboard.py`** – Streamlit entry point:
  * Sidebar lets the user select an anomaly pattern, severity, and window size.
  * Generates base data, trains the LSTM model (cached), injects the chosen anomaly, runs detection, and then visualises each metric with Plotly charts.
  * Shows success/failure messages based on detection results.
- **`src/reporting/anomaly_simulator.py`** – A lighter UI used for quickly exploring the 27 predefined anomaly shapes; integrates the same data‑generation and detection logic.

---

## Data Flow Summary (Step‑by‑Step)
1. **Synthetic Data** – `SyntheticMetricsGenerator.generate_normal_behavior` → `DataFrame`.
2. **Failure Injection** – `SyntheticMetricsGenerator.inject_failure_scenario` (selected via UI).
3. **Model Training** – `AnomalyDetector.train` on the normal portion of the data (cached with `@st.cache_resource`).
4. **Anomaly Detection** – `AnomalyDetector.detect` on the full series (including injected failure).
5. **Granger Causality** – `GrangerAnalyzer.run` on the anomalous metrics only.
6. **Causal Graph** – `CausalGraphBuilder.build` → directed graph with edge attributes (`strength`, `p_value`, `lag`).
7. **Event Correlation** – optional external events matched via `EventCorrelator.correlate`.
8. **Root‑Cause Ranking** – `RootCauseRanker.rank` → sorted list of candidate causes.
9. **Report Generation** – `ReportGenerator.generate_report` builds a Markdown report.
10. **UI Presentation** – Streamlit displays the raw metric chart, anomaly‑score chart, and a banner summarising detection success.

---

## What Was Removed (Cleanup)
- The entire duplicated `rca-system/` directory (old copy of the source tree) – its files are now obsolete because the canonical code lives under the top‑level `src/` package.
- All `__pycache__` folders (Python byte‑code caches) – regenerated automatically.
- Generated output artefacts in `outputs/` and temporary `progress.txt`.
- Default Git hook sample files under `.git/hooks/`.
- Old Word/PDF documentation (`RCA_System_Documentation refined.*`).

These deletions reduce repo size and eliminate confusion about which code path is active.

---

## How the UI (Streamlit) Fits In
The Streamlit app (`dashboard.py`) is *purely* a visual orchestrator:
- It imports the core classes from `src/` (data generation, model, causal pipeline).
- All heavy computation (training, detection, Granger analysis) is performed server‑side; the UI merely displays the results.
- The sidebar lets the user tweak **severity**, **failure shape**, and **window size**.
- Plotly charts render the raw metric trace with a shaded region indicating the injected failure, and a second chart shows the AI‑computed anomaly score with a threshold line.
- A success banner (`st.success`) appears when the model flags the injected anomaly; otherwise an error banner informs the user to adjust parameters.
- The UI can also download the generated Markdown report via `st.download_button` (code not shown here but easy to add).

---

## How to Share This Overview
The file `docs/Repository_Overview.md` (the one you are reading) lives in the repository and can be opened directly on GitHub or shared as a PDF (`pandoc` conversion, etc.).

**File location:** `docs/Repository_Overview.md`

Feel free to distribute this Markdown to teammates so they can quickly understand the system’s components, data flow, and UI interaction.


## High‑level purpose
The project implements an end‑to‑end pipeline that:
1. **Generates synthetic system‑metric data** with realistic normal behavior and injectable failure scenarios.
2. **Trains an LSTM auto‑encoder** on normal data to learn a baseline reconstruction.  Anomalies are detected when reconstruction error exceeds a calibrated threshold.
3. **Runs a causal inference engine** on the metrics flagged as anomalous:
   * Pair‑wise Granger‑causality tests identify statistical cause‑effect relationships.
   * A directed acyclic graph (DAG) is built from those relationships and pruned to keep only the most plausible causal edges.
   * External deployment / config events are correlated with the first‑seen anomaly timestamps.
   * A composite scoring model (weighted mix of causal out‑flow, temporal priority, inflow, severity, event correlation, and PageRank) ranks the most likely root causes.
4. **Displays the whole workflow in an interactive Streamlit dashboard** where the user can:
   * Choose an anomaly scenario to inject.
   * Visualise raw metric traces and the AI‑generated anomaly scores.
   * See which metric triggered the detection and the ranked root‑cause list.

---
## Package layout
```
majorprojectt/
│   Dockerfile, .gitignore, progress.txt, docs/ (specs, UI_overview.md)
│
├─ src/                     # Core Python library used by the dashboard
│   ├─ __init__.py
│   ├─ data_ingestion/      # synthetic_generator.py (produces realistic time‑series)
│   ├─ models/              # lstm_autoencoder.py – LSTM‑based AutoEncoder + wrapper
│   ├─ anomaly_detection/   # anomaly_scorer, dimensionality_reduction, etc.
│   ├─ causal_inference/    # causal_engine.py – Granger, graph builder, event correlator, ranker, pipeline
│   ├─ reporting/           # report_generator.py, dashboard.py, anomaly_simulator.py (Streamlit UI)
│   └─ root_cause_ranking/  # scorer.py (simple scoring helper used by pipeline)
│
├─ rca-system/             # Legacy duplicate of the above modules (now removed)
│
└─ tests/                   # Unit / integration tests for the pipeline components
```

---
## Core modules (what runs where)

### `src/data_ingestion/synthetic_generator.py`
* Generates a pandas DataFrame with timestamps and a suite of metrics (CPU, memory, latency, error‑rate, DB connections, cache hit‑rate, request throughput, disk‑IO).
* Provides `inject_failure_scenario` that can embed known failure patterns (DB slow query, memory leak, CPU spike, etc.) and returns the ground‑truth metadata.

### `src/models/lstm_autoencoder.py`
* Implements `LSTMAutoencoder` (PyTorch) and a thin wrapper `AnomalyDetector`.
* Handles sliding‑window creation, model training on normal data, threshold calibration (99th percentile of validation error), and per‑metric anomaly scoring.

### `src/causal_inference/causal_engine.py`
* **GrangerAnalyzer** – pair‑wise statistical causality tests on anomalous metrics.
* **CausalGraphBuilder** – builds a directed graph from significant Granger pairs and prunes cycles using temporal precedence or weakest‑edge removal.
* **EventCorrelator** – matches the first‑seen anomaly timestamps with recent deployment / config events.
* **RootCauseRanker** – computes a composite score from causal out‑flow, temporal priority, inflow, severity, event correlation, and PageRank on the reversed graph.
* **CausalInferencePipeline** – orchestrates the whole flow; the dashboard calls this to obtain a ranked list of root‑cause candidates.

### `src/reporting/dashboard.py`
* A Streamlit app (`streamlit run reporting/dashboard.py`).
* Sidebar lets the user pick an anomaly pattern, adjust severity, and trigger generation + detection.
* Shows side‑by‑side plots:
  * Raw metric trace with a shaded “injected failure” region.
  * LSTM reconstruction error (anomaly score) with a threshold line.
* Displays a success / error banner and prints the top ranked root‑cause metrics.

### `src/reporting/anomaly_simulator.py`
* Similar to the dashboard but focused on the 27 predefined anomaly scenarios.
* Provides a convenient UI for quick stress‑testing of the LSTM detector.

### `src/root_cause_ranking/scorer.py`
* Helper functions used by the pipeline to compute simple scores (e.g., downstream effect count).

---
## How the pieces interact at runtime (data flow)
1. **Synthetic data** → `SyntheticMetricsGenerator.generate_normal_behavior()`.
2. **Inject failure** → `inject_failure_scenario()` returns a modified DataFrame and metadata.
3. **Train / load LSTM** → `AnomalyDetector.train()` on the normal portion; model saved as `best_autoencoder_model.pt`.
4. **Detect anomalies** → `AnomalyDetector.detect()` produces per‑metric scores and boolean flags (`*_score`, `*_is_anomaly`).
5. **Identify anomalous metrics** → metrics where `*_is_anomaly` is `True` are fed to the causal engine.
6. **Causal inference** → `CausalInferencePipeline.run()` returns:
   * `granger_results` (significant causality pairs)
   * `causal_graph` (NetworkX DAG)
   * `event_correlations` (list of matched deployment events)
   * `root_causes` (ranked list with composite scores, confidence labels, downstream effects, causal chain).
7. **Dashboard** visualises steps 2‑6 and shows the final ranked root‑cause table.

---
## Testing
* `rca-system/tests/test_synthetic_generator.py` – validates that the generator creates a DataFrame of the expected shape and that injected failures change the targeted metric.
* `rca-system/tests/test_integration.py` – runs a full end‑to‑end pipeline on a small synthetic dataset and asserts that a root‑cause is identified.
* Additional unit tests exist under `src` for model training and causal components.

---
## What was removed during cleanup
* The duplicated `rca-system/` package (old copy of the same modules) – all source code now lives under the top‑level `src/` directory.
* All `__pycache__` folders (Python byte‑code) – they are regenerated automatically.
* Generated artefacts in `outputs/` (previous reports and model checkpoints).
* Sample Git‑hook scripts under `.git/hooks/*.sample`.
* Temporary `progress.txt` and old Word/PDF documentation files.

---
## How to run the UI
```bash
# Install dependencies (see requirements.txt)
pip install -r rca-system/requirements.txt   # or install the top‑level requirements if provided

# Train the LSTM on normal synthetic data (one‑time step)
python -m src.models.lstm_autoencoder  # (script prints a short usage example)

# Launch the dashboard
cd src/reporting
streamlit run dashboard.py
```
The UI will be available at `http://localhost:8501`.  Use the sidebar to select an anomaly scenario, adjust severity, and click **Run** to see the detection and root‑cause ranking.

---
## Suggested next steps for contributors
* Add unit tests for the new `CausalInferencePipeline` wrapper (currently exercised only by the demo block).
* Consider persisting the trained LSTM model in a versioned artifact (e.g., a `models/` folder) and loading it from the dashboard instead of retraining each run.
* Refactor the Streamlit UI into reusable components (e.g., separate functions for metric plots, score plot, and root‑cause table) to make the codebase easier to maintain.
* Document the expected schema for external event logs that can be fed into `EventCorrelator`.

---
*Prepared on *2026‑03‑10* – you can share this markdown file with teammates to give them a full mental model of the system.*
