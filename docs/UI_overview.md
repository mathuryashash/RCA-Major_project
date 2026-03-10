# UI Overview and Application Logic

## High‑level purpose
This repository implements an **AI‑powered Root Cause Analysis (RCA) system** that:
1. **Generates synthetic telemetry data** (normal behavior) using `SyntheticMetricsGenerator`.
2. **Trains an unsupervised LSTM auto‑encoder** (`AnomalyDetector`) on that data.
3. **Injects a variety of simulated failure shapes** (spike, step, ramp, noise, etc.) into a short test window.
4. **Runs the trained model on the perturbed data** to produce anomaly scores.
5. **Runs causal‑inference pipelines** (Granger, graph building, ranking) to surface the most likely root causes.
6. **Displays everything in an interactive Streamlit UI**.

---

## Core Streamlit entry points
| File | Role |
|------|------|
| `src/reporting/dashboard.py` | Main interactive dashboard that ties together data generation, model training, anomaly injection, detection, and visualisation. |
| `src/reporting/anomaly_simulator.py` | A focused UI that only runs the synthetic‑data generation + LSTM detection pipeline for 27 pre‑defined anomaly scenarios. |

Both files share the same overall flow but differ in the amount of UI detail they expose.

---

## Detailed flow (dashboard)
1. **Page configuration** (`st.set_page_config`) – sets title, icon, wide layout.
2. **Sidebar controls** – user selects an anomaly pattern and adjusts a **Severity Factor** slider.
3. **Base data generation** (`generate_base_data`) – cached (`@st.cache_data`) to avoid recomputing the 10‑day normal dataset.
4. **Model training** (`train_detector`) – cached (`@st.cache_resource`) so the LSTM is trained once per session (default 4 epochs, batch‑size 32). Returns:
   * `detector` – the trained `AnomalyDetector` instance.
   * `scaler` – a `MinMaxScaler` fitted on the normal data.
   * `feat_cols` – list of metric column names.
5. **Test‑window selection** – the last two days of the base data (`test_df`). The anomaly injection starts at 60 % into this window.
6. **Anomaly injection** (`inject_shape`) – depending on the selected pattern, the appropriate metric column(s) are mutated with a mathematical shape (spike, step, ramp, noise, etc.) using the chosen severity.
7. **Clipping** – ensures percentages stay in `[0,100]` and latency stays reasonable.
8. **Scaling for inference** – the same scaler transforms the perturbed test data.
9. **Detection** (`detector.detect`) – the LSTM predicts reconstruction error for each timestep; the error becomes the **anomaly score**. Helper columns (`*_score`, `*_is_anomaly`) are added.
10. **Result filtering** – aligns timestamps between the original test frame and the detector output.
11. **Visualization** – for each highlighted metric:
    * **Left column**: raw metric line chart (Plotly) with a red vertical rectangle marking the injected failure window.
    * **Right column**: anomaly‑score line chart with a horizontal threshold line (default `1.0`).
    * Success / error messages (`st.success` / `st.error`) indicate whether the model flagged any timesteps as anomalous.
12. **User feedback** – the UI instantly reflects changes to the severity slider or anomaly selection.

---

## Detailed flow (anomaly_simulator)
The simulator follows the same steps as the dashboard but is stripped down to a single page:
* No causal‑inference ranking – it only shows the raw metric and the LSTM score.
* Uses the same sidebar for pattern selection and severity.
* Provides a compact, repeatable “stress‑test” harness for the auto‑encoder.

---

## Supporting modules (used by the UI)
| Module | Key classes / functions | Purpose |
|--------|------------------------|---------|
| `src/data_ingestion/synthetic_generator.py` | `SyntheticMetricsGenerator` | Generates multi‑metric time‑series with realistic daily/weekly seasonality and random noise. |
| `src/models/lstm_autoencoder.py` | `AnomalyDetector` (inherits `torch.nn.Module`) | Implements a window‑based LSTM encoder‑decoder that learns to reconstruct normal sequences; high reconstruction error ⇒ anomaly. |
| `src/causal_inference/causal_engine.py` | `GrangerAnalyzer`, `CausalGraphBuilder`, `EventCorrelator`, `RootCauseRanker`, `CausalInferencePipeline` | Takes the anomaly scores, runs Granger causality tests, builds a directed graph, correlates events, and ranks root causes. (Only referenced in the dashboard, not visualised directly.) |
| `src/reporting/report_generator.py` | `ReportGenerator` | Takes the final ranked root‑cause list and creates a JSON/markdown report that can be downloaded (future UI hook). |

---

## How the UI reflects the pipeline
1. **Sidebar → parameters** → drive the *synthetic* data generation & model training.
2. **Injection step** mimics a real incident (e.g., CPU spike). The UI visualises the ground‑truth window (red overlay).
3. **LSTM detection** produces a per‑timestep score; the UI shows this alongside the raw metric to let the user compare ground‑truth vs model output.
4. **Success message** is based on whether any `*_is_anomaly` flag is true (the model’s binary decision after thresholding).
5. **Future extensions** could call the causal pipeline and display a ranked list of suspects, or enable the download button to share a JSON report.

---

## Sharing this overview
The file you’re reading (`docs/UI_overview.md`) can be shared with teammates. It provides:
* A concise description of each component.
* The end‑to‑end data flow from synthetic generation to UI visualisation.
* Points of extension (causal inference, report download, theming, etc.).

Feel free to copy the markdown file or open it directly in the repository to discuss with your friends.
