# PySide6 Desktop Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Streamlit RCA dashboard into a native PySide6 (Qt 6) desktop application — native widgets for controls/tables, a `QWebEngineView` panel for Plotly graph/timeline visuals — while sharing 100% of the existing ML/causal-inference engine code unchanged.

**Architecture:** Extract the pipeline phase functions already living in `src/train_and_run.py` into an importable `src/pipeline/engine.py` module (no logic changes — pure move) so both the CLI and the new GUI call the same code, eliminating the duplicate pipeline logic currently copy-pasted into `dashboard.py`. Long-running phases (data gen, LSTM training, Granger/causal inference) run on `QThread` workers so the UI never blocks. Native `QWidget` controls handle all forms/tables/buttons; a single reusable `QWebEngineView` wrapper renders the existing Plotly figures (causal graph, anomaly timeline) as self-contained local HTML files — no network calls, fully air-gapped.

**Tech Stack:** PySide6 (Qt 6.7+), PySide6-WebEngine (bundled), existing stack unchanged — PyTorch (CPU), pandas, NetworkX, statsmodels, causal-learn, Plotly (used only for figure generation, rendered via QWebEngineView instead of `st.plotly_chart`). Packaging via PyInstaller. Tests via `pytest` + `pytest-qt`.

**Non-goals:** `dashboard.py` (Streamlit) is left untouched and keeps working standalone — this plan does not migrate or delete it. No React/Tauri, no rewrite of any ML/causal-inference/statistics code.

---

## File Structure

```
requirements.txt                              # MODIFY: append PySide6, pytest-qt, pyinstaller

src/
  pipeline/
    __init__.py                                # CREATE (empty)
    engine.py                                  # CREATE: phase functions moved from train_and_run.py
    visualizations.py                          # CREATE: draw_causal_graph/timeline/bar figure builders (moved from dashboard.py)
  train_and_run.py                             # MODIFY: import from pipeline.engine instead of defining phases inline
  desktop/
    __init__.py                                # CREATE (empty)
    state.py                                   # CREATE: AppState dataclass (replaces st.session_state)
    theme.py                                   # CREATE: dark QSS stylesheet ported from style.css palette
    workers.py                                 # CREATE: TrainWorker, InferenceWorker (QThread subclasses)
    main_window.py                             # CREATE: QMainWindow, tabs, menu, status bar
    main.py                                    # CREATE: QApplication entry point
    views/
      __init__.py                              # CREATE (empty)
      stage1_view.py                           # CREATE: Data Generation & Training tab
      stage2_view.py                           # CREATE: Incident Injection & RCA tab
      graph_panel.py                           # CREATE: reusable QWebEngineView Plotly host widget

packaging/
  rca_desktop.spec                             # CREATE: PyInstaller spec
  build.ps1                                    # CREATE: Windows build script

tests/
  __init__.py                                  # CREATE (empty)
  test_pipeline_engine.py                      # CREATE: unit tests for extracted engine
  test_desktop_smoke.py                        # CREATE: pytest-qt smoke tests
```

**Design boundary:** `pipeline/` has zero Qt/Streamlit imports — it is pure data/ML logic, importable by CLI, GUI, or tests alike. `desktop/` has zero direct ML imports beyond calling into `pipeline.engine` — it only knows about Qt widgets, threads, and wiring signals. `views/` files never call `torch`/`statsmodels`/etc. directly, only through `pipeline.engine` inside a worker thread.

---

## Task 1: Extract shared pipeline engine (no behavior change)

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/engine.py`
- Modify: `src/train_and_run.py`
- Test: `tests/test_pipeline_engine.py`

This is a pure extraction — the function bodies are moved verbatim from `src/train_and_run.py:88-517` into the new module so `train_and_run.py` (CLI) and the future desktop app both call the exact same code. No logic is rewritten.

- [ ] **Step 1: Create the package init**

Create `src/pipeline/__init__.py` (empty file).

- [ ] **Step 2: Create `src/pipeline/engine.py` with the moved phase functions**

```python
"""
Shared RCA pipeline engine — GUI-agnostic.

Every phase function here is a pure move from the original
src/train_and_run.py CLI script. Both the CLI entry point and the
PySide6 desktop app import from this module so there is exactly one
implementation of each pipeline phase.
"""

import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from models.lstm_autoencoder import AnomalyDetector
from anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
from causal_inference.dynamic_graph import DynamicGraphGenerator
from causal_inference.causal_engine import CausalInferencePipeline
from reporting.report_generator import ReportGenerator


def generate_data(
    seed: int,
    baseline_days: int = 30,
    failure_type: str = "database_slow_query",
    severity: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, List[str]]:
    """
    Generate normal baseline data + a failure scenario injected into the last
    ~17 hours of a 3-day window.

    Returns
    -------
    normal_df   : baseline_days-day clean baseline (for LSTM training)
    incident_df : 3-day window with injected failure (for inference)
    metadata    : ground-truth root cause info
    feat_cols   : metric column names (excludes 'timestamp')
    """
    gen = SyntheticMetricsGenerator(seed=seed)
    normal_df = gen.generate_normal_behavior(duration_days=baseline_days)

    gen2 = SyntheticMetricsGenerator(seed=seed + 1)
    incident_base = gen2.generate_normal_behavior(duration_days=3)
    failure_start = len(incident_base) - 200

    incident_df, metadata = gen2.inject_failure_scenario(
        incident_base,
        failure_type=failure_type,
        start_idx=failure_start,
        duration_samples=200,
        severity=severity,
    )

    feat_cols = [c for c in normal_df.columns if c != "timestamp"]
    return normal_df, incident_df, metadata, feat_cols


def preprocess(
    normal_df: pd.DataFrame,
    incident_df: pd.DataFrame,
    feat_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame, MinMaxScaler]:
    """
    Scale all metrics to [0, 1] using MinMaxScaler fitted on normal data.

    Returns
    -------
    normal_scaled   : np.ndarray  — ready for LSTM training
    incident_scaled : pd.DataFrame — scaled incident (preserves 'timestamp')
    scaler          : fitted scaler (for inverse-transform later if needed)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))

    normal_values = normal_df[feat_cols].values
    normal_scaled = scaler.fit_transform(normal_values)

    incident_clean = incident_df[feat_cols].ffill().bfill()
    incident_values = incident_clean.values

    incident_scaled_values = np.clip(
        scaler.transform(incident_values), 0.0, 1.0
    )

    incident_scaled = pd.DataFrame(incident_scaled_values, columns=feat_cols)
    incident_scaled.insert(0, "timestamp", incident_df["timestamp"].values)

    return normal_scaled, incident_scaled, scaler


def train_model(
    normal_scaled: np.ndarray,
    n_features: int,
    epochs: int,
    window_size: int,
    model_path: str,
    skip_train: bool,
) -> AnomalyDetector:
    """
    Train (or reload) the LSTM Autoencoder on normal data.

    The model is saved to `model_path` after training so that subsequent
    runs can use skip_train=True for faster iteration.
    """
    detector = AnomalyDetector(n_features=n_features, window_size=window_size)

    if skip_train and os.path.exists(model_path):
        import torch
        detector.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        windows = detector.create_windows(normal_scaled.astype(np.float32), stride=5)
        split = int(len(windows) * 0.8)
        val_data = windows[split:]
        detector._calibrate_thresholds(val_data)
    else:
        detector.train(
            normal_scaled.astype(np.float32),
            epochs=epochs,
            lr=1e-3,
            val_split=0.2,
            batch_size=32,
        )
        if os.path.exists("best_autoencoder_model.pt"):
            shutil.move("best_autoencoder_model.pt", model_path)

    return detector


def detect_anomalies(
    detector: AnomalyDetector,
    incident_scaled: pd.DataFrame,
    feat_cols: List[str],
    use_ensemble: bool = False,
    normal_scaled: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, pd.Timestamp], List[str]]:
    """
    Run the trained LSTM Autoencoder (or the Ensemble) on the incident window.

    Returns
    -------
    anomaly_scores   : {metric: max normalized reconstruction error}
    anomaly_times    : {metric: first timestamp exceeding threshold}
    active_anomalies : list of metrics that exceeded the threshold
    """
    if use_ensemble:
        ensemble = EnsembleAnomalyDetector(detector)
        if normal_scaled is not None:
            normal_df = pd.DataFrame(normal_scaled, columns=feat_cols)
            ensemble.fit_baselines(normal_df, feat_cols)
        result_df = ensemble.detect(incident_scaled, feat_cols)
    else:
        result_df = detector.detect(incident_scaled, feat_cols)

    anomaly_scores: Dict[str, float] = {}
    anomaly_times: Dict[str, pd.Timestamp] = {}
    active_anomalies: List[str] = []

    for col in feat_cols:
        score_col = f"{col}_score"
        flag_col = f"{col}_is_anomaly"
        if score_col not in result_df.columns:
            continue
        flagged = result_df[result_df[flag_col] == True]  # noqa: E712
        if not flagged.empty:
            active_anomalies.append(col)
            anomaly_scores[col] = float(result_df[score_col].max())
            first_idx = flagged.index[0]
            anomaly_times[col] = incident_scaled.loc[first_idx, "timestamp"]

    return anomaly_scores, anomaly_times, active_anomalies


def run_causal_inference(
    incident_scaled: pd.DataFrame,
    feat_cols: List[str],
    anomaly_scores: Dict[str, float],
    anomaly_times: Dict[str, pd.Timestamp],
    active_anomalies: List[str],
    failure_start_time: pd.Timestamp,
    max_lag: int = 5,
    use_dynamic_topology: bool = False,
) -> Dict:
    """
    Run the full Granger causality analysis and build the directed causal graph.
    Also creates a synthetic deployment event at T-20min before the failure.
    """
    df_for_granger = incident_scaled.set_index("timestamp")[active_anomalies]

    events_df = pd.DataFrame([{
        "timestamp": failure_start_time - pd.Timedelta(minutes=20),
        "description": "Code deployment or config change preceding incident",
        "type": "deployment",
    }])

    pipeline = CausalInferencePipeline(max_lag=max_lag, significance_level=0.05)
    results = pipeline.run(
        df=df_for_granger,
        anomalous_metrics=active_anomalies,
        anomaly_scores=anomaly_scores,
        anomaly_first_seen=anomaly_times,
        events_df=events_df,
    )

    causal_graph = results["causal_graph"]

    if use_dynamic_topology:
        dyn_gen = DynamicGraphGenerator()
        refined_graph = dyn_gen.refine_causal_graph(causal_graph)
        results["causal_graph"] = refined_graph

    return results


def rank_root_causes(results: Dict) -> List[Dict]:
    """Return the ranked root cause candidates (already sorted by the ranker)."""
    return results.get("root_causes", [])


def generate_reports(
    results: Dict,
    root_causes: List[Dict],
    anomaly_times: Dict[str, pd.Timestamp],
    metadata: Dict,
    failure_type: str,
    output_dir: str,
    incident_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate Markdown and JSON incident reports and save to output_dir.

    Returns
    -------
    {"incident_id": str, "md_path": str, "json_path": str,
     "md_report": str, "json_report": dict}
    """
    import json as json_mod
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    if incident_id is None:
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    report_gen = ReportGenerator()

    ranked_tuples = []
    for rc in root_causes:
        explanation = {
            "out_edges": rc.get("downstream_effects", []),
            "components": rc.get("scores_breakdown", {}),
            "pagerank": rc.get("pagerank", 0.0),
        }
        ranked_tuples.append((rc["metric"], rc["composite_score"], explanation))

    md_report = report_gen.generate_report(
        incident_id=incident_id,
        ranked_candidates=ranked_tuples,
        causal_graph=results["causal_graph"],
        anomaly_times=anomaly_times,
    )

    md_path = os.path.join(output_dir, f"{incident_id}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    causal_graph = results["causal_graph"]
    edges_serializable = [
        {
            "cause": u,
            "effect": v,
            "strength": round(float(d.get("strength", 0.0)), 4),
            "lag": d.get("lag"),
            "p_value": round(float(d.get("p_value", 1.0)), 6),
        }
        for u, v, d in causal_graph.edges(data=True)
    ]

    json_report = {
        "incident_id": incident_id,
        "timestamp": datetime.now().isoformat() + "Z",
        "failure_type": failure_type,
        "ground_truth": {
            "root_cause": metadata.get("root_cause"),
            "causal_chain": metadata.get("causal_chain"),
        },
        "root_causes": [
            {
                "rank": rc["rank"],
                "metric": rc["metric"],
                "composite_score": rc["composite_score"],
                "confidence": rc["confidence"],
                "scores_breakdown": rc.get("scores_breakdown", {}),
                "downstream_effects": rc.get("downstream_effects", []),
                "causal_chain": rc.get("causal_chain", []),
            }
            for rc in root_causes
        ],
        "causal_graph": {
            "nodes": list(causal_graph.nodes),
            "edges": edges_serializable,
        },
        "event_correlations": results.get("event_correlations", []),
        "anomaly_detection_times": {k: str(v) for k, v in anomaly_times.items()},
    }

    json_path = os.path.join(output_dir, f"{incident_id}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json_mod.dump(json_report, f, indent=2, default=str)

    return {
        "incident_id": incident_id,
        "md_path": md_path,
        "json_path": json_path,
        "md_report": md_report,
        "json_report": json_report,
    }
```

- [ ] **Step 3: Rewrite `src/train_and_run.py` to import from the new engine**

Replace lines 37-66 (imports) — keep only what the CLI itself still needs — and delete the now-duplicated phase function definitions (former lines 88-517: `generate_data`, `preprocess`, `train_model`, `detect_anomalies`, `run_causal_inference`, `rank_root_causes`, `generate_reports`, plus the `banner`/`step` helpers that were only used for their console prints — those prints move into the CLI's `main()` around each call instead).

```python
"""
End-to-End Training Pipeline and RCA Runner
============================================
AI-Powered Root Cause Analysis System — PRD §1.1 through §1.1.6

CLI entry-point. All pipeline phases live in pipeline.engine (shared with
the PySide6 desktop app) — this file only parses args and prints progress.

Usage
-----
    python src/train_and_run.py
    python src/train_and_run.py --failure memory_leak --severity 0.9
    python src/train_and_run.py --skip-train --failure cpu_spike

CLI Flags
---------
    --failure     one of: database_slow_query | memory_leak | cpu_spike  (default: database_slow_query)
    --severity    float 0.1–1.0  (default: 0.8)
    --epochs      LSTM training epochs  (default: 15)
    --skip-train  skip training; load saved model weights instead
    --output-dir  directory for report artefacts  (default: ./outputs)
    --seed        RNG seed for reproducibility  (default: 42)
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from pipeline import engine


def banner(text: str) -> None:
    line = "-" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def step(num: int, label: str) -> None:
    print(f"\n[Step {num}] {label} ...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCA System — End-to-End Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--failure", default="database_slow_query",
                         choices=["database_slow_query", "memory_leak", "cpu_spike"],
                         help="Failure scenario to inject (default: database_slow_query)")
    parser.add_argument("--severity", type=float, default=0.8,
                         help="Failure severity 0.1–1.0 (default: 0.8)")
    parser.add_argument("--epochs", type=int, default=15,
                         help="LSTM training epochs (default: 15)")
    parser.add_argument("--skip-train", action="store_true",
                         help="Skip training; load saved weights instead")
    parser.add_argument("--output-dir", default="outputs",
                         help="Directory for report artefacts (default: ./outputs)")
    parser.add_argument("--use-ensemble", action="store_true",
                         help="Use Ensemble Anomaly Detection instead of bare LSTM")
    parser.add_argument("--use-dynamic-topology", action="store_true",
                         help="Refine causal graph using real-time Jaeger topology")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--window-size", type=int, default=12,
                         help="LSTM sliding window size in time-steps (default: 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    banner("AI-Powered Root Cause Analysis System - Full Pipeline")
    print(f"  Failure type : {args.failure}  (severity={args.severity})")
    print(f"  LSTM epochs  : {args.epochs}  | Window size: {args.window_size}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  RNG seed     : {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "lstm_autoencoder_best.pt")
    t_pipeline_start = time.time()

    step(1, "Synthetic Data Generation")
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=args.seed, failure_type=args.failure, severity=args.severity,
    )
    print(f"  Ground truth: {metadata['root_cause']}")

    step(2, "Preprocessing — MinMax normalization [0, 1]")
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )

    step(3, "LSTM Autoencoder Training")
    detector = engine.train_model(
        normal_scaled=normal_scaled, n_features=len(feat_cols),
        epochs=args.epochs, window_size=args.window_size,
        model_path=model_path, skip_train=args.skip_train,
    )

    step(4, "Anomaly Detection")
    anomaly_scores, anomaly_times, active_anomalies = engine.detect_anomalies(
        detector, incident_scaled, feat_cols,
        use_ensemble=args.use_ensemble,
        normal_scaled=normal_scaled if args.use_ensemble else None,
    )
    print(f"  Anomalous metrics detected: {len(active_anomalies)}")

    gen_tmp = SyntheticMetricsGenerator(seed=args.seed + 1)
    incident_base_len = len(gen_tmp.generate_normal_behavior(duration_days=3))
    failure_start_idx = incident_base_len - 200
    failure_start_time = incident_df.iloc[failure_start_idx]["timestamp"]

    if len(active_anomalies) < 2:
        print("\n!  LSTM did not naturally flag the synthetic failure. Forcing anomalies for pipeline test.")
        active_anomalies = feat_cols[:3]
        for k in active_anomalies:
            anomaly_scores[k] = 1.0
            anomaly_times[k] = failure_start_time + pd.Timedelta(minutes=5)

    step(5, "Causal Inference - Granger Causality & Graph Construction")
    causal_results = engine.run_causal_inference(
        incident_scaled=incident_scaled, feat_cols=feat_cols,
        anomaly_scores=anomaly_scores, anomaly_times=anomaly_times,
        active_anomalies=active_anomalies, failure_start_time=failure_start_time,
        use_dynamic_topology=args.use_dynamic_topology,
    )
    causal_graph = causal_results["causal_graph"]
    print(f"  Causal graph: {len(causal_graph.nodes)} nodes, {len(causal_graph.edges)} edges")

    step(6, "Root Cause Ranking")
    root_causes = engine.rank_root_causes(causal_results)
    for rc in root_causes[:5]:
        print(f"  #{rc['rank']:<4} {rc['metric']:<35} "
              f"{rc['composite_score']:>6.3f}  {rc['confidence']}")

    step(7, "Report Generation")
    report_paths = engine.generate_reports(
        results=causal_results, root_causes=root_causes,
        anomaly_times=anomaly_times, metadata=metadata,
        failure_type=args.failure, output_dir=args.output_dir,
    )
    print(f"  Markdown report  -> {report_paths['md_path']}")
    print(f"  JSON report      -> {report_paths['json_path']}")

    elapsed = time.time() - t_pipeline_start
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print("\nDone. OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the extraction regression test**

```python
# tests/test_pipeline_engine.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import engine


def test_generate_data_returns_expected_shapes():
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=1, baseline_days=1, failure_type="cpu_spike", severity=0.9,
    )
    assert "timestamp" not in feat_cols
    assert len(feat_cols) == 10
    assert len(normal_df) > 0
    assert len(incident_df) > 0
    assert metadata["root_cause"] == "cpu_usage_percent" or "cpu" in metadata["root_cause"]


def test_preprocess_scales_to_unit_range():
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=1, baseline_days=1, failure_type="memory_leak", severity=0.8,
    )
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )
    assert normal_scaled.min() >= 0.0 - 1e-6
    assert normal_scaled.max() <= 1.0 + 1e-6
    assert incident_scaled[feat_cols].min().min() >= 0.0 - 1e-6
    assert incident_scaled[feat_cols].max().max() <= 1.0 + 1e-6
    assert list(incident_scaled.columns)[0] == "timestamp"


def test_train_and_detect_roundtrip(tmp_path):
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=2, baseline_days=2, failure_type="database_slow_query", severity=1.0,
    )
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )
    model_path = str(tmp_path / "test_model.pt")
    detector = engine.train_model(
        normal_scaled=normal_scaled, n_features=len(feat_cols),
        epochs=1, window_size=6, model_path=model_path, skip_train=False,
    )
    anomaly_scores, anomaly_times, active = engine.detect_anomalies(
        detector, incident_scaled, feat_cols,
    )
    assert isinstance(anomaly_scores, dict)
    assert isinstance(active, list)
```

- [ ] **Step 5: Run the tests**

Run: `cd D:\vscode\majorprojectt && python -m pytest tests/test_pipeline_engine.py -v`
Expected: `3 passed` (takes ~15-30s due to 1-2 epoch LSTM training).

- [ ] **Step 6: Verify the CLI still works end-to-end**

Run: `cd D:\vscode\majorprojectt\src && python train_and_run.py --epochs 1 --output-dir ../outputs_test`
Expected: Completes through `Done. OK`, prints a Top-5 root cause table, writes `outputs_test/*_report.md` and `*_report.json`.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline src/train_and_run.py tests/test_pipeline_engine.py
git commit -m "refactor: extract shared pipeline engine from train_and_run CLI"
```

---

## Task 2: Move visualization builders into a GUI-agnostic module

**Files:**
- Create: `src/pipeline/visualizations.py`
- Test: manual (verified visually in Task 6)

`dashboard.py` keeps its own inline `draw_causal_graph` (untouched, still works standalone). This task copies the pure-Plotly logic — no Streamlit dependency — into a shared module the desktop app imports, plus two more figure builders that Stage 2 currently builds inline in `dashboard.py:1007-1057`.

- [ ] **Step 1: Create `src/pipeline/visualizations.py`**

```python
"""
Plotly figure builders — no Streamlit or Qt imports. Pure functions:
graph/data in, plotly.graph_objects.Figure out. Shared by dashboard.py
(if it chooses to import them) and the PySide6 desktop app.
"""

from typing import Dict, List

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def draw_causal_graph(G: nx.DiGraph, root_cause_metric: str) -> go.Figure:
    """Return a Plotly figure for the causal graph with arrows and legend."""
    if len(G.nodes) == 0:
        return go.Figure().update_layout(title="No causal edges identified")

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


def build_score_bar_figure(anomaly_scores: Dict[str, float]) -> go.Figure:
    """Bar chart of max anomaly score per flagged metric."""
    score_df = pd.DataFrame.from_dict(
        anomaly_scores, orient="index", columns=["Max Score"]
    ).sort_values("Max Score", ascending=False)

    fig = px.bar(score_df, y="Max Score", title="Max Anomaly Score per Metric",
                color="Max Score", color_continuous_scale="reds")
    fig.update_layout(
        paper_bgcolor="#151a2e", plot_bgcolor="#151a2e", font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(tickangle=-45),
    )
    return fig
```

- [ ] **Step 2: Sanity-check it imports and runs standalone**

Run:
```
cd D:\vscode\majorprojectt\src && python -c "
from pipeline.visualizations import draw_causal_graph
import networkx as nx
G = nx.DiGraph(); G.add_edge('a', 'b', lag=1, strength=0.8, p_value=0.01)
fig = draw_causal_graph(G, 'a')
print('nodes in figure traces:', len(fig.data))
"
```
Expected: `nodes in figure traces: <some number > 0>` with no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/visualizations.py
git commit -m "feat: add GUI-agnostic Plotly figure builders for desktop reuse"
```

---

## Task 3: Install desktop dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append desktop-app dependencies**

Add to the end of `requirements.txt`:

```
# Desktop app (PySide6 conversion)
PySide6>=6.7.0
pytest-qt>=4.4.0
pyinstaller>=6.10.0
```

- [ ] **Step 2: Install and verify**

Run: `cd D:\vscode\majorprojectt && pip install -r requirements.txt`

Then verify WebEngine is present (it's a separate wheel pulled in automatically by the `PySide6` metapackage on Windows/macOS/Linux x86_64):

Run:
```
python -c "from PySide6.QtWidgets import QApplication; from PySide6.QtWebEngineWidgets import QWebEngineView; print('PySide6 + WebEngine OK')"
```
Expected: `PySide6 + WebEngine OK`. If this raises `ModuleNotFoundError: PySide6.QtWebEngineWidgets`, run `pip install PySide6-Addons` explicitly (older PySide6 releases split it out) and re-check.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add PySide6, pytest-qt, pyinstaller for desktop app"
```

---

## Task 4: App shell — state, theme, main window, entry point

**Files:**
- Create: `src/desktop/__init__.py`
- Create: `src/desktop/state.py`
- Create: `src/desktop/theme.py`
- Create: `src/desktop/main_window.py`
- Create: `src/desktop/main.py`
- Create: `src/desktop/views/__init__.py`
- Test: `tests/test_desktop_smoke.py` (started here, extended in later tasks)

- [ ] **Step 1: Create package inits**

Create `src/desktop/__init__.py` and `src/desktop/views/__init__.py` (both empty).

- [ ] **Step 2: Create `src/desktop/state.py`**

Replaces Streamlit's `st.session_state` — a plain object the two views share via the main window.

```python
"""Shared application state — replaces st.session_state from the Streamlit app."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class AppState:
    model_trained: bool = False

    normal_df: Optional[pd.DataFrame] = None
    feat_cols: List[str] = field(default_factory=list)
    scaler: Optional[MinMaxScaler] = None
    detector: Any = None  # models.lstm_autoencoder.AnomalyDetector

    baseline_days: int = 30
    training_epochs: int = 5
    window_size: int = 12
    seed: int = 42

    last_causal_results: Optional[Dict] = None
    last_root_causes: Optional[List[Dict]] = None
    last_incident_scaled: Optional[pd.DataFrame] = None
    last_anomaly_scores: Optional[Dict[str, float]] = None
    last_anomaly_times: Optional[Dict[str, pd.Timestamp]] = None
    last_metadata: Optional[Dict] = None
    last_report: Optional[Dict[str, str]] = None
```

- [ ] **Step 3: Create `src/desktop/theme.py`**

Dark QSS ported from `src/reporting/style.css`'s custom-property palette (same hex values: `--primary #667eea`, `--primary-alt #764ba2`, `--accent #f093fb`, `--bg-dark #0f1628`, `--bg-mid #1a1f3a`, `--text-bright #e2e8f0`, `--success #48bb78`, `--danger #ff4757`).

```python
"""Dark theme QSS, palette-matched to the Streamlit dashboard's style.css."""

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0f1628;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10.5pt;
}

QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: #151a2e;
    border-radius: 8px;
}

QTabBar::tab {
    background: #1a1f3a;
    color: #a0aec0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:selected {
    background: rgba(102, 126, 234, 0.18);
    color: #e2e8f0;
    border-bottom: 2px solid #667eea;
}

QGroupBox {
    background-color: rgba(30, 33, 48, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-top: 1.2em;
    padding: 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #e2e8f0;
}

QPushButton {
    background: #667eea;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}

QPushButton:hover { background: #7688ee; }
QPushButton:pressed { background: #5568d3; }
QPushButton:disabled { background: #2a2f47; color: #6b7280; }

QPushButton#primaryAction {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    padding: 10px 28px;
    font-size: 11pt;
}

QSlider::groove:horizontal {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #667eea;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QProgressBar {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    text-align: center;
    color: #e2e8f0;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #f093fb);
    border-radius: 6px;
}

QTableWidget {
    background-color: #151a2e;
    gridline-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
}

QHeaderView::section {
    background-color: #1a1f3a;
    color: #a0aec0;
    padding: 6px;
    border: none;
    font-weight: 600;
}

QPlainTextEdit {
    background-color: #0b0f1c;
    color: #7bed9f;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    font-family: Consolas, monospace;
}

QStatusBar {
    background: #1a1f3a;
    color: #a0aec0;
}

QLabel#heroTitle {
    font-size: 20pt;
    font-weight: 800;
    color: #e2e8f0;
    padding: 6px 0;
}

QLabel#heroSubtitle {
    color: #a0aec0;
    font-size: 10pt;
    padding-bottom: 8px;
}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(DARK_QSS)
```

- [ ] **Step 4: Create `src/desktop/main_window.py`**

```python
"""Main window — tab shell wiring Stage 1 and Stage 2 views together."""

from PySide6.QtWidgets import QMainWindow, QTabWidget, QLabel, QVBoxLayout, QWidget

from desktop.state import AppState
from desktop.views.stage1_view import Stage1View
from desktop.views.stage2_view import Stage2View


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Root Cause Analysis")
        self.resize(1400, 900)

        self.state = AppState()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel("🔍 AI-Powered Root Cause Analysis")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "Diagnose production failures using LSTM Autoencoders, "
            "Granger Causality, and Multi-factor Root Cause Scoring"
        )
        subtitle.setObjectName("heroSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.stage1 = Stage1View(self.state)
        self.stage2 = Stage2View(self.state)
        self.tabs.addTab(self.stage1, "1 — Data Generation && Training")
        self.tabs.addTab(self.stage2, "2 — Run RCA Inference")
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.stage1.model_trained.connect(self._on_model_trained)

    def _on_model_trained(self):
        self.state.model_trained = True
        self.stage2.set_enabled(True)
        self.statusBar().showMessage("Model trained — Stage 2 unlocked", 5000)
```

- [ ] **Step 5: Create `src/desktop/main.py`**

```python
"""Desktop app entry point."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

from PySide6.QtWidgets import QApplication

from desktop.theme import apply_theme
from desktop.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RCA Desktop")
    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

This references `Stage1View`/`Stage2View` which don't exist yet — that's expected, Task 5 and Task 6 create them. Skip running `main.py` until then.

- [ ] **Step 6: Commit**

```bash
git add src/desktop/__init__.py src/desktop/state.py src/desktop/theme.py src/desktop/main_window.py src/desktop/main.py src/desktop/views/__init__.py
git commit -m "feat: add PySide6 app shell (state, theme, main window, entry point)"
```

---

## Task 5: Background workers (QThread)

**Files:**
- Create: `src/desktop/workers.py`

Both workers call straight into `pipeline.engine` — no ML logic here, only signal wiring so the UI thread never blocks.

- [ ] **Step 1: Create `src/desktop/workers.py`**

```python
"""QThread workers wrapping pipeline.engine calls so the UI never blocks."""

from PySide6.QtCore import QThread, Signal

from pipeline import engine


class TrainWorker(QThread):
    """Stage 1: generate baseline data + train the LSTM Autoencoder."""

    progress = Signal(int, str)
    finished_ok = Signal(object)  # (normal_df, feat_cols, scaler, detector, elapsed_seconds)
    failed = Signal(str)

    def __init__(self, baseline_days: int, epochs: int, window_size: int, seed: int, parent=None):
        super().__init__(parent)
        self.baseline_days = baseline_days
        self.epochs = epochs
        self.window_size = window_size
        self.seed = seed

    def run(self):
        try:
            import time
            self.progress.emit(10, "Generating baseline data …")
            normal_df, _incident_df, _meta, feat_cols = engine.generate_data(
                seed=self.seed, baseline_days=self.baseline_days,
            )

            self.progress.emit(40, "Preprocessing …")
            normal_scaled, _incident_scaled, scaler = engine.preprocess(
                normal_df, normal_df, feat_cols
            )

            self.progress.emit(55, f"Training LSTM ({self.epochs} epoch(s)) …")
            t0 = time.time()
            detector = engine.train_model(
                normal_scaled=normal_scaled, n_features=len(feat_cols),
                epochs=self.epochs, window_size=self.window_size,
                model_path="outputs/lstm_autoencoder_best.pt", skip_train=False,
            )
            elapsed = time.time() - t0

            self.progress.emit(100, "Model trained")
            self.finished_ok.emit((normal_df, feat_cols, scaler, detector, elapsed))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Stage 2: inject a failure scenario and run the full RCA pipeline."""

    progress = Signal(int, str)
    finished_ok = Signal(object)  # dict payload, see run()
    failed = Signal(str)

    def __init__(self, normal_df, feat_cols, detector, failure_type: str,
                 severity: float, max_granger_lag: int, seed: int, parent=None):
        super().__init__(parent)
        self.normal_df = normal_df
        self.feat_cols = feat_cols
        self.detector = detector
        self.failure_type = failure_type
        self.severity = severity
        self.max_granger_lag = max_granger_lag
        self.seed = seed

    def run(self):
        try:
            import pandas as pd

            self.progress.emit(10, "Generating incident data …")
            _normal_df, incident_df, metadata, _feat_cols = engine.generate_data(
                seed=self.seed, failure_type=self.failure_type, severity=self.severity,
            )

            self.progress.emit(25, "Preprocessing …")
            _normal_scaled, incident_scaled, _scaler = engine.preprocess(
                self.normal_df, incident_df, self.feat_cols
            )

            self.progress.emit(45, "Detecting anomalies …")
            anomaly_scores, anomaly_times, active_anomalies = engine.detect_anomalies(
                self.detector, incident_scaled, self.feat_cols,
            )

            if len(active_anomalies) == 0:
                self.failed.emit(
                    "No anomalies detected. Try increasing severity or training epochs."
                )
                return

            failure_start_idx = len(incident_df) - 200
            failure_start_time = pd.Timestamp(incident_df.iloc[failure_start_idx]["timestamp"])

            self.progress.emit(70, "Running Granger causality & ranking root causes …")
            causal_results = engine.run_causal_inference(
                incident_scaled=incident_scaled, feat_cols=self.feat_cols,
                anomaly_scores=anomaly_scores, anomaly_times=anomaly_times,
                active_anomalies=active_anomalies, failure_start_time=failure_start_time,
                max_lag=self.max_granger_lag,
            )
            root_causes = engine.rank_root_causes(causal_results)

            self.progress.emit(90, "Generating reports …")
            report = engine.generate_reports(
                results=causal_results, root_causes=root_causes,
                anomaly_times=anomaly_times, metadata=metadata,
                failure_type=self.failure_type, output_dir="outputs",
            )

            self.progress.emit(100, "Pipeline complete")
            self.finished_ok.emit({
                "causal_results": causal_results,
                "root_causes": root_causes,
                "incident_scaled": incident_scaled,
                "anomaly_scores": anomaly_scores,
                "anomaly_times": anomaly_times,
                "metadata": metadata,
                "report": report,
            })
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
```

- [ ] **Step 2: Commit**

```bash
git add src/desktop/workers.py
git commit -m "feat: add QThread workers for training and inference"
```

---

## Task 6: Stage 1 view — Data Generation & Training tab

**Files:**
- Create: `src/desktop/views/stage1_view.py`

Native Qt controls (`QSlider` + `QSpinBox` pairs, matching the Streamlit sliders), a `QPushButton` to launch `TrainWorker`, a `QProgressBar`, and a `QPlainTextEdit` log console.

- [ ] **Step 1: Create `src/desktop/views/stage1_view.py`**

```python
"""Stage 1 tab: generate synthetic baseline data and train the LSTM Autoencoder."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSlider, QSpinBox, QPushButton, QProgressBar, QPlainTextEdit, QLabel,
)
from PySide6.QtCore import Qt

from desktop.workers import TrainWorker


def _slider_with_spinbox(minimum, maximum, default, parent_layout, label):
    row = QHBoxLayout()
    slider = QSlider(Qt.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(default)
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(default)
    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    row.addWidget(slider, stretch=3)
    row.addWidget(spin, stretch=1)
    parent_layout.addRow(label, row)
    return spin


class Stage1View(QWidget):
    model_trained = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None

        layout = QVBoxLayout(self)

        params_box = QGroupBox("Training Parameters")
        form = QFormLayout()
        self.baseline_days_spin = _slider_with_spinbox(10, 60, 30, form, "Baseline Training Days")
        self.epochs_spin = _slider_with_spinbox(1, 30, 5, form, "LSTM Training Epochs")
        self.window_size_spin = _slider_with_spinbox(6, 60, 12, form, "LSTM Window Size (timesteps)")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        form.addRow("Random Seed", self.seed_spin)
        params_box.setLayout(form)
        layout.addWidget(params_box)

        self.train_button = QPushButton("Generate Data && Train Model")
        self.train_button.setObjectName("primaryAction")
        self.train_button.clicked.connect(self._on_train_clicked)
        layout.addWidget(self.train_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        log_box.setLayout(log_layout)
        layout.addWidget(log_box, stretch=1)

    def _on_train_clicked(self):
        self.train_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_console.appendPlainText("Starting Stage 1 pipeline …")

        self.state.baseline_days = self.baseline_days_spin.value()
        self.state.training_epochs = self.epochs_spin.value()
        self.state.window_size = self.window_size_spin.value()
        self.state.seed = self.seed_spin.value()

        self.worker = TrainWorker(
            baseline_days=self.state.baseline_days,
            epochs=self.state.training_epochs,
            window_size=self.state.window_size,
            seed=self.state.seed,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct: int, message: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)
        self.log_console.appendPlainText(f"[{pct:3d}%] {message}")

    def _on_finished(self, payload):
        normal_df, feat_cols, scaler, detector, elapsed = payload
        self.state.normal_df = normal_df
        self.state.feat_cols = feat_cols
        self.state.scaler = scaler
        self.state.detector = detector
        self.state.model_trained = True

        self.log_console.appendPlainText(
            f"Model trained in {elapsed:.1f}s | {len(normal_df):,} samples | "
            f"{len(feat_cols)} features"
        )
        self.train_button.setEnabled(True)
        self.model_trained.emit()

    def _on_failed(self, message: str):
        self.log_console.appendPlainText(f"ERROR: {message}")
        self.status_label.setText(f"Failed: {message}")
        self.train_button.setEnabled(True)
```

- [ ] **Step 2: Commit**

```bash
git add src/desktop/views/stage1_view.py
git commit -m "feat: add Stage 1 (data generation & training) view"
```

---

## Task 7: `QWebEngineView` Plotly host widget

**Files:**
- Create: `src/desktop/views/graph_panel.py`

**Important:** `QWebEngineView.setHtml()` has a documented ~2MB content size limit in Qt. A fully self-contained Plotly HTML export (`include_plotlyjs=True`) embeds the ~4.5MB minified plotly.js and will silently fail to render if passed to `setHtml()`. This widget avoids that by writing the HTML to a temp file and loading it via `setUrl(QUrl.fromLocalFile(...))` instead — the correct approach for large local content, and it keeps the app fully air-gapped (no CDN fetch for plotly.js).

- [ ] **Step 1: Create `src/desktop/views/graph_panel.py`**

```python
"""Reusable widget that renders a Plotly figure inside a QWebEngineView.

Writes each figure to a temp HTML file (with plotly.js embedded inline —
no network access needed) and loads it via a file:// URL, since
QWebEngineView.setHtml() silently truncates content over ~2MB and a
fully self-contained Plotly export is larger than that.
"""

import os
import tempfile

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False


class PlotlyWebView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tmp_dir = tempfile.mkdtemp(prefix="rca_desktop_")
        self._file_counter = 0

        if _WEBENGINE_AVAILABLE:
            self.view = QWebEngineView()
            layout.addWidget(self.view)
        else:
            from PySide6.QtWidgets import QLabel
            self.view = QLabel(
                "QtWebEngine is not installed — graph view unavailable.\n"
                "Run: pip install PySide6-Addons"
            )
            layout.addWidget(self.view)

    def show_figure(self, fig) -> None:
        if not _WEBENGINE_AVAILABLE:
            return
        self._file_counter += 1
        html_path = os.path.join(self._tmp_dir, f"figure_{self._file_counter}.html")
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        self.view.setUrl(QUrl.fromLocalFile(html_path))
```

- [ ] **Step 2: Commit**

```bash
git add src/desktop/views/graph_panel.py
git commit -m "feat: add QWebEngineView Plotly host widget"
```

---

## Task 8: Stage 2 view — Incident Injection & RCA tab

**Files:**
- Create: `src/desktop/views/stage2_view.py`

Native controls for scenario/severity/lag, results in a `QTableWidget`, causal graph + timeline in two `PlotlyWebView` sub-tabs, and export buttons using `QFileDialog`.

- [ ] **Step 1: Create `src/desktop/views/stage2_view.py`**

```python
"""Stage 2 tab: inject a failure scenario and run the full RCA pipeline."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QSlider, QSpinBox, QPushButton, QProgressBar, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog, QPlainTextEdit,
)
from PySide6.QtCore import Qt

from desktop.workers import InferenceWorker
from desktop.views.graph_panel import PlotlyWebView
from pipeline.visualizations import draw_causal_graph, build_timeline_figure, build_score_bar_figure

SCENARIO_DESCRIPTIONS = {
    "database_slow_query": "Simulates slow DB queries causing cascading latency and throughput drops",
    "memory_leak": "Gradual memory exhaustion leading to OOM errors and degraded performance",
    "cpu_spike": "CPU saturation from a runaway process, causing request queuing and timeouts",
    "network_partition": "Network failures causing error rate spikes and intermittent connectivity",
    "thread_pool_exhaustion": "Thread pool starvation reducing throughput and increasing response times",
    "disk_io_spike": "Disk I/O bottleneck from excessive logging, impacting DB connections",
}


class Stage2View(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        self._last_payload = None

        layout = QVBoxLayout(self)

        self.locked_label = QLabel("Train a model in Stage 1 first.")
        layout.addWidget(self.locked_label)

        config_box = QGroupBox("Failure Injection")
        form = QFormLayout()

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIO_DESCRIPTIONS.keys()))
        self.scenario_combo.currentTextChanged.connect(self._on_scenario_changed)
        form.addRow("Failure Scenario", self.scenario_combo)

        self.scenario_desc_label = QLabel(SCENARIO_DESCRIPTIONS[self.scenario_combo.currentText()])
        self.scenario_desc_label.setWordWrap(True)
        form.addRow("", self.scenario_desc_label)

        sev_row = QHBoxLayout()
        self.severity_slider = QSlider(Qt.Horizontal)
        self.severity_slider.setRange(1, 10)
        self.severity_slider.setValue(8)
        self.severity_value_label = QLabel("0.8")
        self.severity_slider.valueChanged.connect(
            lambda v: self.severity_value_label.setText(f"{v / 10:.1f}")
        )
        sev_row.addWidget(self.severity_slider, stretch=3)
        sev_row.addWidget(self.severity_value_label, stretch=1)
        form.addRow("Severity (0.1 – 1.0)", sev_row)

        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(2, 10)
        self.lag_spin.setValue(5)
        form.addRow("Granger Max Lag", self.lag_spin)

        config_box.setLayout(form)
        layout.addWidget(config_box)

        self.run_button = QPushButton("Simulate Incident && Run Full RCA")
        self.run_button.setObjectName("primaryAction")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_tabs = QTabWidget()

        self.root_cause_table = QTableWidget()
        self.root_cause_table.setColumnCount(6)
        self.root_cause_table.setHorizontalHeaderLabels(
            ["Rank", "Metric", "Composite Score", "Confidence", "Causal Outflow", "Downstream"]
        )
        self.results_tabs.addTab(self.root_cause_table, "Root Causes")

        self.graph_view = PlotlyWebView()
        self.results_tabs.addTab(self.graph_view, "Causal Graph")

        self.timeline_view = PlotlyWebView()
        self.results_tabs.addTab(self.timeline_view, "Anomaly Timeline")

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.results_tabs.addTab(self.report_text, "Markdown Report")

        layout.addWidget(self.results_tabs, stretch=1)

        export_row = QHBoxLayout()
        self.export_md_button = QPushButton("Export Markdown Report")
        self.export_json_button = QPushButton("Export JSON Report")
        self.export_md_button.clicked.connect(self._export_md)
        self.export_json_button.clicked.connect(self._export_json)
        export_row.addWidget(self.export_md_button)
        export_row.addWidget(self.export_json_button)
        layout.addLayout(export_row)

        self.set_enabled(False)

    def set_enabled(self, enabled: bool):
        self.locked_label.setVisible(not enabled)
        self.run_button.setEnabled(enabled)

    def _on_scenario_changed(self, scenario: str):
        self.scenario_desc_label.setText(SCENARIO_DESCRIPTIONS.get(scenario, ""))

    def _on_run_clicked(self):
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = InferenceWorker(
            normal_df=self.state.normal_df,
            feat_cols=self.state.feat_cols,
            detector=self.state.detector,
            failure_type=self.scenario_combo.currentText(),
            severity=self.severity_slider.value() / 10,
            max_granger_lag=self.lag_spin.value(),
            seed=self.state.seed,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct: int, message: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def _on_finished(self, payload: dict):
        self._last_payload = payload
        self.state.last_causal_results = payload["causal_results"]
        self.state.last_root_causes = payload["root_causes"]
        self.state.last_incident_scaled = payload["incident_scaled"]
        self.state.last_anomaly_scores = payload["anomaly_scores"]
        self.state.last_anomaly_times = payload["anomaly_times"]
        self.state.last_metadata = payload["metadata"]
        self.state.last_report = payload["report"]

        root_causes = payload["root_causes"]
        self.root_cause_table.setRowCount(len(root_causes))
        for row, rc in enumerate(root_causes):
            downstream = rc.get("downstream_effects", [])
            downstream_str = ", ".join(downstream[:3]) + (f" (+{len(downstream) - 3} more)" if len(downstream) > 3 else "")
            values = [
                str(rc["rank"]), rc["metric"], f"{rc['composite_score']:.4f}",
                rc["confidence"], f"{rc.get('scores_breakdown', {}).get('causal_outflow', 0):.3f}",
                downstream_str or "—",
            ]
            for col, val in enumerate(values):
                self.root_cause_table.setItem(row, col, QTableWidgetItem(val))
        self.root_cause_table.resizeColumnsToContents()

        causal_graph = payload["causal_results"]["causal_graph"]
        top_metric = root_causes[0]["metric"] if root_causes else ""
        self.graph_view.show_figure(draw_causal_graph(causal_graph, top_metric))

        self.timeline_view.show_figure(build_timeline_figure(
            payload["incident_scaled"], payload["anomaly_scores"], payload["anomaly_times"]
        ))

        self.report_text.setPlainText(payload["report"]["md_report"])

        self.run_button.setEnabled(True)
        self.status_label.setText("Pipeline complete")

    def _on_failed(self, message: str):
        self.status_label.setText(f"Failed: {message}")
        self.run_button.setEnabled(True)

    def _export_md(self):
        if not self._last_payload:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Markdown Report", "report.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_payload["report"]["md_report"])

    def _export_json(self):
        if not self._last_payload:
            return
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON Report", "report.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_payload["report"]["json_report"], f, indent=2, default=str)
```

- [ ] **Step 2: Commit**

```bash
git add src/desktop/views/stage2_view.py
git commit -m "feat: add Stage 2 (incident injection & RCA) view"
```

---

## Task 9: First launch and manual verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

Run: `cd D:\vscode\majorprojectt\src && python -m desktop.main`

Expected: A window titled "AI-Powered Root Cause Analysis" opens, dark themed, two tabs. Stage 2 shows "Train a model in Stage 1 first." and its Run button is disabled.

- [ ] **Step 2: Manual smoke walkthrough**

1. On Stage 1, leave defaults, click **Generate Data && Train Model**. Progress bar advances, log console prints messages, button re-enables when done (~10-30s for 5 epochs).
2. Confirm the status bar shows "Model trained — Stage 2 unlocked" and Stage 2's Run button becomes enabled.
3. Switch to Stage 2, leave defaults (`database_slow_query`, severity 0.8), click **Simulate Incident && Run Full RCA**.
4. Confirm the Root Causes table populates, the Causal Graph tab renders an interactive Plotly graph (pan/zoom/hover work), the Anomaly Timeline tab renders a line chart, and the Markdown Report tab shows report text.
5. Click **Export Markdown Report** and **Export JSON Report**, confirm both save dialogs work and files are written correctly.

- [ ] **Step 3: Record any deviations**

If anything in Step 2 fails, fix it now before proceeding — this is the primary functional gate for the whole conversion.

---

## Task 10: Automated smoke test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_desktop_smoke.py`

A minimal `pytest-qt` test that boots the window and drives one button click without a real training run (mocks the worker's heavy engine call) — catches import errors, layout crashes, and signal wiring bugs on every CI run without needing 30s of LSTM training per test.

- [ ] **Step 1: Create `tests/__init__.py`** (empty)

- [ ] **Step 2: Create `tests/test_desktop_smoke.py`**

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from desktop.main_window import MainWindow


def test_main_window_boots(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "AI-Powered Root Cause Analysis"
    assert window.tabs.count() == 2
    assert window.state.model_trained is False


def test_stage2_locked_until_trained(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.stage2.run_button.isEnabled() is False

    window.state.model_trained = True
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is True


def test_stage1_train_button_triggers_worker(qtbot, monkeypatch):
    import desktop.workers as workers_module

    class FakeTrainWorker:
        progress = None
        finished_ok = None
        failed = None

        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    calls = {"started": False}

    def fake_worker_init(self, *a, **k):
        calls["started"] = True

    monkeypatch.setattr(workers_module.TrainWorker, "start", lambda self: None)

    window = MainWindow()
    qtbot.addWidget(window)
    qtbot.mouseClick(window.stage1.train_button, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.LeftButton)

    assert window.stage1.worker is not None
```

- [ ] **Step 3: Run the smoke tests**

Run: `cd D:\vscode\majorprojectt && python -m pytest tests/test_desktop_smoke.py -v`
Expected: `3 passed`. (Runs headless via Qt's offscreen platform plugin automatically under `pytest-qt`; if it complains about no display on a CI box, set `QT_QPA_PLATFORM=offscreen` before running.)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_desktop_smoke.py
git commit -m "test: add pytest-qt smoke tests for the desktop shell"
```

---

## Task 11: Packaging with PyInstaller

**Files:**
- Create: `packaging/rca_desktop.spec`
- Create: `packaging/build.ps1`

- [ ] **Step 1: Create `packaging/rca_desktop.spec`**

```python
# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
project_root = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))
src_dir = os.path.join(project_root, "src")

a = Analysis(
    [os.path.join(src_dir, "desktop", "main.py")],
    pathex=[src_dir],
    binaries=[],
    datas=[
        (os.path.join(project_root, "best_autoencoder_model.pt"), "."),
    ],
    hiddenimports=[
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "statsmodels.tsa.stattools",
        "causal_learn",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torchvision", "torchaudio", "torchao", "pytorch_lightning", "geopandas"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RCA-Desktop",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="RCA-Desktop",
)
```

Note the `excludes` list: `pip list` on this machine showed `torchvision`, `torchaudio`, `torchao`, and `pytorch_lightning` installed globally alongside `torch`, but nothing in `src/` imports them (`requirements.txt` only lists `torch`). PyInstaller's import scanner won't pull them in on its own, but the explicit exclude is cheap insurance against a stray transitive import bloating the build — this is the single biggest lever on final `.exe` size, since bare `torch` CPU alone is already ~600MB-900MB unpacked.

- [ ] **Step 2: Create `packaging/build.ps1`**

```powershell
# Build the RCA Desktop app into a distributable folder.
# Run from the repository root: .\packaging\build.ps1

$ErrorActionPreference = "Stop"

Write-Host "Cleaning previous build..."
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "Running PyInstaller..."
pyinstaller packaging\rca_desktop.spec --noconfirm

Write-Host "Build complete: dist\RCA-Desktop\RCA-Desktop.exe"
$size = (Get-ChildItem -Recurse dist\RCA-Desktop | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Total distribution size: {0:N0} MB" -f $size)
```

- [ ] **Step 3: Run the build**

Run: `cd D:\vscode\majorprojectt && powershell -File packaging\build.ps1`
Expected: Completes without error, prints `Build complete: dist\RCA-Desktop\RCA-Desktop.exe` and a total size (expect roughly 700MB-1.1GB unpacked — this is driven by the PyTorch CPU wheel, not by PySide6; see the size note in the plan header).

- [ ] **Step 4: Smoke-test the packaged exe**

Run: `dist\RCA-Desktop\RCA-Desktop.exe`
Expected: App launches identically to Task 9's manual walkthrough, no missing-module errors. Repeat the Stage 1 → Stage 2 walkthrough from Task 9 once against the packaged build.

- [ ] **Step 5: Commit**

```bash
git add packaging/rca_desktop.spec packaging/build.ps1
git commit -m "build: add PyInstaller spec and Windows build script"
```

---

## Task 12: Update README with desktop app instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Desktop App" section after the existing "Dashboard" section (around line 141)**

```markdown
## 🖥️ Desktop App (PySide6)

A native desktop version of the dashboard, built with PySide6 (Qt 6). Same
pipeline engine as the Streamlit dashboard and the CLI — shared via
`src/pipeline/engine.py` — with native widgets for controls/tables and an
embedded, fully offline Plotly view for the causal graph and anomaly timeline.

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add desktop app run and build instructions"
```

---

## Self-Review Notes

- **Spec coverage:** native shell (Task 4), QWebEngineView graph panel (Task 7), background threading so training/inference never blocks the UI (Task 5), shared engine so no ML logic is duplicated or rewritten (Task 1), packaging into a standalone `.exe` (Task 11) — all covered.
- **Placeholder scan:** every task has complete, runnable code — no `TODO`/`fill in later` markers.
- **Type/signature consistency:** `engine.generate_data` / `preprocess` / `train_model` / `detect_anomalies` / `run_causal_inference` / `rank_root_causes` / `generate_reports` signatures are identical between Task 1 (definition), `train_and_run.py`'s CLI usage, and `workers.py`'s GUI usage. `AppState` fields match what `stage1_view.py` and `stage2_view.py` read/write.
- **Known risk carried forward:** total app size is dominated by the PyTorch CPU wheel (~600-900MB unpacked) regardless of UI framework choice — this was the core finding from the earlier framework comparison and is unaffected by anything in this plan. If size becomes a hard constraint later, the next lever is swapping `torch` for `onnxruntime` at inference time (train in torch, export the trained autoencoder to ONNX, drop the torch runtime dependency from the packaged app) — out of scope here, noted for future work.
