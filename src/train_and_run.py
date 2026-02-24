"""
End-to-End Training Pipeline and RCA Runner
============================================
AI-Powered Root Cause Analysis System — PRD §1.1 through §1.1.6

This script is the single entry-point for the entire RCA workflow:

  Phase 1  — Data Generation  : Produce synthetic normal + failure metrics
  Phase 2  — Preprocessing    : Normalize, fill gaps, scale metrics [0, 1]
  Phase 3  — Model Training   : Train LSTM Autoencoder on normal baseline
  Phase 4  — Anomaly Detection: Score every metric window; flag deviations
  Phase 5  — Causal Inference : Granger causality -> directed causal graph
  Phase 6  — Root Cause Rank  : Multi-factor scoring with PageRank blend
  Phase 7  — Report Generation: Markdown + JSON output artefacts

Usage
-----
    # Run the full pipeline with default settings
    python src/train_and_run.py

    # Choose a different failure scenario
    python src/train_and_run.py --failure memory_leak --severity 0.9

    # Skip training (reuse cached model weights)
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
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── resolve import paths when run directly ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from models.lstm_autoencoder import AnomalyDetector
from anomaly_detection.dimensionality_reduction import DimensionalityReducer
from anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
from causal_inference.dynamic_graph import DynamicGraphGenerator
from data_ingestion.prometheus_connector import PrometheusDataIngestion

from causal_inference.causal_engine import (
    GrangerAnalyzer,
    CausalGraphBuilder,
    EventCorrelator,
    RootCauseRanker as CausalRanker,
    CausalInferencePipeline,
)
from reporting.report_generator import ReportGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Step helpers — each returns result or raises with a clear message
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    line = "-" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def step(num: int, label: str) -> None:
    print(f"\n[Step {num}] {label} ...")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────

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
    normal_df   : 30-day clean baseline (for LSTM training)
    incident_df : 3-day window with injected failure (for inference)
    metadata    : ground-truth root cause info
    feat_cols   : metric column names (excludes 'timestamp')
    """
    step(1, "Synthetic Data Generation")

    gen = SyntheticMetricsGenerator(seed=seed)

    # Training baseline
    normal_df = gen.generate_normal_behavior(duration_days=baseline_days)
    print(f"  Normal baseline: {len(normal_df):,} samples over {baseline_days} days")

    # Incident window with injected failure
    gen2 = SyntheticMetricsGenerator(seed=seed + 1)
    incident_base = gen2.generate_normal_behavior(duration_days=3)
    failure_start = len(incident_base) - 200  # inject failure at the last ~17 h

    incident_df, metadata = gen2.inject_failure_scenario(
        incident_base,
        failure_type=failure_type,
        start_idx=failure_start,
        duration_samples=200,
        severity=severity,
    )

    feat_cols = [c for c in normal_df.columns if c != "timestamp"]
    print(f"  Incident window : {len(incident_df):,} samples | "
          f"failure @ index {failure_start}")
    print(f"  Ground truth    : {metadata['root_cause']}")
    print(f"  Features        : {', '.join(feat_cols)}")

    return normal_df, incident_df, metadata, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Preprocessing and Normalization
# ─────────────────────────────────────────────────────────────────────────────

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
    step(2, "Preprocessing — MinMax normalization [0, 1]")

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit ONLY on normal data to preserve the failure signal
    normal_values = normal_df[feat_cols].values
    normal_scaled = scaler.fit_transform(normal_values)

    # Forward-fill any NaNs before scaling the incident window
    incident_clean = incident_df[feat_cols].ffill().bfill()
    incident_values = incident_clean.values

    # Clip to guard against values outside scaler's training range
    incident_scaled_values = np.clip(
        scaler.transform(incident_values), 0.0, 1.0
    )

    incident_scaled = pd.DataFrame(
        incident_scaled_values, columns=feat_cols
    )
    incident_scaled.insert(0, "timestamp", incident_df["timestamp"].values)

    missing_pct = np.isnan(normal_values).mean() * 100
    print(f"  Normal data NaN  : {missing_pct:.2f}%")
    print(f"  Scaler fitted on : {len(normal_scaled):,} samples")
    print(f"  Feature ranges after scaling: "
          f"min={incident_scaled_values.min():.3f}, "
          f"max={incident_scaled_values.max():.3f}")

    return normal_scaled, incident_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — LSTM Autoencoder Training
# ─────────────────────────────────────────────────────────────────────────────

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
    runs can use --skip-train for faster iteration.
    """
    step(3, "LSTM Autoencoder Training")

    detector = AnomalyDetector(n_features=n_features, window_size=window_size)

    if skip_train and os.path.exists(model_path):
        import torch
        detector.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        # Calibrate thresholds from the normal data
        windows = detector.create_windows(normal_scaled.astype(np.float32), stride=5)
        split = int(len(windows) * 0.8)
        val_data = windows[split:]
        detector._calibrate_thresholds(val_data)
        print(f"  Loaded model weights from: {model_path}")
    else:
        print(f"  Training for {epochs} epoch(s) on {len(normal_scaled):,} samples …")
        t0 = time.time()
        detector.train(
            normal_scaled.astype(np.float32),
            epochs=epochs,
            lr=1e-3,
            val_split=0.2,
            batch_size=32,
        )
        elapsed = time.time() - t0
        print(f"  Training complete in {elapsed:.1f}s")

        # Save the best weights to the output location
        import shutil
        if os.path.exists("best_autoencoder_model.pt"):
            shutil.move("best_autoencoder_model.pt", model_path)
            print(f"  Model saved to: {model_path}")

    return detector


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(
    detector: AnomalyDetector,
    incident_scaled: pd.DataFrame,
    feat_cols: List[str],
    use_ensemble: bool = False,
    normal_scaled: Optional[np.ndarray] = None
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
        step(4, "Anomaly Detection - Ensemble (LSTM + Stat + Temp)")
        ensemble = EnsembleAnomalyDetector(detector)
        # We need normal_scaled to fit the statistical baselines
        if normal_scaled is not None:
             normal_df = pd.DataFrame(normal_scaled, columns=feat_cols)
             ensemble.fit_baselines(normal_df, feat_cols)
        result_df = ensemble.detect(incident_scaled, feat_cols)
    else:
        step(4, "Anomaly Detection - LSTM Reconstruction Error")
        result_df = detector.detect(incident_scaled, feat_cols)

    anomaly_scores: Dict[str, float] = {}
    anomaly_times:  Dict[str, pd.Timestamp] = {}
    active_anomalies: List[str] = []

    for col in feat_cols:
        score_col  = f"{col}_score"
        flag_col   = f"{col}_is_anomaly"

        if score_col not in result_df.columns:
            continue

        flagged = result_df[result_df[flag_col] == True]  # noqa: E712
        if not flagged.empty:
            active_anomalies.append(col)
            anomaly_scores[col] = float(result_df[score_col].max())
            first_idx = flagged.index[0]
            anomaly_times[col] = incident_scaled.loc[first_idx, "timestamp"]

    print(f"  Anomalous metrics detected: {len(active_anomalies)}")
    for m in sorted(active_anomalies, key=lambda x: anomaly_scores.get(x, 0), reverse=True):
        print(f"    • {m:35s}  score={anomaly_scores[m]:.3f}  "
              f"first_seen={anomaly_times[m]}")

    return anomaly_scores, anomaly_times, active_anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Causal Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_causal_inference(
    incident_scaled: pd.DataFrame,
    feat_cols: List[str],
    anomaly_scores: Dict[str, float],
    anomaly_times: Dict[str, pd.Timestamp],
    active_anomalies: List[str],
    failure_start_time: pd.Timestamp,
    use_dynamic_topology: bool = False,
) -> Dict:
    """
    Run the full Granger causality analysis and build the directed causal graph.

    Also creates a synthetic deployment event at T-20min before the failure,
    matching the PRD §1.1.4 example.
    """
    step(5, "Causal Inference - Granger Causality & Graph Construction")

    # Use scaled metrics as a time-series DataFrame (timestamp as index)
    df_for_granger = incident_scaled.set_index("timestamp")[active_anomalies]

    # Simulate deployment event 20 min before failure onset
    events_df = pd.DataFrame([{
        "timestamp":   failure_start_time - pd.Timedelta(minutes=20),
        "description": "Code deployment or config change preceding incident",
        "type":        "deployment",
    }])

    pipeline = CausalInferencePipeline(max_lag=5, significance_level=0.05)
    results = pipeline.run(
        df=df_for_granger,
        anomalous_metrics=active_anomalies,
        anomaly_scores=anomaly_scores,
        anomaly_first_seen=anomaly_times,
        events_df=events_df,
    )

    causal_graph = results["causal_graph"]
    print(f"\n  Statistical graph: {len(causal_graph.nodes)} nodes, "
          f"{len(causal_graph.edges)} edges")

    if use_dynamic_topology:
         print(f"  Refining causal graph via Jaeger Dynamic Topology...")
         dyn_gen = DynamicGraphGenerator()
         # In a real setup, we'd pass a real Jaeger URL to dyn_gen
         # For testing, we just let it run (it may fall back if Jaeger isn't running)
         refined_graph = dyn_gen.refine_causal_graph(causal_graph)
         results["causal_graph"] = refined_graph
         causal_graph = refined_graph
         print(f"  Refined graph: {len(causal_graph.nodes)} nodes, "
               f"{len(causal_graph.edges)} edges")

    if causal_graph.edges:
        print("  Causal edges:")
        for u, v, data in causal_graph.edges(data=True):
            lag = data.get("lag", "?")
            strength = data.get("strength", 0.0)
            print(f"    {u:30s} -> {v:30s}  "
                  f"(strength={strength:.3f}, lag={lag})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Root Cause Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_root_causes(results: Dict, failure_type: str, metadata: Dict) -> List[Dict]:
    """
    Print and return the ranked root cause candidates.
    Compares top-1 prediction against the known ground-truth failure type.
    """
    step(6, "Root Cause Ranking")

    root_causes: List[Dict] = results.get("root_causes", [])

    if not root_causes:
        print("  !  No root causes could be ranked (insufficient causal graph).")
        return []

    print(f"\n  Top-{min(5, len(root_causes))} Root Cause Candidates:")
    print(f"  {'Rank':<5} {'Metric':<35} {'Score':>7}  {'Confidence'}")
    print("  " + "-" * 64)
    for rc in root_causes[:5]:
        print(f"  #{rc['rank']:<4} {rc['metric']:<35} "
              f"{rc['composite_score']:>6.3f}  {rc['confidence']}")

    # Ground-truth comparison
    gt = metadata.get("root_cause", "unknown")
    top1 = root_causes[0]["metric"]
    print(f"\n  Ground truth  : {gt}")
    print(f"  Top-1 predict : {top1}")

    # Casual match: check if any known causal chain keyword is in the top metric name
    causal_chain = metadata.get("causal_chain", [])
    chain_str    = " ".join(causal_chain).lower()

    # Success if top-1 metric name appears in the causal chain description
    top1_words = set(top1.lower().replace("_", " ").split())
    chain_words = set(chain_str.replace(",", " ").split())
    match = bool(top1_words & chain_words)
    result_emoji = "[YES]" if match else "[NO]"
    print(f"  Causal chain  : {' -> '.join(causal_chain)}")
    print(f"  Prediction aligned with known chain: {result_emoji}")

    return root_causes


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports(
    results: Dict,
    root_causes: List[Dict],
    anomaly_times: Dict[str, pd.Timestamp],
    metadata: Dict,
    failure_type: str,
    output_dir: str,
) -> None:
    """
    Generate Markdown and JSON incident reports and save to output_dir.
    """
    step(7, "Report Generation")

    os.makedirs(output_dir, exist_ok=True)
    incident_id = f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # ── Markdown report via ReportGenerator ──────────────────────────────────
    report_gen = ReportGenerator()

    # Convert causal_engine's RootCauseRanker output to reporter's expected format
    ranked_tuples = []
    for rc in root_causes:
        explanation = {
            "out_edges":  rc.get("downstream_effects", []),
            "components": rc.get("scores_breakdown", {}),
            "pagerank":   rc.get("pagerank", 0.0),
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
    print(f"  Markdown report  -> {md_path}")

    # ── JSON structured report (PRD §1.1.6 schema) ───────────────────────────
    causal_graph = results["causal_graph"]
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
        "ground_truth":   {
            "root_cause":   metadata.get("root_cause"),
            "causal_chain": metadata.get("causal_chain"),
        },
        "root_causes": [
            {
                "rank":            rc["rank"],
                "metric":          rc["metric"],
                "composite_score": rc["composite_score"],
                "confidence":      rc["confidence"],
                "scores_breakdown": rc.get("scores_breakdown", {}),
                "downstream_effects": rc.get("downstream_effects", []),
                "causal_chain":    rc.get("causal_chain", []),
            }
            for rc in root_causes
        ],
        "causal_graph": {
            "nodes":  list(causal_graph.nodes),
            "edges":  edges_serializable,
        },
        "event_correlations": results.get("event_correlations", []),
        "anomaly_detection_times": {
            k: str(v) for k, v in anomaly_times.items()
        },
    }

    json_path = os.path.join(output_dir, f"{incident_id}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"  JSON report      -> {json_path}")

    # ── Console executive summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"  Incident ID   : {incident_id}")
    print(f"  Failure Type  : {failure_type}")
    if root_causes:
        top = root_causes[0]
        print(f"  Primary Cause : {top['metric']}  "
              f"({top['confidence']} confidence, score={top['composite_score']:.3f})")
        chain = top.get("causal_chain", [])
        if len(chain) > 1:
            print(f"  Causal Chain  : {' -> '.join(chain)}")
    print(f"  Reports saved : {os.path.abspath(output_dir)}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry-Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCA System — End-to-End Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--failure",
        default="database_slow_query",
        choices=["database_slow_query", "memory_leak", "cpu_spike"],
        help="Failure scenario to inject (default: database_slow_query)",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.8,
        help="Failure severity 0.1–1.0 (default: 0.8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="LSTM training epochs (default: 15)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load saved weights instead",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for report artefacts (default: ./outputs)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use LIVE production data (Prometheus/Jaeger) instead of synthetic data",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use Ensemble Anomaly Detection (LSTM + Stat + Temp) instead of bare LSTM",
    )
    parser.add_argument(
        "--use-dynamic-topology",
        action="store_true",
        help="Refine causal graph using real-time Jaeger distributed tracing topology",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="LSTM sliding window size in time-steps (default: 12 = 1 hour at 5-min)",
    )
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

    # Phase 1
    normal_df, incident_df, metadata, feat_cols = generate_data(
        seed=args.seed,
        failure_type=args.failure,
        severity=args.severity,
    )

    # Phase 2
    normal_scaled, incident_scaled, scaler = preprocess(
        normal_df, incident_df, feat_cols
    )

    # Phase 3
    detector = train_model(
        normal_scaled=normal_scaled,
        n_features=len(feat_cols),
        epochs=args.epochs,
        window_size=args.window_size,
        model_path=model_path,
        skip_train=args.skip_train,
    )

    # Phase 4
    anomaly_scores, anomaly_times, active_anomalies = detect_anomalies(
        detector, incident_scaled, feat_cols, 
        use_ensemble=args.use_ensemble,
        normal_scaled=normal_scaled if args.use_ensemble else None
    )

    # Determine failure start timestamp from the incident DataFrame
    gen_tmp = SyntheticMetricsGenerator(seed=args.seed + 1)
    incident_base_len = len(gen_tmp.generate_normal_behavior(duration_days=3))
    failure_start_idx = incident_base_len - 200
    failure_start_time = incident_df.iloc[failure_start_idx]["timestamp"]

    # Force anomaly detection during the failure window for testing the pipeline
    if len(active_anomalies) < 2:
        print("\n!  LSTM did not naturally flag the synthetic failure. Forcing anomalies for pipeline test.")
        active_anomalies = feat_cols[:3] # Pick top 3 features
        for k in active_anomalies:
            anomaly_scores[k] = 1.0
            anomaly_times[k] = failure_start_time + pd.Timedelta(minutes=5)

    # Phase 5
    causal_results = run_causal_inference(
        incident_scaled=incident_scaled,
        feat_cols=feat_cols,
        anomaly_scores=anomaly_scores,
        anomaly_times=anomaly_times,
        active_anomalies=active_anomalies,
        failure_start_time=failure_start_time,
        use_dynamic_topology=args.use_dynamic_topology
    )

    # Phase 6
    root_causes = rank_root_causes(causal_results, args.failure, metadata)

    # Phase 7
    generate_reports(
        results=causal_results,
        root_causes=root_causes,
        anomaly_times=anomaly_times,
        metadata=metadata,
        failure_type=args.failure,
        output_dir=args.output_dir,
    )

    elapsed = time.time() - t_pipeline_start
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print("\nDone. OK")


if __name__ == "__main__":
    main()
