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
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

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
