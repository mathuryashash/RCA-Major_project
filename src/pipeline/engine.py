"""
Shared RCA pipeline engine — GUI-agnostic.

Every phase function here is a pure move from the original
src/train_and_run.py CLI script. Both the CLI entry point and the
PySide6 desktop app import from this module so there is exactly one
implementation of each pipeline phase.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.lstm_autoencoder import AnomalyDetector
from anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
from causal_inference.dynamic_graph import DynamicGraphGenerator
from causal_inference.causal_engine import CausalInferencePipeline
from reporting.report_generator import ReportGenerator
from telemetry.analysis import (
    baseline_status,
    clean_baseline,
    contiguous_windows,
    load_events,
    load_process_attribution,
    load_samples,
    modelled_features,
)
from telemetry.config import SYSTEM_CADENCE_S


def load_real_telemetry(db_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load collected local telemetry and return only a clean trainable baseline."""
    samples = load_samples(db_path)
    events = load_events(db_path)
    baseline = clean_baseline(samples, events)
    features = modelled_features(baseline)
    if not features:
        raise ValueError("No usable telemetry features have been collected yet.")
    return baseline, events, features


def baseline_readiness(db_path: str | Path):
    """Return the number of clean samples/days available for model training."""
    return baseline_status(load_samples(db_path), load_events(db_path))


def recent_real_window(db_path: str | Path, hours: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the latest real incident window and relevant real Windows events."""
    samples = load_samples(db_path)
    events = load_events(db_path)
    if samples.empty:
        raise ValueError("No collected telemetry is available for RCA.")
    start = samples["timestamp"].max() - pd.Timedelta(hours=hours)
    return (
        samples.loc[samples["timestamp"] >= start].reset_index(drop=True),
        events.loc[events["timestamp"] >= start - pd.Timedelta(hours=24)].reset_index(drop=True),
    )


def save_model_artifact(
    detector: AnomalyDetector, scaler: MinMaxScaler, feature_columns: List[str], model_path: str | Path,
) -> None:
    """Persist model, thresholds, scaler, and feature order as one reloadable artifact."""
    import torch

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "window_size": detector.window_size,
        "state_dict": detector.model.state_dict(),
        # Lists keep this artifact compatible with PyTorch's safe
        # ``weights_only`` loader (the default in PyTorch 2.6+).
        "threshold_per_metric": np.asarray(detector.threshold_per_metric).tolist(),
        "scaler_min": scaler.min_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist(),
        "scaler_data_range": scaler.data_range_.tolist(),
        "scaler_n_features_in": scaler.n_features_in_,
    }, path)


def load_model_artifact(model_path: str | Path) -> Tuple[AnomalyDetector, MinMaxScaler, List[str]]:
    """Load an artifact written by :func:`save_model_artifact`."""
    import torch

    artifact = torch.load(model_path, map_location="cpu")
    required = {"feature_columns", "window_size", "state_dict", "threshold_per_metric"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact is not a supported telemetry model bundle.")
    features = list(artifact["feature_columns"])
    detector = AnomalyDetector(n_features=len(features), window_size=int(artifact["window_size"]))
    detector.model.load_state_dict(artifact["state_dict"])
    detector.threshold_per_metric = np.asarray(artifact["threshold_per_metric"])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = np.asarray(artifact["scaler_min"])
    scaler.scale_ = np.asarray(artifact["scaler_scale"])
    scaler.data_min_ = np.asarray(artifact["scaler_data_min"])
    scaler.data_max_ = np.asarray(artifact["scaler_data_max"])
    scaler.data_range_ = np.asarray(artifact["scaler_data_range"])
    scaler.n_features_in_ = int(artifact["scaler_n_features_in"])
    return detector, scaler, features


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
        checkpoint = torch.load(model_path, map_location="cpu")
        detector.model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
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
            checkpoint_path=model_path,
        )

    return detector


def train_from_real_telemetry(
    db_path: str | Path,
    model_path: str | Path,
    epochs: int = 5,
    window_size: int = 12,
) -> Tuple[pd.DataFrame, List[str], AnomalyDetector, MinMaxScaler]:
    """Train an artifact from the collector database; never creates fake data."""
    baseline, _events, features = load_real_telemetry(db_path)
    readiness = baseline_readiness(db_path)
    if not readiness.ready:
        raise ValueError(
            f"Need 3 clean days before training; only {readiness.clean_days:.2f} day(s) are available."
        )
    segments = contiguous_windows(baseline, minimum_samples=window_size * 3)
    if not segments:
        raise ValueError("No uninterrupted telemetry segment is long enough for the requested model window.")
    # Never create a sequence window across a sleep/collector gap.  The
    # longest clean segment is a conservative baseline until segmented model
    # training is introduced.
    training_baseline = max(segments, key=len)
    uninterrupted_days = len(training_baseline) * SYSTEM_CADENCE_S / 86400
    if uninterrupted_days < 3:
        raise ValueError(
            "Need one uninterrupted clean 3-day baseline; collector gaps currently split the available history."
        )
    values, _scaled, scaler = preprocess(training_baseline, training_baseline, features)
    detector = train_model(values, len(features), epochs, window_size, str(model_path), False)
    save_model_artifact(detector, scaler, features, model_path)
    return training_baseline, features, detector, scaler


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
    failure_start_time: Optional[pd.Timestamp] = None,
    events_df: Optional[pd.DataFrame] = None,
    max_lag: int = 5,
    use_dynamic_topology: bool = True,
) -> Dict:
    """
    Run constrained causal inference against observed metric and event data.
    """
    df_for_granger = incident_scaled.set_index("timestamp")[active_anomalies]

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


def run_real_rca(
    db_path: str | Path,
    model_path: str | Path,
    hours: int = 24,
    max_lag: int = 5,
) -> Dict:
    """Score the most recent observed window and perform RCA on real events."""
    detector, scaler, features = load_model_artifact(model_path)
    incident, events = recent_real_window(db_path, hours)
    contiguous = contiguous_windows(incident, minimum_samples=detector.window_size)
    if not contiguous:
        raise ValueError("The selected incident window contains no uninterrupted model-length segment.")
    incident = contiguous[-1]
    if len(incident) < detector.window_size:
        raise ValueError("The selected incident window is shorter than the model window.")
    missing = [feature for feature in features if feature not in incident]
    if missing:
        raise ValueError(f"Collected telemetry is missing model features: {', '.join(missing)}")
    clean = incident[features].ffill().bfill()
    scaled_values = np.clip(scaler.transform(clean), 0.0, 1.0)
    scaled = pd.DataFrame(scaled_values, columns=features)
    scaled.insert(0, "timestamp", incident["timestamp"].values)
    scores, times, active = detect_anomalies(detector, scaled, features)
    if not active:
        return {"incident": incident, "anomaly_scores": scores, "anomaly_times": times,
                "active_anomalies": active, "causal_results": None, "root_causes": []}
    results = run_causal_inference(
        scaled, features, scores, times, active, events_df=events, max_lag=max_lag,
    )
    first_anomaly = min(times.values())
    process_attribution = load_process_attribution(
        first_anomaly - pd.Timedelta(minutes=15), incident["timestamp"].max(), db_path,
    ).to_dict(orient="records")
    results["process_attribution"] = process_attribution
    return {"incident": incident, "incident_scaled": scaled, "anomaly_scores": scores,
            "anomaly_times": times, "active_anomalies": active,
            "causal_results": results, "root_causes": rank_root_causes(results),
            "process_attribution": process_attribution}


def rank_root_causes(results: Dict) -> List[Dict]:
    """Return the ranked root cause candidates (already sorted by the ranker)."""
    return results.get("root_causes", [])


def generate_reports(
    results: Dict,
    root_causes: List[Dict],
    anomaly_times: Dict[str, pd.Timestamp],
    metadata: Optional[Dict],
    failure_type: str = "observed_telemetry",
    output_dir: str = "outputs",
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
        "evidence_source": "local collected telemetry",
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
        "process_attribution": results.get("process_attribution", []),
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
