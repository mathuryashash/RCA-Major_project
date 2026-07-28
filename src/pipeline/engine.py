"""
Shared RCA pipeline engine — GUI-agnostic.

Every phase function here is a pure move from the original
src/train_and_run.py CLI script. Both the CLI entry point and the
PySide6 desktop app import from this module so there is exactly one
implementation of each pipeline phase.
"""

import os
from dataclasses import dataclass
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
    Incident,
    baseline_status,
    clean_baseline,
    contiguous_windows,
    event_incidents,
    load_events,
    load_process_attribution,
    load_samples,
    merge_incidents,
    modelled_features,
    required_samples,
)

# A model is stale when the recent median reconstruction error drifts beyond
# this multiple of its value at training time: the machine's notion of "normal"
# has moved and the thresholds no longer describe it.
STALENESS_RATIO = 2.0


def load_real_telemetry(db_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load collected local telemetry and return only a clean trainable baseline."""
    samples = load_samples(db_path)
    events = load_events(db_path)
    baseline = clean_baseline(samples, events)
    features = modelled_features(baseline)
    if not features:
        raise ValueError("No usable telemetry features have been collected yet.")
    return baseline, events, features


def baseline_readiness(db_path: str | Path, window_size: int | None = None):
    """Return how much clean, uninterrupted telemetry is available for training."""
    if window_size is None:
        return baseline_status(load_samples(db_path), load_events(db_path))
    return baseline_status(load_samples(db_path), load_events(db_path), window_size)


def median_recon_error(detector: AnomalyDetector, scaled: pd.DataFrame, features: List[str]) -> float:
    """Median across rows of the mean per-metric reconstruction error.

    One scalar summarising how well the model currently reconstructs the data,
    used as the reference point for staleness detection.
    """
    scored = detector.detect(scaled, features)
    score_columns = [f"{feature}_score" for feature in features if f"{feature}_score" in scored]
    if not score_columns:
        return 0.0
    return float(scored[score_columns].mean(axis=1).median())


@dataclass(frozen=True)
class ModelStatus:
    """Whether a trained model exists and how old it is."""

    exists: bool
    trained_at: Optional[str] = None
    age_days: Optional[float] = None
    reference_error: Optional[float] = None
    reason: str = ""


def model_status(model_path: str | Path) -> ModelStatus:
    """Does a loadable artifact exist, and how old is it?

    Deliberately does not score any data. Drift is measured during RCA, where
    the model is being run anyway -- a separate scoring pass just to colour a
    status label would cost seconds for nothing.
    """
    path = Path(model_path)
    if not path.exists():
        return ModelStatus(exists=False, reason="No model has been trained yet.")
    try:
        import torch

        artifact = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - a corrupt artifact must not crash the UI
        return ModelStatus(exists=False, reason=f"Model artifact could not be read: {exc}")
    if not isinstance(artifact, dict) or "feature_columns" not in artifact:
        return ModelStatus(exists=False, reason="Model artifact is not a telemetry model bundle.")

    trained_at = artifact.get("created_at")
    age_days = None
    if trained_at:
        try:
            age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(trained_at)).total_seconds() / 86400
        except ValueError:
            age_days = None

    return ModelStatus(
        exists=True,
        trained_at=trained_at,
        age_days=age_days,
        reference_error=artifact.get("reference_recon_error"),
    )


def detect_incidents(
    db_path: str | Path,
    model_path: str | Path,
    lookback_hours: int = 168,
    min_consecutive: int = 3,
) -> List[Incident]:
    """Discover incidents rather than being told where they are.

    Two independent triggers produce the same record: contiguous runs of
    detector-flagged rows, and Windows Event Log faults. Runs shorter than
    ``min_consecutive`` samples are dropped as single-sample noise, and windows
    within five minutes of each other are merged so one episode is one report.
    """
    samples = load_samples(db_path)
    events = load_events(db_path)

    # Both triggers must honour the same cutoff. Filtering only the detector
    # side would let a "last 7 days" view show event incidents from the full
    # 365-day event retention.
    if samples.empty:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
    else:
        cutoff = samples["timestamp"].max() - pd.Timedelta(hours=lookback_hours)
    if not events.empty:
        events = events.loc[events["timestamp"] >= cutoff]

    incidents: List[Incident] = list(event_incidents(events))

    status = model_status(model_path)
    if status.exists and not samples.empty:
        detector, scaler, features = load_model_artifact(model_path)
        recent = samples.loc[samples["timestamp"] >= cutoff]
        for segment in contiguous_windows(recent, minimum_samples=detector.window_size):
            missing = [feature for feature in features if feature not in segment]
            if missing:
                continue
            clean = segment[features].ffill().bfill()
            scaled = pd.DataFrame(np.clip(scaler.transform(clean), 0.0, 1.0), columns=features)
            # .reset_index, not .values: .values on a tz-aware Series drops the
            # timezone, and a naive timestamp cannot be compared or merged with
            # the tz-aware ones the event path produces.
            scaled.insert(0, "timestamp", segment["timestamp"].reset_index(drop=True))
            scored = detector.detect(scaled, features)

            flag_columns = [f"{feature}_is_anomaly" for feature in features if f"{feature}_is_anomaly" in scored]
            if not flag_columns:
                continue
            flagged = scored[flag_columns].any(axis=1)
            score_columns = [f"{feature}_score" for feature in features if f"{feature}_score" in scored]
            severity = scored[score_columns].mean(axis=1) if score_columns else pd.Series(0.0, index=scored.index)

            # Same cumsum-grouping idiom as contiguous_windows: each change in
            # the flag starts a new group, so anomalous runs fall out directly.
            runs = (flagged != flagged.shift()).cumsum()
            for _, run in flagged.groupby(runs):
                if not run.iloc[0] or len(run) < min_consecutive:
                    continue
                # .loc, not .iloc: detect() indexes its result from
                # window_size-1, so these are labels rather than positions.
                # They coincide only while `scaled` carries a RangeIndex.
                incidents.append(Incident(
                    start=pd.Timestamp(scaled["timestamp"].loc[run.index[0]]),
                    end=pd.Timestamp(scaled["timestamp"].loc[run.index[-1]]),
                    trigger="detector",
                    label="Anomalous telemetry",
                    severity=float(severity.loc[run.index].max()),
                ))

    return sorted(merge_incidents(incidents), key=lambda incident: incident.start, reverse=True)


def window_between(
    db_path: str | Path, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Samples and correlated events for an explicit time range."""
    samples = load_samples(db_path)
    events = load_events(db_path)
    if samples.empty:
        raise ValueError("No collected telemetry is available for RCA.")
    window = samples.loc[samples["timestamp"].between(start, end)].reset_index(drop=True)
    relevant = (
        events.loc[events["timestamp"].between(start - pd.Timedelta(hours=1), end)].reset_index(drop=True)
        if not events.empty else events
    )
    return window, relevant


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
    detector: AnomalyDetector,
    scaler: MinMaxScaler,
    feature_columns: List[str],
    model_path: str | Path,
    reference_recon_error: Optional[float] = None,
    training_samples: Optional[int] = None,
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
        # Reference point for staleness: a model does not decay on its own, but
        # it goes stale when usage moves away from what it was trained on.
        "reference_recon_error": reference_recon_error,
        "training_samples": training_samples,
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
    incident_scaled.insert(0, "timestamp", incident_df["timestamp"].reset_index(drop=True))

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
    # One rule, shared with the UI gate: readiness and training must agree, or
    # the app enables a button that then fails.
    readiness = baseline_readiness(db_path, window_size=window_size)
    if not readiness.ready:
        raise ValueError(
            f"Need {readiness.required_samples:,} uninterrupted clean samples for a "
            f"window size of {window_size}; the longest clean run is "
            f"{readiness.uninterrupted_samples:,} "
            f"({readiness.days_remaining:.2f} more day(s) of collection)."
        )
    # Never create a sequence window across a sleep/collector gap. The longest
    # clean segment is a conservative baseline until segmented training exists.
    segments = contiguous_windows(baseline, minimum_samples=required_samples(window_size))
    if not segments:
        raise ValueError("No uninterrupted telemetry segment is long enough for the requested model window.")
    training_baseline = max(segments, key=len)
    values, scaled, scaler = preprocess(training_baseline, training_baseline, features)
    detector = train_model(values, len(features), epochs, window_size, str(model_path), False)
    reference_error = median_recon_error(detector, scaled, features)
    save_model_artifact(
        detector, scaler, features, model_path,
        reference_recon_error=reference_error,
        training_samples=len(training_baseline),
    )
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
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    trigger: str = "manual",
) -> Dict:
    """Score an observed window and perform RCA on real events.

    Pass ``start``/``end`` for a chosen incident or custom range; otherwise the
    most recent ``hours`` are used.
    """
    detector, scaler, features = load_model_artifact(model_path)
    status = model_status(model_path)
    if start is not None and end is not None:
        incident, events = window_between(db_path, start, end)
    else:
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
    scaled.insert(0, "timestamp", incident["timestamp"].reset_index(drop=True))
    scores, times, active = detect_anomalies(detector, scaled, features)

    # Drift is measured here because the model has just been run over this
    # window anyway; a stale model still produces a report, clearly labelled.
    current_error = median_recon_error(detector, scaled, features)
    reference_error = status.reference_error
    drift_ratio = (
        current_error / reference_error
        if reference_error else None
    )
    stale = drift_ratio is not None and drift_ratio > STALENESS_RATIO

    evidence = {
        "trigger": trigger,
        "window_start": str(incident["timestamp"].min()),
        "window_end": str(incident["timestamp"].max()),
        "samples_analysed": int(len(incident)),
        "model_trained_at": status.trained_at,
        "model_age_days": round(status.age_days, 2) if status.age_days is not None else None,
        "reference_recon_error": reference_error,
        "current_recon_error": round(current_error, 6),
        "drift_ratio": round(drift_ratio, 3) if drift_ratio is not None else None,
        "model_stale": stale,
    }

    if not active:
        return {"incident": incident, "incident_scaled": scaled, "anomaly_scores": scores,
                "anomaly_times": times, "active_anomalies": active,
                "causal_results": None, "root_causes": [],
                "process_attribution": [], "evidence": evidence, "model_stale": stale}

    results = run_causal_inference(
        scaled, features, scores, times, active, events_df=events, max_lag=max_lag,
    )
    first_anomaly = min(times.values())
    process_attribution = load_process_attribution(
        first_anomaly - pd.Timedelta(minutes=15), incident["timestamp"].max(), db_path,
    ).to_dict(orient="records")
    results["process_attribution"] = process_attribution

    graph = results.get("causal_graph")
    evidence.update({
        "anomalous_metrics": len(active),
        "surviving_causal_edges": graph.number_of_edges() if graph is not None else 0,
        "attributed_processes": len(process_attribution),
        "correlated_events": len(results.get("event_correlations", []) or []),
        # No causal chain survived the FDR and effect-size gates: correlation
        # only, and the report must not imply otherwise.
        "causal_support": "supported" if graph is not None and graph.number_of_edges() else "no supported causal chain",
    })
    results["evidence"] = evidence

    return {"incident": incident, "incident_scaled": scaled, "anomaly_scores": scores,
            "anomaly_times": times, "active_anomalies": active,
            "causal_results": results, "root_causes": rank_root_causes(results),
            "process_attribution": process_attribution,
            "evidence": evidence, "model_stale": stale}


def rank_root_causes(results: Dict) -> List[Dict]:
    """Return the ranked root cause candidates (already sorted by the ranker)."""
    return results.get("root_causes", [])


def _evidence_markdown(results: Dict) -> str:
    """Append an honest statement of what the evidence supports.

    There is no ground truth for a real incident, so the report says how much
    of the answer is actually backed: how many causal edges survived
    correction, how many processes were attributable, and whether the model
    still matches current usage.
    """
    evidence = results.get("evidence") or {}
    if not evidence:
        return ""

    lines = ["", "---", "", "## Evidence & Confidence", ""]
    lines.append(f"- Analysis window: {evidence.get('window_start')} to {evidence.get('window_end')}")
    lines.append(f"- Triggered by: {evidence.get('trigger', 'manual')}")
    lines.append(f"- Samples analysed: {evidence.get('samples_analysed', 0)}")
    lines.append(f"- Anomalous metrics: {evidence.get('anomalous_metrics', 0)}")
    lines.append(f"- Causal edges surviving FDR and effect-size gates: {evidence.get('surviving_causal_edges', 0)}")
    lines.append(f"- Correlated Windows events: {evidence.get('correlated_events', 0)}")
    lines.append(f"- Processes attributable in window: {evidence.get('attributed_processes', 0)}")

    if evidence.get("causal_support") == "no supported causal chain":
        lines.append("")
        lines.append(
            "> **No supported causal chain.** No edge survived multiple-testing "
            "correction and the effect-size floor. The metrics below are "
            "correlated with the incident; no causal claim is made."
        )

    if evidence.get("model_stale"):
        lines.append("")
        lines.append(
            f"> **Model may be stale.** Reconstruction error is "
            f"{evidence.get('drift_ratio')}x its value at training time "
            f"(threshold {STALENESS_RATIO}x). Retrain to match current usage."
        )

    attribution = results.get("process_attribution") or []
    if attribution:
        lines += ["", "### Process attribution", "",
                  "| Process | Samples | Avg CPU % | Peak RSS (MB) | I/O (MB) |",
                  "|---|---|---|---|---|"]
        for row in attribution[:10]:
            lines.append(
                f"| {row.get('name')} | {row.get('samples')} | "
                f"{(row.get('avg_cpu_pct') or 0):.1f} | "
                f"{(row.get('max_rss_bytes') or 0) / 1e6:.0f} | "
                f"{(row.get('io_bytes') or 0) / 1e6:.1f} |"
            )
    else:
        lines += ["", "_No retained process snapshots cover this window "
                  "(process detail is purged after 30 days)._"]

    return "\n".join(lines) + "\n"


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
    md_report += _evidence_markdown(results)

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
        # A real incident has no oracle, so the report states what the evidence
        # actually supports instead of asserting a verified answer.
        "evidence": results.get("evidence", {}),
        "process_attribution": results.get("process_attribution", []),
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
