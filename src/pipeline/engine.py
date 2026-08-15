"""
Shared RCA pipeline engine — GUI-agnostic.

Every phase function here is a pure move from the original
src/train_and_run.py CLI script. Both the CLI entry point and the
PySide6 desktop app import from this module so there is exactly one
implementation of each pipeline phase.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    TRAINING_STRIDE,
    baseline_status,
    clean_baseline,
    contiguous_windows,
    event_incidents,
    load_events,
    load_process_attribution,
    load_samples,
    merge_incidents,
    DEFAULT_WINDOW_SIZE,
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


def baseline_readiness(db_path: str | Path, window_size: int = DEFAULT_WINDOW_SIZE):
    """Return how much clean, uninterrupted telemetry is available for training."""
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

        # weights_only=True restricts the unpickler to plain data and tensors.
        # torch 2.6+ already defaults to this, so it changes nothing today --
        # it is stated explicitly so that running against an older torch
        # cannot silently turn "load a model file" back into "execute whatever
        # is in it". Verified: the real artifact holds only OrderedDict, list,
        # str, int and float, so nothing needs the permissive loader.
        artifact = torch.load(path, map_location="cpu", weights_only=True)
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
                # Widen a short run to the model window before offering it.
                # A flagged run can be as short as min_consecutive samples,
                # which RCA cannot score -- but the run sits inside a segment
                # already known to hold at least window_size gap-free samples,
                # so the context needed to analyse it is there. Reporting only
                # the flagged rows meant a brief spike, the common case, was
                # dropped as unanalysable despite its data being present.
                first, last = run.index[0], run.index[-1]
                if last - first + 1 < detector.window_size:
                    first = max(scaled.index[0], last - detector.window_size + 1)
                incidents.append(Incident(
                    start=pd.Timestamp(scaled["timestamp"].loc[first]),
                    end=pd.Timestamp(scaled["timestamp"].loc[last]),
                    trigger="detector",
                    label="Anomalous telemetry",
                    severity=float(severity.loc[run.index].max()),
                ))

    # Only offer incidents RCA can actually analyse. Windows events are kept
    # for a year while samples exist only while the collector ran, so an event
    # fault can name a window holding no telemetry at all. Those used to be
    # listed and then failed the moment they were selected, reading as a
    # broken analysis rather than an unanalysable window.
    #
    # Merge first: two adjacent windows that are each too short can together
    # cover enough contiguous samples to analyse, and filtering first threw
    # both away before they could be combined. Merging also makes this
    # predicate run over the exact range run_real_rca will be handed.
    window_size = detector.window_size if status.exists and not samples.empty else DEFAULT_WINDOW_SIZE
    merged = merge_incidents(incidents)
    analysable = [
        incident for incident in merged
        if contiguous_windows(
            samples.loc[samples["timestamp"].between(incident.start, incident.end)],
            minimum_samples=window_size,
        )
    ]

    return sorted(analysable, key=lambda incident: incident.start, reverse=True)


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
    # load_events only adds a timestamp column to a non-empty frame, so an
    # empty event table must not be filtered on it. A machine that has logged
    # no allowlisted events yet is normal, not an error.
    relevant = (
        events.loc[events["timestamp"] >= start - pd.Timedelta(hours=24)].reset_index(drop=True)
        if not events.empty else events
    )
    return samples.loc[samples["timestamp"] >= start].reset_index(drop=True), relevant


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

    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
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


#: Beyond this, thread dispatch costs more than the parallelism gains.
MAX_TORCH_THREADS = 4


def _cap_torch_threads() -> int:
    """Limit torch to MAX_TORCH_THREADS and return what was applied."""
    import torch

    threads = min(MAX_TORCH_THREADS, os.cpu_count() or MAX_TORCH_THREADS)
    torch.set_num_threads(threads)
    return threads


def train_model(
    normal_scaled: np.ndarray | None,
    n_features: int,
    epochs: int,
    window_size: int,
    model_path: str,
    skip_train: bool,
    windows=None,
    on_epoch=None,
) -> AnomalyDetector:
    """
    Train (or reload) the LSTM Autoencoder on normal data.

    The model is saved to `model_path` after training so that subsequent
    runs can use skip_train=True for faster iteration.
    """
    # Torch defaults to one thread per core, which is measurably slower here:
    # the per-op work is tiny and the LSTM's sequential timesteps limit real
    # parallelism, so thread dispatch costs more than it saves. Measured 3.9x
    # slower at 20 threads than at 4 on this model.
    _cap_torch_threads()

    detector = AnomalyDetector(n_features=n_features, window_size=window_size)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if skip_train and os.path.exists(model_path):
        import torch
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        detector.model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
        windows = detector.create_windows(normal_scaled.astype(np.float32), stride=5)
        split = int(len(windows) * 0.8)
        val_data = windows[split:]
        detector._calibrate_thresholds(val_data)
    else:
        detector.train(
            None if normal_scaled is None else normal_scaled.astype(np.float32),
            windows=windows,
            epochs=epochs,
            lr=1e-3,
            val_split=0.2,
            batch_size=32,
            checkpoint_path=model_path,
            on_epoch=on_epoch,
        )

    return detector


# Measured on the development machine, 1,701 windows, while a build ran (so
# the constants lean slow rather than optimistic). Per-epoch cost is linear in
# the window length -- an LSTM walks every timestep -- and linear in the number
# of windows. Fitted from 1/5/10/20 epochs at window 12, and 5 epochs at
# windows 12/30/60; the held-out points land within 0.05s.
_TRAIN_FIXED_SECONDS = 2.1              # load, window, scale, score, save
_TRAIN_COLD_START_SECONDS = 5.0         # torch pulls in Dynamo on first use
_TRAIN_EPOCH_BASE = 0.40                # per epoch at the reference size
_TRAIN_EPOCH_PER_WINDOW_SAMPLE = 0.0308
_TRAIN_REFERENCE_WINDOWS = 1701

_TIMING_FILE = "timing.json"


def _timing_path() -> Path:
    from telemetry import config
    return config.app_dir() / _TIMING_FILE


def _observed_rate(key: str) -> Optional[float]:
    """A rate measured on this machine, if one has been recorded."""
    try:
        with open(_timing_path(), encoding="utf-8") as handle:
            return float(json.load(handle)[key])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _record_rate(key: str, value: float) -> None:
    """Remember a measured rate so later estimates fit this machine."""
    path = _timing_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = {}
        data[key] = value
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
    except OSError:
        pass                            # an estimate is not worth failing over


def _reference_epoch_seconds(n_windows: int, window_size: int) -> float:
    """Per-epoch cost on the machine these constants were measured on."""
    return (
        (_TRAIN_EPOCH_BASE + _TRAIN_EPOCH_PER_WINDOW_SAMPLE * window_size)
        * (max(n_windows, 1) / _TRAIN_REFERENCE_WINDOWS)
    )


def estimate_training_seconds(
    n_windows: int, window_size: int, epochs: int, cold_start: bool = True
) -> float:
    """Roughly how long training will take, before it is started.

    Training is the longest thing the app does and the settings that drive it
    are adjustable, so the cost of a choice should be visible before making
    it. Calibrated against the last real run on this machine when there has
    been one; the built-in constants are only a starting point, and a slower
    or faster machine would otherwise be quoted someone else's numbers.
    """
    # Calibrate the magnitude, not the shape: cost per epoch is
    # (base + per_timestep * window_size), not proportional to window_size, so
    # a raw seconds-per-window-per-timestep rate fitted at one window length
    # mispredicts every other length.
    per_epoch = _reference_epoch_seconds(n_windows, window_size)
    per_epoch *= _observed_rate("train_scale") or 1.0

    total = _TRAIN_FIXED_SECONDS + max(epochs, 1) * per_epoch
    if cold_start:
        total += _TRAIN_COLD_START_SECONDS
    return total


# RCA cost is dominated by Granger, which tests every ordered pair of
# anomalous metrics, and the count of those grows with the window -- so the
# total grows faster than the sample count. Fitted across 104/464/1334-sample
# windows: predicts 0.8/2.1/12.6 against 0.8/1.4/12.5 measured.
_RCA_FIXED_SECONDS = 0.7
_RCA_PER_SAMPLE_SQUARED = 6.7e-6


def estimate_rca_seconds(n_samples: int) -> float:
    """Roughly how long RCA will take on a window of this size.

    Approximate by nature: the real driver is how many metrics turn out to be
    anomalous, which is not known until the window has been scored.
    """
    observed = _observed_rate("rca_per_sample_squared") or _RCA_PER_SAMPLE_SQUARED
    return _RCA_FIXED_SECONDS + observed * max(n_samples, 0) ** 2


def format_duration(seconds: float) -> str:
    """A short human reading of an estimate."""
    if seconds < 90:
        rounded = max(round(seconds), 1)
        return f"~{rounded} second" + ("" if rounded == 1 else "s")
    minutes = seconds / 60
    if minutes < 90:
        rounded = round(minutes)
        return f"~{rounded} minute" + ("" if rounded == 1 else "s")
    return f"~{minutes / 60:.1f} hours"


def train_from_real_telemetry(
    db_path: str | Path,
    model_path: str | Path,
    epochs: int = 5,
    window_size: int = 12,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], AnomalyDetector, MinMaxScaler]:
    """Train an artifact from the collector database; never creates fake data.

    ``progress`` is called with a percentage and a description as each stage
    starts, and once per epoch during the fit. Training is the longest thing
    the app does and every epoch looks the same from outside, so without it a
    caller can only show a bar that does not move.
    """
    stage = progress or (lambda pct, message: None)
    stage(5, "Loading collected telemetry …")
    baseline, _events, features = load_real_telemetry(db_path)
    # One rule, shared with the UI gate: readiness and training must agree, or
    # the app enables a button that then fails.
    readiness = baseline_readiness(db_path, window_size=window_size)
    if not readiness.ready:
        raise ValueError(
            f"Need {readiness.required_windows:,} training windows; only "
            f"{readiness.total_windows:,} are available across all clean segments "
            f"({readiness.days_remaining * 24:.1f} more hour(s) of collection, "
            f"which may be interrupted)."
        )

    # A window must never span a sleep or collector gap, so windows are built
    # inside each clean segment and then concatenated. Using only the longest
    # segment discarded valid windows for sitting elsewhere; on a laptop that
    # sleeps, a single run long enough may never happen.
    segments = [
        segment for segment in contiguous_windows(baseline, minimum_samples=window_size)
        if len(segment) >= window_size
    ]
    if not segments:
        raise ValueError("No clean telemetry segment reaches the model window length.")

    stage(15, f"Building training windows across {len(segments)} clean segment(s) …")
    training_baseline = pd.concat(segments, ignore_index=True)
    _, scaled, scaler = preprocess(training_baseline, training_baseline, features)

    detector = AnomalyDetector(n_features=len(features), window_size=window_size)
    stacked = []
    for segment in segments:
        segment_values = np.clip(
            scaler.transform(segment[features].ffill().bfill()), 0.0, 1.0
        ).astype(np.float32)
        built = detector.create_windows(segment_values, stride=TRAINING_STRIDE)
        if len(built):
            stacked.append(built)

    import torch

    windows = torch.cat(stacked) if len(stacked) > 1 else stacked[0]

    # Epochs occupy the bulk of the runtime, so they get the bulk of the bar.
    epoch_marks: List[float] = []

    def _epoch(done: int, total: int, train_loss: float, val_loss: float) -> None:
        epoch_marks.append(time.monotonic())
        stage(
            25 + int(60 * done / max(total, 1)),
            f"Training epoch {done}/{total} — loss {train_loss:.4f}, validation {val_loss:.4f}",
        )

    stage(25, f"Training on {len(windows):,} windows …")
    detector = train_model(
        None, len(features), epochs, window_size, str(model_path), False,
        windows=windows, on_epoch=_epoch,
    )
    # Calibrate against what this machine actually did, so the next quote is
    # its own number rather than the development machine's. Measure from the
    # gaps between epochs and drop the first: torch pulls in Dynamo through
    # the optimiser on first use, which cost 4.6s of a 5.9s run here and would
    # otherwise be smeared across every epoch of the estimate.
    if len(epoch_marks) > 2:
        steady = [b - a for a, b in zip(epoch_marks[1:], epoch_marks[2:])]
        measured = sorted(steady)[len(steady) // 2]
        reference = _reference_epoch_seconds(len(windows), window_size)
        if reference > 0:
            _record_rate("train_scale", measured / reference)

    stage(90, "Measuring the model's reference error …")
    reference_error = median_recon_error(detector, scaled, features)
    stage(95, "Saving the trained model …")
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
    anomaly_scores: Dict[str, float],
    anomaly_times: Dict[str, pd.Timestamp],
    active_anomalies: List[str],
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
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """Score an observed window and perform RCA on real events.

    Pass ``start``/``end`` for a chosen incident or custom range; otherwise the
    most recent ``hours`` are used.

    ``progress`` is called with a percentage and a description as each stage
    starts. Causal inference dominates the runtime, so without it a caller has
    no way to tell a long analysis apart from a hung one.
    """
    stage = progress or (lambda pct, message: None)
    stage(10, "Loading the trained model …")
    detector, scaler, features = load_model_artifact(model_path)
    status = model_status(model_path)
    stage(25, "Loading collected telemetry for the selected window …")
    if start is not None and end is not None:
        incident, events = window_between(db_path, start, end)
    else:
        incident, events = recent_real_window(db_path, hours)
    stage(40, "Validating the observed window …")
    contiguous = contiguous_windows(incident, minimum_samples=detector.window_size)
    if not contiguous:
        raise ValueError("The selected incident window contains no uninterrupted model-length segment.")
    # Largest segment, not the last. For an event-triggered incident the
    # interesting data is *before* the event, so taking the trailing fragment
    # after a sleep gap would analyse the wrong side of the crash.
    incident = max(contiguous, key=len)
    if len(incident) < detector.window_size:
        raise ValueError("The selected incident window is shorter than the model window.")
    missing = [feature for feature in features if feature not in incident]
    if missing:
        raise ValueError(f"Collected telemetry is missing model features: {', '.join(missing)}")
    clean = incident[features].ffill().bfill()
    scaled_values = np.clip(scaler.transform(clean), 0.0, 1.0)
    scaled = pd.DataFrame(scaled_values, columns=features)
    scaled.insert(0, "timestamp", incident["timestamp"].reset_index(drop=True))
    stage(55, "Scoring anomaly windows …")
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

    stage(70, "Building the constrained causal graph …")
    results = run_causal_inference(
        scaled, scores, times, active, events_df=events, max_lag=max_lag,
    )
    stage(85, "Ranking root causes and attributing processes …")
    first_anomaly = min(times.values())
    process_attribution = load_process_attribution(
        first_anomaly - pd.Timedelta(minutes=15), incident["timestamp"].max(), db_path,
    ).to_dict(orient="records")
    results["process_attribution"] = process_attribution

    graph = results.get("causal_graph")
    edges = graph.number_of_edges() if graph is not None else 0

    # "Nothing survived the gates" and "nothing was ever tested" produced an
    # identical empty graph and an identical report, which is the difference
    # between a negative result and no result at all. Granger skips any pair
    # with fewer than max_lag * 3 aligned samples, and differencing for
    # stationarity costs up to two more, so a short window silently tests
    # nothing: 16 samples at lag 5 yielded no comparisons whatsoever.
    tested_pairs = len(results.get("granger_results") or {})
    minimum_for_causality = max_lag * 3 + 2
    too_short = tested_pairs == 0 and len(incident) < minimum_for_causality
    # An edge can also be lost *after* passing every statistical gate, because
    # the subsystem topology forbids that direction. Measured: a disk-fault
    # window accepted net_sent_bps -> cpu_pct_max_core and the map has no
    # network-to-CPU path, so the graph emptied. Reporting that as "nothing
    # survived correction" blames the statistics for a decision the topology
    # made, which is the same overstatement in the opposite direction.
    pruned_by_topology = max(0, tested_pairs - edges)
    if edges:
        support = "supported"
    elif too_short:
        support = "not tested - window too short"
    elif pruned_by_topology and tested_pairs:
        support = "pruned by topology"
    else:
        support = "no supported causal chain"

    evidence.update({
        "anomalous_metrics": len(active),
        "surviving_causal_edges": edges,
        "attributed_processes": len(process_attribution),
        "correlated_events": len(results.get("event_correlations", []) or []),
        "causal_pairs_tested": tested_pairs,
        "pairs_pruned_by_topology": pruned_by_topology,
        "samples_needed_for_causality": minimum_for_causality,
        "causal_support": support,
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

    support = evidence.get("causal_support")
    if support == "not tested - window too short":
        lines.append("")
        lines.append(
            f"> **Causality was not tested.** Granger needs "
            f"{evidence.get('samples_needed_for_causality', 0)} samples at this lag "
            f"and the window holds {evidence.get('samples_analysed', 0)}, so no pair "
            "was compared and the graph is empty for lack of data rather than for "
            "lack of a relationship. **The ranking below therefore carries no causal "
            "evidence at all**: with no edges, every metric has equal graph influence "
            "and identical outflow, so the order reflects only which metric deviated "
            "first and by how much. Widen the range or lower the Granger max lag."
        )
    elif support == "pruned by topology":
        lines.append("")
        lines.append(
            f"> **No causal chain reported.** "
            f"{evidence.get('causal_pairs_tested', 0)} pair(s) passed multiple-testing "
            f"correction and the effect-size floor, and "
            f"{evidence.get('pairs_pruned_by_topology', 0)} of those were then removed "
            "because the subsystem map declares no path in that direction. The "
            "statistics found something the topology does not permit — either the "
            "relationship is spurious, or the map is incomplete. **No causal claim is "
            "made**, and the order below reflects timing and severity only."
        )
    elif support == "no supported causal chain":
        lines.append("")
        lines.append(
            f"> **No supported causal chain.** "
            f"{evidence.get('causal_pairs_tested', 0)} metric pair(s) were tested and "
            "no edge survived multiple-testing correction and the effect-size floor. "
            "The metrics below are correlated with the incident; no causal claim is "
            "made, and their order reflects timing and severity only."
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
