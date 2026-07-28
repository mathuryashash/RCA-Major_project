"""Read and prepare collected telemetry for model training and RCA."""

import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import config

MODELLED_COLUMNS = (
    "cpu_pct", "cpu_pct_max_core", "cpu_freq_mhz", "cpu_freq_ratio",
    "mem_pct", "mem_available_mb", "swap_pct", "swap_used_bytes", "swap_used_delta",
    "disk_read_bps", "disk_write_bps", "disk_busy_pct", "disk_free_pct",
    "net_sent_bps", "net_recv_bps", "process_count", "battery_pct",
    "battery_drain_rate", "power_plugged",
)
#: Derived from the collector allowlist so the two cannot drift apart.
BAD_EVENT_IDS = {
    event_id
    for ids in config.EVENT_ALLOWLIST.values() if ids
    for event_id in ids
}


#: Sequence windows needed before the autoencoder has anything to learn from.
MIN_TRAINING_WINDOWS = 500
TRAINING_STRIDE = 5
DEFAULT_WINDOW_SIZE = 12


def required_samples(window_size: int = DEFAULT_WINDOW_SIZE) -> int:
    """Uninterrupted samples needed to form MIN_TRAINING_WINDOWS windows.

    The requirement is a function of ``window_size``, not a fixed number of
    days. A flat "3 days" rule assumed a 60-sample window; at the window size
    the app actually defaults to, it demands roughly three times the data
    training needs and blocks a model that would train perfectly well.
    """
    return window_size + MIN_TRAINING_WINDOWS * TRAINING_STRIDE


@dataclass(frozen=True)
class BaselineStatus:
    """Whether enough clean telemetry exists to train.

    ``uninterrupted_samples`` is the figure that gates training: model windows
    may not bridge a collector gap, so history scattered across fragments
    cannot train even when the total looks sufficient.
    """

    clean_samples: int
    clean_days: float
    uninterrupted_samples: int
    required_samples: int
    ready: bool
    days_remaining: float

    @property
    def hours_remaining(self) -> float:
        return self.days_remaining * 24.0


@dataclass(frozen=True)
class Incident:
    """An observed window worth explaining."""

    start: pd.Timestamp
    end: pd.Timestamp
    trigger: str          # "detector" or "event"
    label: str
    severity: float = 0.0

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


def load_samples(path: Path | str | None = None) -> pd.DataFrame:
    """Return collector samples in timestamp order with real datetimes."""
    connection = sqlite3.connect(str(path or config.db_path()))
    try:
        frame = pd.read_sql_query("SELECT * FROM samples ORDER BY ts", connection)
    finally:
        connection.close()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame.pop("ts"), unit="s", utc=True)
    return frame


def load_events(path: Path | str | None = None) -> pd.DataFrame:
    connection = sqlite3.connect(str(path or config.db_path()))
    try:
        frame = pd.read_sql_query("SELECT * FROM events ORDER BY ts", connection)
    finally:
        connection.close()
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame.pop("ts"), unit="s", utc=True)
        frame["description"] = frame["provider"].fillna("Windows event") + " " + frame["event_id"].astype(str)
        frame["type"] = "windows_event"
    return frame


def load_process_attribution(
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path | str | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Aggregate retained process snapshots over an observed incident interval."""
    connection = sqlite3.connect(str(path or config.db_path()))
    try:
        frame = pd.read_sql_query(
            "SELECT name, COUNT(*) AS samples, AVG(cpu_pct) AS avg_cpu_pct, "
            "MAX(rss) AS max_rss_bytes, SUM(COALESCE(io_read_delta, 0) + COALESCE(io_write_delta, 0)) AS io_bytes "
            "FROM proc_samples WHERE ts BETWEEN ? AND ? GROUP BY name "
            "ORDER BY avg_cpu_pct DESC, io_bytes DESC LIMIT ?",
            connection,
            params=(int(start.timestamp()), int(end.timestamp()), limit),
        )
    finally:
        connection.close()
    return frame


def _gap_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    delta = frame["timestamp"].diff().dt.total_seconds()
    return delta.gt(config.gap_threshold_s()) | frame["elapsed_ms"].isna()


def clean_baseline(samples: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Exclude event lead-up/recovery windows, gaps, and unusable model rows."""
    if samples.empty:
        return samples.copy()
    usable = samples.loc[~_gap_mask(samples)].copy()
    # Battery data is legitimately absent on desktops; such an all-null channel
    # is omitted later rather than disqualifying every otherwise-clean row.
    required = [column for column in MODELLED_COLUMNS if column in usable.columns and usable[column].notna().any()]
    usable = usable.dropna(subset=required)
    if events.empty:
        return usable
    bad = events.loc[events["event_id"].isin(BAD_EVENT_IDS), "timestamp"]
    excluded = pd.Series(False, index=usable.index)
    for timestamp in bad:
        excluded |= usable["timestamp"].between(timestamp - timedelta(minutes=60), timestamp + timedelta(minutes=15))
    return usable.loc[~excluded].reset_index(drop=True)


def modelled_features(samples: pd.DataFrame) -> list[str]:
    """Columns usable by the model on this particular machine."""
    return [column for column in MODELLED_COLUMNS if column in samples and samples[column].notna().any()]


def baseline_status(
    samples: pd.DataFrame, events: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE
) -> BaselineStatus:
    """Report training readiness against the uninterrupted-segment requirement.

    Readiness deliberately keys off the longest gap-free segment rather than the
    total clean count: training rejects fragmented history, so reporting "ready"
    on the total would promise a training run that then fails.
    """
    clean = clean_baseline(samples, events)
    days = len(clean) * config.SYSTEM_CADENCE_S / 86400

    segments = contiguous_windows(clean, minimum_samples=1)
    longest = max((len(segment) for segment in segments), default=0)
    needed = required_samples(window_size)

    return BaselineStatus(
        clean_samples=len(clean),
        clean_days=days,
        uninterrupted_samples=longest,
        required_samples=needed,
        ready=longest >= needed,
        days_remaining=max(0, needed - longest) * config.SYSTEM_CADENCE_S / 86400,
    )


def event_incidents(events: pd.DataFrame, lead_minutes: int = 30, tail_minutes: int = 5) -> list[Incident]:
    """Windows defined by a Windows Event Log fault.

    A crash or unexpected shutdown produces no gradual metric anomaly -- the
    machine simply stops -- so a detector-only design would never surface one.
    The event defines the window and RCA asks what was abnormal beforehand.
    """
    if events.empty or "event_id" not in events:
        return []
    bad = events.loc[events["event_id"].isin(BAD_EVENT_IDS)]
    incidents = []
    for _, row in bad.iterrows():
        timestamp = row["timestamp"]
        provider = row.get("provider") or "Windows event"
        incidents.append(
            Incident(
                start=timestamp - timedelta(minutes=lead_minutes),
                end=timestamp + timedelta(minutes=tail_minutes),
                trigger="event",
                label=f"{provider} {int(row['event_id'])}",
                severity=1.0,
            )
        )
    return incidents


def merge_incidents(incidents: list[Incident], merge_gap_minutes: int = 5) -> list[Incident]:
    """Collapse overlapping or near-adjacent windows so one episode is one report."""
    if not incidents:
        return []
    ordered = sorted(incidents, key=lambda incident: incident.start)
    merged = [ordered[0]]
    for incident in ordered[1:]:
        last = merged[-1]
        if incident.start - last.end <= timedelta(minutes=merge_gap_minutes):
            merged[-1] = Incident(
                start=last.start,
                end=max(last.end, incident.end),
                # An event-defined window is the stronger claim, so it wins.
                trigger="event" if "event" in (last.trigger, incident.trigger) else last.trigger,
                label=last.label if last.trigger == "event" else incident.label,
                severity=max(last.severity, incident.severity),
            )
        else:
            merged.append(incident)
    return merged


def contiguous_windows(samples: pd.DataFrame, minimum_samples: int = 60) -> list[pd.DataFrame]:
    """Split history at collector gaps so no model window bridges a sleep/drop."""
    if samples.empty:
        return []
    group = _gap_mask(samples).cumsum()
    return [part.reset_index(drop=True) for _, part in samples.groupby(group) if len(part) >= minimum_samples]
