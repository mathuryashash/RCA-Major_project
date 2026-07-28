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
BAD_EVENT_IDS = {41, 1000, 1002, 7, 51, 153, 2004}


@dataclass(frozen=True)
class BaselineStatus:
    clean_samples: int
    clean_days: float
    ready: bool


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


def baseline_status(samples: pd.DataFrame, events: pd.DataFrame) -> BaselineStatus:
    clean = clean_baseline(samples, events)
    days = len(clean) * config.SYSTEM_CADENCE_S / 86400
    return BaselineStatus(len(clean), days, days >= 3)


def contiguous_windows(samples: pd.DataFrame, minimum_samples: int = 60) -> list[pd.DataFrame]:
    """Split history at collector gaps so no model window bridges a sleep/drop."""
    if samples.empty:
        return []
    group = _gap_mask(samples).cumsum()
    return [part.reset_index(drop=True) for _, part in samples.groupby(group) if len(part) >= minimum_samples]
