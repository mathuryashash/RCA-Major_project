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

#: Providers whose every event is a fault. These are allowlisted with no id
#: filter, so deriving badness from listed ids alone would silently classify a
#: hardware error as benign. The other unfiltered providers
#: (WindowsUpdateClient, MsiInstaller) are deliberately NOT here: they are
#: change events, the laptop analogue of a deployment, not failures.
BAD_EVENT_PROVIDERS = {"Microsoft-Windows-WHEA-Logger"}


def is_bad_event(event_id, provider) -> bool:
    """Whether an event marks a fault worth excluding and explaining."""
    return event_id in BAD_EVENT_IDS or provider in BAD_EVENT_PROVIDERS


def _bad_event_mask(events: pd.DataFrame) -> pd.Series:
    providers = events["provider"] if "provider" in events else pd.Series("", index=events.index)
    return events["event_id"].isin(BAD_EVENT_IDS) | providers.isin(BAD_EVENT_PROVIDERS)


#: Sequence windows needed before the autoencoder has anything to learn from.
#: Windows are gathered across every clean segment, not just the longest, so
#: this is a total rather than a demand for one unbroken run.
MIN_TRAINING_WINDOWS = 250
TRAINING_STRIDE = 5
DEFAULT_WINDOW_SIZE = 12


def windows_in(sample_count: int, window_size: int = DEFAULT_WINDOW_SIZE) -> int:
    """Sequence windows obtainable from one uninterrupted run."""
    if sample_count < window_size:
        return 0
    return (sample_count - window_size) // TRAINING_STRIDE + 1


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
    current_run_samples: int
    required_samples: int
    ready: bool
    days_remaining: float
    total_windows: int = 0
    required_windows: int = MIN_TRAINING_WINDOWS

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
            "FROM proc_samples WHERE ts BETWEEN ? AND ? "
            # System Idle Process is Windows' accounting placeholder for unused
            # CPU, not a consumer. Left in, it tops every CPU ranking and names
            # idleness as the cause of a slowdown.
            "AND name NOT IN ('System Idle Process') "
            "GROUP BY name",
            connection,
            params=(int(start.timestamp()), int(end.timestamp())),
        )
    finally:
        connection.close()
    if frame.empty:
        return frame

    # Rank by CPU *and* by memory, not by CPU alone. Measured on an injected
    # memory fault: a process holding 1.15 GB while sleeping uses no CPU, so
    # ordering by avg_cpu_pct could never surface it -- the run detected the
    # anomaly and then attributed it to SearchIndexer, Taskmgr and MsMpEng.
    # max_rss_bytes was already selected and simply never sorted on.
    #
    # Half the slots to each, deduplicated, so a report still holds `limit`
    # rows. ProcessSampler already takes the union of the CPU-heaviest and
    # RSS-heaviest processes for the same reason; this is that rule applied
    # one layer later.
    half = max(1, limit // 2)
    ranked = pd.concat([
        frame.nlargest(half, "avg_cpu_pct"),
        frame.nlargest(half, "max_rss_bytes"),
    ]).drop_duplicates(subset="name")
    return (ranked.sort_values("avg_cpu_pct", ascending=False)
                  .head(limit)
                  .reset_index(drop=True))


def store_summary(path: Path | str | None = None) -> dict:
    """Counts, span and disk size of the collected store, for the data view."""
    db = Path(path or config.db_path())
    summary: dict = {"path": db, "exists": db.exists(), "size_bytes": 0,
                     "samples": 0, "events": 0, "proc_samples": 0, "gaps": 0,
                     "first_ts": None, "last_ts": None, "latest": {}, "available": set(),
                     "sampling_gaps": 0, "gap_hours": 0.0,
                     "expected_samples": 0, "coverage_pct": 0.0}
    if not db.exists():
        return summary

    summary["size_bytes"] = sum(
        candidate.stat().st_size
        for candidate in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm"))
        if candidate.exists()
    )
    connection = sqlite3.connect(str(db))
    try:
        for table in ("samples", "events", "proc_samples"):
            summary[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        summary["gaps"] = connection.execute("SELECT COUNT(*) FROM collection_gaps").fetchone()[0]
        bounds = connection.execute("SELECT MIN(ts), MAX(ts) FROM samples").fetchone()
        if bounds and bounds[0]:
            summary["first_ts"] = pd.to_datetime(bounds[0], unit="s", utc=True)
            summary["last_ts"] = pd.to_datetime(bounds[1], unit="s", utc=True)

            # Sampling gaps are derived from timestamps, never stored. The
            # collection_gaps table only records Event Log watermark resets, so
            # reporting it as "coverage gaps" claimed unbroken collection while
            # the sample series was actually in pieces.
            span_s = bounds[1] - bounds[0]
            expected = int(span_s / config.SYSTEM_CADENCE_S) + 1 if span_s else summary["samples"]
            summary["expected_samples"] = expected
            summary["coverage_pct"] = (100.0 * summary["samples"] / expected) if expected else 0.0

            times = pd.Series([
                row[0] for row in connection.execute("SELECT ts FROM samples ORDER BY ts")
            ])
            deltas = times.diff()
            missed = deltas > config.gap_threshold_s()
            summary["sampling_gaps"] = int(missed.sum())
            summary["gap_hours"] = float(deltas[missed].sum() / 3600.0)
        row = connection.execute("SELECT * FROM samples ORDER BY ts DESC LIMIT 1").fetchone()
        if row:
            names = [description[0] for description in connection.execute("SELECT * FROM samples LIMIT 1").description]
            summary["latest"] = dict(zip(names, row))

            # Availability is a property of the store, not of one row. Rate
            # channels are NULL on the first tick after a start or a gap, so
            # judging from the newest row alone would report a working channel
            # as unsupported by the hardware.
            counts = connection.execute(
                "SELECT " + ", ".join(f'COUNT("{name}")' for name in names) + " FROM samples"
            ).fetchone()
            summary["available"] = {
                name for name, count in zip(names, counts) if count
            }
    finally:
        connection.close()
    return summary


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
    bad = events.loc[_bad_event_mask(events), "timestamp"]
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

    The time remaining is measured from the CURRENT run, not the longest one.
    A longer segment further back is closed -- it can never grow -- so counting
    down from it reports an arrival time that will not happen. After several
    collector restarts the two diverge sharply.
    """
    clean = clean_baseline(samples, events)
    days = len(clean) * config.SYSTEM_CADENCE_S / 86400

    segments = contiguous_windows(clean, minimum_samples=1)
    longest = max((len(segment) for segment in segments), default=0)
    current = len(segments[-1]) if segments else 0
    needed = required_samples(window_size)

    # Readiness counts windows across every clean segment. A window never spans
    # a gap either way, so demanding they all come from one unbroken run threw
    # away valid training data purely for sitting in a different segment -- and
    # on a laptop that sleeps, one run long enough may never occur.
    total_windows = sum(windows_in(len(segment), window_size) for segment in segments)
    shortfall_windows = max(0, MIN_TRAINING_WINDOWS - total_windows)

    return BaselineStatus(
        clean_samples=len(clean),
        clean_days=days,
        uninterrupted_samples=longest,
        current_run_samples=current,
        required_samples=needed,
        ready=total_windows >= MIN_TRAINING_WINDOWS,
        days_remaining=shortfall_windows * TRAINING_STRIDE * config.SYSTEM_CADENCE_S / 86400,
        total_windows=total_windows,
        required_windows=MIN_TRAINING_WINDOWS,
    )


def event_incidents(events: pd.DataFrame, lead_minutes: int = 30, tail_minutes: int = 5) -> list[Incident]:
    """Windows defined by a Windows Event Log fault.

    A crash or unexpected shutdown produces no gradual metric anomaly -- the
    machine simply stops -- so a detector-only design would never surface one.
    The event defines the window and RCA asks what was abnormal beforehand.
    """
    if events.empty or "event_id" not in events:
        return []
    bad = events.loc[_bad_event_mask(events)]
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
