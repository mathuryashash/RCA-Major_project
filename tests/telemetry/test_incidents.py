"""Baseline readiness and incident discovery."""

import pandas as pd
import pytest

from telemetry.analysis import (
    DEFAULT_WINDOW_SIZE,
    Incident,
    baseline_status,
    event_incidents,
    merge_incidents,
    required_samples,
)


def _samples(count, start="2026-01-01"):
    ts = pd.date_range(start, periods=count, freq="30s", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "elapsed_ms": [None] + [30000] * (count - 1),
        "cpu_pct": 1.0,
        "mem_pct": 2.0,
    })


# Enough for the window count training needs, plus rows lost to the NULL-rate
# first tick and to the gap boundary in the split case.
ENOUGH = required_samples(DEFAULT_WINDOW_SIZE) + 4


def test_readiness_requires_uninterrupted_history_not_merely_total():
    """Fragmented history cannot train, so it must not report ready.

    Training rejects a baseline split by collector gaps. Reporting readiness on
    the total sample count would promise a training run that then fails.
    """
    whole = _samples(ENOUGH)
    assert baseline_status(whole, pd.DataFrame()).ready is True

    split = _samples(ENOUGH)
    midpoint = len(split) // 2
    split.loc[midpoint:, "timestamp"] += pd.Timedelta(hours=2)

    status = baseline_status(split, pd.DataFrame())
    # Enough clean samples in total, but only half of them in one run.
    assert status.clean_samples >= status.required_samples
    assert status.uninterrupted_samples < status.required_samples
    assert status.ready is False                            # so it cannot train
    assert status.days_remaining > 0


def test_days_remaining_is_zero_once_ready():
    status = baseline_status(_samples(ENOUGH), pd.DataFrame())
    assert status.days_remaining == 0.0


def test_requirement_scales_with_window_size():
    """The gate is a function of window size, not a fixed number of days."""
    assert required_samples(60) > required_samples(12)


def test_event_incidents_only_for_bad_event_ids():
    events = pd.DataFrame({
        "event_id": [41, 4624],          # unexpected shutdown, ordinary logon
        "provider": ["Kernel-Power", "Security-Auditing"],
        "timestamp": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"]),
    })
    incidents = event_incidents(events)
    assert len(incidents) == 1
    assert incidents[0].trigger == "event"
    assert "41" in incidents[0].label


def test_event_incident_window_leads_the_event():
    """The pathology precedes the crash, so the window must look backwards."""
    events = pd.DataFrame({
        "event_id": [41],
        "provider": ["Kernel-Power"],
        "timestamp": pd.to_datetime(["2026-01-01T10:00:00Z"]),
    })
    incident = event_incidents(events, lead_minutes=30, tail_minutes=5)[0]
    assert incident.start == pd.Timestamp("2026-01-01T09:30:00Z")
    assert incident.end == pd.Timestamp("2026-01-01T10:05:00Z")


def test_merge_collapses_adjacent_windows():
    """One episode must not fragment into several reports."""
    base = pd.Timestamp("2026-01-01T10:00:00Z")
    incidents = [
        Incident(base, base + pd.Timedelta(minutes=2), "detector", "a", 0.4),
        Incident(base + pd.Timedelta(minutes=4), base + pd.Timedelta(minutes=6), "detector", "b", 0.9),
    ]
    merged = merge_incidents(incidents, merge_gap_minutes=5)
    assert len(merged) == 1
    assert merged[0].end == base + pd.Timedelta(minutes=6)
    assert merged[0].severity == 0.9


def test_merge_keeps_distant_windows_separate():
    base = pd.Timestamp("2026-01-01T10:00:00Z")
    incidents = [
        Incident(base, base + pd.Timedelta(minutes=2), "detector", "a", 0.4),
        Incident(base + pd.Timedelta(minutes=30), base + pd.Timedelta(minutes=32), "detector", "b", 0.5),
    ]
    assert len(merge_incidents(incidents, merge_gap_minutes=5)) == 2


def test_event_trigger_wins_when_merged_with_detector():
    """An event-defined window is the stronger claim about what happened."""
    base = pd.Timestamp("2026-01-01T10:00:00Z")
    incidents = [
        Incident(base, base + pd.Timedelta(minutes=2), "detector", "anomaly", 0.4),
        Incident(base + pd.Timedelta(minutes=3), base + pd.Timedelta(minutes=5), "event", "Kernel-Power 41", 1.0),
    ]
    merged = merge_incidents(incidents)
    assert len(merged) == 1
    assert merged[0].trigger == "event"


def test_merge_handles_empty():
    assert merge_incidents([]) == []


def test_duration_minutes():
    base = pd.Timestamp("2026-01-01T10:00:00Z")
    assert Incident(base, base + pd.Timedelta(minutes=6), "detector", "x").duration_minutes == 6.0


def test_hardware_errors_count_as_faults_without_listed_ids():
    """WHEA-Logger is allowlisted with no id filter.

    Deriving badness from listed ids alone classified every hardware error as
    benign: no incident raised, and its lead-up left in the training baseline.
    """
    from telemetry.analysis import clean_baseline

    events = pd.DataFrame({
        "event_id": [17, 19],
        "provider": ["Microsoft-Windows-WHEA-Logger", "Microsoft-Windows-WindowsUpdateClient"],
        "timestamp": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z"]),
    })

    incidents = event_incidents(events)
    assert len(incidents) == 1                       # WHEA only
    assert "WHEA" in incidents[0].label

    samples = _samples(400)
    samples["timestamp"] = pd.date_range("2026-01-01T09:30:00Z", periods=400, freq="30s", tz="UTC")
    cleaned = clean_baseline(samples, events)
    # The hardware error's lead-up is excluded; the update event's is not.
    assert len(cleaned) < len(samples) - 1


def test_gpu_channels_are_collected_but_not_modelled_yet():
    """A newly added channel must not invalidate history collected before it.

    clean_baseline drops rows with NULLs in modelled columns, so putting a new
    channel straight into the model would discard every sample recorded before
    it existed and reset the baseline clock to zero.
    """
    from telemetry.analysis import MODELLED_COLUMNS
    from telemetry.sampler import SAMPLE_COLUMNS

    for channel in ("gpu_util_pct", "gpu_mem_used_bytes", "gpu_temp_c"):
        assert channel in SAMPLE_COLUMNS, "GPU channels must be collected"
        assert channel not in MODELLED_COLUMNS, "GPU channels must not be modelled yet"


def test_schema_migration_adds_new_channels_to_an_existing_store(tmp_path):
    """CREATE TABLE IF NOT EXISTS leaves an older table untouched."""
    from telemetry import store

    path = tmp_path / "old.db"
    conn = store.connect(path)
    conn.execute("CREATE TABLE samples (ts INTEGER PRIMARY KEY, cpu_pct REAL)")
    conn.execute("CREATE TABLE events (ts INTEGER)")
    conn.execute("CREATE TABLE proc_samples (ts INTEGER)")
    conn.execute("CREATE TABLE collection_gaps (channel TEXT)")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

    added = store._add_missing_sample_columns(conn)
    assert "gpu_temp_c" in added

    columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    assert {"gpu_util_pct", "gpu_mem_used_bytes", "gpu_temp_c"} <= columns
    assert store._add_missing_sample_columns(conn) == []      # idempotent


def test_time_remaining_counts_from_the_current_run_not_the_longest():
    """A longer earlier segment is closed and can never reach the threshold.

    Counting down from it reports an arrival time that will never happen —
    which is exactly what several collector restarts produced in practice.
    """
    from telemetry.analysis import required_samples

    # 400 clean samples, then a gap, then a shorter current run of 100.
    first = _samples(400)
    second = _samples(100, start="2026-01-02")          # far enough to be a gap
    samples = pd.concat([first, second], ignore_index=True)

    status = baseline_status(samples, pd.DataFrame())
    needed = required_samples()

    assert status.uninterrupted_samples == 399          # longest, minus first-tick row
    assert status.current_run_samples == 99             # the run that can still grow
    # Remaining must be based on the current run, so it is the LARGER figure.
    expected_hours = (needed - status.current_run_samples) * 30 / 3600
    assert status.hours_remaining == pytest.approx(expected_hours, abs=0.01)
    assert status.hours_remaining > (needed - status.uninterrupted_samples) * 30 / 3600
