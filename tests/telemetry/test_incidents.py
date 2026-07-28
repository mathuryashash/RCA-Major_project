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
