"""Model lifecycle, explicit-range windows, and report evidence."""

import sqlite3

import networkx as nx
import pandas as pd
import pytest

from pipeline import engine
from telemetry import store


def _db(tmp_path, rows=120, start_ts=1_800_000_000):
    """A store with real-shaped rows written through the real schema."""
    path = tmp_path / "telemetry.db"
    conn = store.connect(path)
    store.init_schema(conn)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(samples)")]
    values = {name: 1.0 for name in columns}
    for index in range(rows):
        values.update({"ts": start_ts + index * 30, "elapsed_ms": None if index == 0 else 30000})
        conn.execute(
            f"INSERT INTO samples ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})",
            [values[name] for name in columns],
        )
    conn.close()
    return path


def test_model_status_reports_missing_artifact(tmp_path):
    status = engine.model_status(tmp_path / "absent.pt")
    assert status.exists is False
    assert "No model" in status.reason


def test_model_status_rejects_a_file_that_is_not_a_bundle(tmp_path):
    """A corrupt or foreign file must not crash the UI that polls this."""
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a torch archive")
    status = engine.model_status(path)
    assert status.exists is False
    assert status.reason


def test_model_status_reads_age_and_reference_error(tmp_path):
    import torch

    path = tmp_path / "model.pt"
    torch.save({
        "feature_columns": ["cpu_pct"],
        "created_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=4)).isoformat(),
        "reference_recon_error": 0.25,
    }, path)

    status = engine.model_status(path)
    assert status.exists is True
    assert status.age_days == pytest.approx(4.0, abs=0.1)
    assert status.reference_error == 0.25


def test_window_between_selects_only_the_requested_range(tmp_path):
    path = _db(tmp_path, rows=120, start_ts=1_800_000_000)
    start = pd.Timestamp(1_800_000_000 + 30 * 10, unit="s", tz="UTC")
    end = pd.Timestamp(1_800_000_000 + 30 * 20, unit="s", tz="UTC")

    window, _events = engine.window_between(path, start, end)

    assert len(window) == 11                       # inclusive on both ends
    assert window["timestamp"].min() >= start
    assert window["timestamp"].max() <= end


def test_window_between_raises_on_empty_store(tmp_path):
    path = tmp_path / "empty.db"
    conn = store.connect(path)
    store.init_schema(conn)
    conn.close()
    with pytest.raises(ValueError):
        engine.window_between(path, pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC"))


def test_detect_incidents_applies_lookback_to_events_too(tmp_path):
    """A 'last N hours' view must not surface events from the whole retention.

    Events are kept for a year while the detector side is filtered, so an
    unfiltered event path would show year-old crashes in a 7-day view.
    """
    path = _db(tmp_path, rows=10, start_ts=1_800_000_000)
    conn = sqlite3.connect(str(path))
    recent = 1_800_000_000 - 3600                  # inside a 24h lookback
    ancient = 1_800_000_000 - 90 * 86400           # long outside it
    conn.executemany(
        "INSERT INTO events (ts, channel, record_id, provider, event_id, level)"
        " VALUES (?,?,?,?,?,?)",
        [(recent, "System", 1, "Kernel-Power", 41, "Critical"),
         (ancient, "System", 2, "Kernel-Power", 41, "Critical")],
    )
    conn.commit()
    conn.close()

    incidents = engine.detect_incidents(path, tmp_path / "absent.pt", lookback_hours=24)

    assert len(incidents) == 1
    assert incidents[0].start > pd.Timestamp(ancient, unit="s", tz="UTC")


def test_evidence_markdown_flags_absence_of_a_causal_chain():
    """No surviving edge must be stated, not quietly presented as a result."""
    results = {
        "evidence": {
            "window_start": "a", "window_end": "b", "trigger": "detector",
            "samples_analysed": 100, "anomalous_metrics": 2,
            "surviving_causal_edges": 0, "correlated_events": 0,
            "attributed_processes": 0,
            "causal_support": "no supported causal chain",
        },
        "process_attribution": [],
    }
    markdown = engine._evidence_markdown(results)
    assert "No supported causal chain" in markdown
    assert "no causal claim is made" in markdown


def test_evidence_markdown_warns_when_model_is_stale():
    results = {
        "evidence": {
            "window_start": "a", "window_end": "b", "trigger": "event",
            "samples_analysed": 100, "surviving_causal_edges": 3,
            "model_stale": True, "drift_ratio": 4.2,
        },
        "process_attribution": [{"name": "chrome.exe", "samples": 5,
                                 "avg_cpu_pct": 12.0, "max_rss_bytes": 5.9e9,
                                 "io_bytes": 1.2e6}],
    }
    markdown = engine._evidence_markdown(results)
    assert "Model may be stale" in markdown
    assert "4.2x" in markdown
    assert "chrome.exe" in markdown


def test_evidence_markdown_notes_purged_process_detail():
    results = {"evidence": {"window_start": "a", "window_end": "b",
                            "surviving_causal_edges": 1},
               "process_attribution": []}
    assert "purged after 30 days" in engine._evidence_markdown(results)


def test_evidence_markdown_is_empty_without_evidence():
    assert engine._evidence_markdown({}) == ""
