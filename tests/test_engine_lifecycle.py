"""Model lifecycle, explicit-range windows, and report evidence."""

import sqlite3

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
    # Enough samples for the recent event's window to be analysable: an event
    # naming a window with no telemetry is dropped now, so a 10-row fixture
    # would test the wrong thing.
    path = _db(tmp_path, rows=200, start_ts=1_800_000_000)
    conn = sqlite3.connect(str(path))
    recent = 1_800_000_000 + 100 * 30              # inside both the samples and a 24h lookback
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


class _StubDetector:
    """Returns caller-supplied flags so run segmentation can be tested alone.

    Mirrors AnomalyDetector.detect(), which indexes its result from
    window_size-1 rather than 0 -- the offset that makes label-vs-position
    confusion easy here.
    """

    window_size = 5

    def __init__(self, flags):
        self.flags = flags

    def detect(self, df, feature_columns):
        index = df.index[self.window_size - 1:]
        flags = list(self.flags)[: len(index)]
        flags += [False] * (len(index) - len(flags))     # pad to the full index
        return pd.DataFrame(
            {"cpu_pct_is_anomaly": flags, "cpu_pct_score": [1.5 if f else 0.1 for f in flags]},
            index=index,
        )


def _patch_detector(monkeypatch, flags):
    monkeypatch.setattr(
        engine, "load_model_artifact",
        lambda path: (_StubDetector(flags), _ScalerPassthrough(), ["cpu_pct"]),
    )
    monkeypatch.setattr(
        engine, "model_status", lambda path: engine.ModelStatus(exists=True, age_days=1.0)
    )


class _ScalerPassthrough:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


def test_detector_runs_shorter_than_minimum_are_ignored(tmp_path, monkeypatch):
    """Two consecutive flags is noise, not an incident."""
    path = _db(tmp_path, rows=40)
    flags = [False] * 30
    flags[10:12] = [True, True]
    _patch_detector(monkeypatch, flags)

    incidents = engine.detect_incidents(path, tmp_path / "m.pt", min_consecutive=3)
    assert [i for i in incidents if i.trigger == "detector"] == []


def test_detector_run_becomes_an_incident_with_correct_bounds(tmp_path, monkeypatch):
    path = _db(tmp_path, rows=40)
    flags = [False] * 30
    flags[10:16] = [True] * 6
    _patch_detector(monkeypatch, flags)

    detected = [i for i in engine.detect_incidents(path, tmp_path / "m.pt", min_consecutive=3)
                if i.trigger == "detector"]
    assert len(detected) == 1
    # Result index starts at window_size-1 = 4, so flag position 10 is row 14.
    assert detected[0].start == pd.Timestamp(1_800_000_000 + 14 * 30, unit="s", tz="UTC")
    assert detected[0].end == pd.Timestamp(1_800_000_000 + 19 * 30, unit="s", tz="UTC")
    assert detected[0].severity == pytest.approx(1.5)


def test_trailing_run_at_end_of_history_is_captured(tmp_path, monkeypatch):
    """A run still in progress at the last sample must not be dropped."""
    path = _db(tmp_path, rows=40)
    flags = [False] * 26 + [True] * 10
    _patch_detector(monkeypatch, flags)

    detected = [i for i in engine.detect_incidents(path, tmp_path / "m.pt", min_consecutive=3)
                if i.trigger == "detector"]
    assert len(detected) == 1


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


def test_rca_analyses_the_largest_segment_not_the_trailing_one(tmp_path, monkeypatch):
    """A gap must not push analysis onto the wrong side of an incident.

    For an event-triggered crash the interesting data is *before* the event, so
    taking the trailing fragment after a sleep gap analyses the aftermath.
    """
    path = _db(tmp_path, rows=60, start_ts=1_800_000_000)
    # Push a small tail far into the future, creating a gap before it.
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE samples SET ts = ts + 7200 WHERE ts > ?", (1_800_000_000 + 55 * 30,))
    conn.commit()
    conn.close()

    captured = {}

    class _Detector(_StubDetector):
        def detect(self, df, feature_columns):
            captured["rows"] = len(df)
            return super().detect(df, feature_columns)

    monkeypatch.setattr(
        engine, "load_model_artifact",
        lambda p: (_Detector([False] * 200), _ScalerPassthrough(), ["cpu_pct"]),
    )
    monkeypatch.setattr(engine, "model_status", lambda p: engine.ModelStatus(exists=True))

    engine.run_real_rca(path, tmp_path / "m.pt", hours=999)

    # The leading 56-row segment, not the 4-row tail.
    assert captured["rows"] > 40


def test_run_real_rca_reports_progress_through_every_stage(tmp_path, monkeypatch):
    """Causal inference dominates the runtime and used to run silently.

    The UI jumped from 10% to 75% across one blocking call, so a slow analysis
    was indistinguishable from a hung one.
    """
    path = _db(tmp_path, rows=60)

    monkeypatch.setattr(
        engine, "load_model_artifact",
        lambda p: (_StubDetector([True] * 200), _ScalerPassthrough(), ["cpu_pct"]),
    )
    monkeypatch.setattr(engine, "model_status", lambda p: engine.ModelStatus(exists=True))
    monkeypatch.setattr(
        engine, "run_causal_inference",
        lambda *a, **k: {"causal_graph": None, "root_causes": [], "event_correlations": []},
    )
    monkeypatch.setattr(
        engine, "load_process_attribution", lambda *a, **k: pd.DataFrame(),
    )

    seen = []
    engine.run_real_rca(path, tmp_path / "m.pt", hours=999, progress=lambda pct, msg: seen.append((pct, msg)))

    percentages = [pct for pct, _ in seen]
    assert percentages == sorted(percentages)                  # never goes backwards
    assert percentages[0] >= 10 and percentages[-1] >= 85      # spans model load to ranking
    assert len(percentages) >= 5                               # the causal stage is not silent
    assert all(message.strip() for _, message in seen)


def test_run_real_rca_still_runs_without_a_progress_callback(tmp_path, monkeypatch):
    """The CLI entry point passes no callback."""
    path = _db(tmp_path, rows=60)
    monkeypatch.setattr(
        engine, "load_model_artifact",
        lambda p: (_StubDetector([False] * 200), _ScalerPassthrough(), ["cpu_pct"]),
    )
    monkeypatch.setattr(engine, "model_status", lambda p: engine.ModelStatus(exists=True))

    assert engine.run_real_rca(path, tmp_path / "m.pt", hours=999)["active_anomalies"] == []


def test_detect_incidents_hides_windows_rca_cannot_analyse(tmp_path, monkeypatch):
    """Every offered incident must survive being selected.

    Windows events are retained for a year while samples exist only while the
    collector ran, so an event fault can name a window holding no telemetry at
    all. Those were listed and then failed the instant they were chosen, with
    "no uninterrupted model-length segment" -- which reads as a broken
    analysis rather than an unanalysable window.
    """
    path = _db(tmp_path, rows=120, start_ts=1_800_000_000)
    covered = pd.Timestamp(1_800_000_000 + 60 * 30, unit="s", tz="UTC")
    uncovered = pd.Timestamp(1_800_000_000, unit="s", tz="UTC") - pd.Timedelta(days=30)

    events = pd.DataFrame({
        "timestamp": [covered, uncovered],
        "event_id": [41, 41],
        "provider": ["Microsoft-Windows-Kernel-Power"] * 2,
        "level": [1, 1],
    })
    monkeypatch.setattr(engine, "load_events", lambda p: events)
    # No model, so the detector path is skipped and the default window applies.
    monkeypatch.setattr(engine, "model_status", lambda p: engine.ModelStatus(exists=False))

    incidents = engine.detect_incidents(path, tmp_path / "absent.pt", lookback_hours=24 * 365)

    starts = [incident.start for incident in incidents]
    assert any(start <= covered <= end for start, end in ((i.start, i.end) for i in incidents)), \
        "the event inside collected telemetry should still be offered"
    assert all(start > uncovered for start in starts), \
        "an event from before collection began has no telemetry to analyse"
