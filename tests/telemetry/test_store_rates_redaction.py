import sqlite3

from telemetry import config, sampler, store
from telemetry.rates import CounterTracker
from telemetry.redaction import redact


def connection(tmp_path):
    conn = store.connect(tmp_path / "telemetry.db")
    store.init_schema(conn)
    return conn


def test_schema_wal_meta_and_channel_scoped_events(tmp_path):
    conn = connection(tmp_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"samples", "proc_samples", "events", "collection_gaps", "meta"} <= tables
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    store.set_meta(conn, "x", "one")
    assert store.get_meta(conn, "x") == "one"
    conn.execute("INSERT INTO events(ts, channel, record_id) VALUES (1, 'System', 1)")
    conn.execute("INSERT INTO events(ts, channel, record_id) VALUES (1, 'Application', 1)")
    try:
        conn.execute("INSERT INTO events(ts, channel, record_id) VALUES (2, 'System', 1)")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("same record ID must be unique within a channel")


def test_rates_are_monotonic_and_reset_safe():
    tracker = CounterTracker()
    assert tracker.tick({"bytes": 10}, now=1.0) == (None, {"bytes": None})
    elapsed, deltas = tracker.tick({"bytes": 70}, now=3.5)
    assert (elapsed, deltas["bytes"]) == (2500, 60)
    _, deltas = tracker.tick({"bytes": 1}, now=4.5)
    assert deltas["bytes"] is None
    tracker.reset()
    assert tracker.tick({"bytes": 2}, now=5.0)[0] is None


def test_sample_schema_and_gap_detection(tmp_path):
    conn = connection(tmp_path)
    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, None, {name: None for name in raw["counters"]})
    assert set(row) == set(sampler.SAMPLE_COLUMNS)
    assert row["disk_read_bps"] is None
    for timestamp in (0, 30, 90):
        store.insert_sample(conn, timestamp, row)
    assert store.find_gaps(conn) == [(30, 90)]
    assert config.gap_threshold_s() == 45


def test_redaction_removes_known_sensitive_forms():
    value = redact(r"C:\Users\alice\secret.txt \\server\share\x https://example.com/a a@b.test", "alice")
    assert "alice" not in value and "server" not in value and "example.com" not in value and "a@b.test" not in value
