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


def test_gauge_decrease_is_signed_not_discarded():
    """Swap in use falls as pages are freed.

    Treating it as a monotonic counter returns None on every decrease, and
    downstream cleaning drops those rows -- which silently destroyed most of
    the collected history.
    """
    from telemetry.rates import CounterTracker

    tracker = CounterTracker(signed={"swap_used_bytes"})
    tracker.tick({"swap_used_bytes": 1000.0, "disk_read_bytes": 100.0}, now=0.0)
    _elapsed, deltas = tracker.tick(
        {"swap_used_bytes": 600.0, "disk_read_bytes": 50.0}, now=30.0
    )

    assert deltas["swap_used_bytes"] == -400.0      # signed gauge: kept
    assert deltas["disk_read_bytes"] is None        # counter reset: unknowable


def test_signed_gauge_still_reports_increases():
    from telemetry.rates import CounterTracker

    tracker = CounterTracker(signed={"swap_used_bytes"})
    tracker.tick({"swap_used_bytes": 100.0}, now=0.0)
    _elapsed, deltas = tracker.tick({"swap_used_bytes": 250.0}, now=30.0)
    assert deltas["swap_used_bytes"] == 150.0


def test_gpu_sampling_recovers_after_the_context_goes_stale(monkeypatch):
    """An Optimus dGPU sleeping invalidates the whole NVML context.

    Without a re-init the first sample succeeds and every later one records
    NULL forever, which is exactly what happened on real hardware.
    """
    from telemetry import sampler

    class _FakeNvml:
        def __init__(self):
            self.calls = 0
            self.inits = 0

        def nvmlInit(self):
            self.inits += 1

        def nvmlShutdown(self):
            pass

        def nvmlDeviceGetHandleByIndex(self, index):
            self.calls += 1
            # Fail until the context has been re-initialised.
            if self.inits == 0:
                raise RuntimeError("NVMLError_Unknown(999)")
            return object()

        def nvmlDeviceGetUtilizationRates(self, handle):
            return type("U", (), {"gpu": 7})()

        def nvmlDeviceGetMemoryInfo(self, handle):
            return type("M", (), {"used": 1234})()

        def nvmlDeviceGetTemperature(self, handle, sensor):
            return 51

        NVML_TEMPERATURE_GPU = 0

    fake = _FakeNvml()
    monkeypatch.setattr(sampler, "pynvml", fake)
    monkeypatch.setattr(sampler, "_NVML_READY", True)

    result = sampler.sample_gpu()

    assert fake.inits == 1, "a stale context must trigger exactly one re-init"
    assert result["gpu_temp_c"] == 51.0
    assert result["gpu_util_pct"] == 7.0


def test_gpu_sampling_returns_nulls_without_nvml(monkeypatch):
    from telemetry import sampler

    monkeypatch.setattr(sampler, "_NVML_READY", False)
    assert sampler.sample_gpu() == {
        "gpu_util_pct": None, "gpu_mem_used_bytes": None, "gpu_temp_c": None,
    }
