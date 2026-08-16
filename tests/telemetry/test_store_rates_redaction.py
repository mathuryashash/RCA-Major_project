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


class _Exploding:
    """A process whose attributes cannot be read at all."""

    pid = 4242

    def as_dict(self, attrs):
        raise MemoryError("proc_info failed")


class _Fine:
    pid = 7

    def as_dict(self, attrs):
        return {
            "name": "ok.exe",
            "create_time": 2.0,
            "cpu_times": type("T", (), {"user": 1.0, "system": 0.5})(),
            "memory_info": type("M", (), {"rss": 100})(),
            "io_counters": None,
        }


def test_process_sampling_survives_a_non_psutil_exception(monkeypatch):
    """psutil's Windows backend raises MemoryError, not psutil.Error.

    A narrow (psutil.Error, OSError, KeyError) guard let it escape and kill
    the collector; it then stayed dead until the next logon, and the baseline
    silently stopped growing.
    """
    from telemetry import sampler

    monkeypatch.setattr(sampler.psutil, "process_iter",
                        lambda *a, **k: iter([_Exploding(), _Fine()]))

    s = sampler.ProcessSampler()
    s.sample(top_n=5, elapsed_s=None)          # establish a baseline
    rows = s.sample(top_n=5, elapsed_s=30.0)   # must not raise
    assert [row["name"] for row in rows] == ["ok.exe"]


def test_process_sampling_survives_a_failure_raised_by_the_iterator(monkeypatch):
    """The real crash came from the `for` statement, not the loop body.

    psutil.process_iter(attrs=[...]) calls as_dict() *inside the generator*,
    so an unreadable process raises before any `except` in the body can see
    it. The previous test faked a list and could never reproduce that, which
    is why it passed while the packaged collector died with MemoryError at
    telemetry/sampler.py line 199 -- taking the already-collected system
    sample down with it.
    """
    from telemetry import sampler

    calls = []

    def recording_iter(*args, **kwargs):
        calls.append((args, kwargs))
        # Reproduce the eager form: asking for attrs makes psutil read every
        # process from inside the generator, which is where it exploded.
        if args or kwargs:
            raise MemoryError("proc_info failed inside process_iter")
        return iter([_Exploding(), _Fine()])

    monkeypatch.setattr(sampler.psutil, "process_iter", recording_iter)

    s = sampler.ProcessSampler()
    s.sample(top_n=5, elapsed_s=None)
    rows = s.sample(top_n=5, elapsed_s=30.0)

    assert calls, "process_iter was never called"
    assert all(not args and not kwargs for args, kwargs in calls), (
        "process_iter must be called bare so as_dict() runs inside the guard"
    )
    assert [row["name"] for row in rows] == ["ok.exe"]


def test_a_corrupt_database_is_moved_aside_and_replaced(tmp_path):
    """An unclean shutdown must not leave the app opening to a traceback."""
    import sqlite3

    from telemetry import store

    path = tmp_path / "telemetry.db"
    path.write_bytes(b"this is not a database, it is 40 bytes of junk")

    conn = store.connect(path)                 # must not raise
    store.init_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0

    moved = list(tmp_path.glob("telemetry.db.corrupt-*"))
    assert len(moved) == 1, "the damaged file must be kept, not deleted"
    assert moved[0].read_bytes().startswith(b"this is not a database")


def test_a_locked_database_is_never_quarantined(tmp_path):
    """The dangerous failure mode: OperationalError subclasses DatabaseError.

    "database is locked" and "file is not a database" arrive as the same
    exception family. Treating contention as damage would move a healthy
    database aside whenever the collector held a write lock -- destroying the
    user's history to fix a crash that was not happening.
    """
    import sqlite3

    from telemetry import store

    path = tmp_path / "telemetry.db"
    good = store.connect(path)
    store.init_schema(good)
    good.execute("INSERT INTO meta (key, value) VALUES ('canary', '1')")

    assert store.is_corruption(sqlite3.OperationalError("database is locked")) is False
    assert store.is_corruption(sqlite3.OperationalError("unable to open database file")) is False
    assert store.is_corruption(sqlite3.DatabaseError("file is not a database")) is True

    # And the real file survives a reopen while the first handle is still held.
    store.connect(path)
    assert not list(tmp_path.glob("*.corrupt-*")), "a live database was quarantined"
    assert good.execute("SELECT value FROM meta WHERE key='canary'").fetchone()[0] == "1"


def _fill(conn, table, first_ts, count, step=30):
    """Insert `count` rows of bulk so the database has something to reclaim."""
    if table == "samples":
        conn.executemany(
            "INSERT INTO samples (ts, foreground_app) VALUES (?, ?)",
            [(first_ts + i * step, "x" * 200) for i in range(count)],
        )
    else:
        conn.executemany(
            "INSERT INTO proc_samples (ts, pid, create_time, name) VALUES (?, ?, ?, ?)",
            [(first_ts + i * step, i, 1.0, "y" * 200) for i in range(count)],
        )


def test_metric_history_expires_like_everything_else(tmp_path):
    """`samples` had no retention while every other table did.

    It was the largest permanent contributor to a database measured growing
    3.33 MB/day with no ceiling.
    """
    from telemetry import store

    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    now = 1_800_000_000
    _fill(conn, "samples", now - 400 * 86400, 50)      # older than a year
    _fill(conn, "samples", now - 10 * 86400, 50)       # recent

    removed = store.purge_samples(conn, now - 365 * 86400)

    assert removed == 50
    assert store.sample_count(conn) == 50, "recent history must survive"


def test_vacuum_actually_returns_space_to_the_filesystem(tmp_path):
    """Deleting rows frees pages inside the file, not on the disk.

    Without VACUUM the retention rules delete hundreds of thousands of rows
    and the file does not shrink by one byte, so the purge looks like it did
    nothing. This is the test that would fail if reclaim() were removed.
    """
    from telemetry import store

    path = tmp_path / "t.db"
    conn = store.connect(path)
    store.init_schema(conn)
    _fill(conn, "proc_samples", 1_000_000, 40_000)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = path.stat().st_size

    store.purge_proc_samples(conn, 2_000_000)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    after_delete = path.stat().st_size
    assert after_delete >= before * 0.9, "deleting alone should not shrink the file"
    assert store.reclaimable_bytes(conn) > 0

    freed = store.reclaim(conn)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    after_vacuum = path.stat().st_size

    assert freed > 0
    assert after_vacuum < after_delete * 0.5, (
        f"file did not shrink: {after_delete} -> {after_vacuum}"
    )


def test_vacuum_is_skipped_when_there_is_little_to_reclaim(tmp_path):
    """A full rewrite every day for a few kilobytes is not worth the churn."""
    from telemetry import store

    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    _fill(conn, "samples", 1_000_000, 20)

    assert store.reclaim(conn) == 0


def test_vacuum_refuses_when_the_disk_cannot_hold_the_copy(tmp_path, monkeypatch):
    """VACUUM builds a complete second copy before swapping it in.

    Starting one without room is how a cleanup task becomes the outage it was
    meant to prevent -- the same rule the memory fault harness had to learn.
    """
    import shutil

    from telemetry import store

    path = tmp_path / "t.db"
    conn = store.connect(path)
    store.init_schema(conn)
    _fill(conn, "proc_samples", 1_000_000, 40_000)
    store.purge_proc_samples(conn, 2_000_000)
    assert store.reclaimable_bytes(conn) > 0        # would otherwise vacuum

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda p: type("U", (), {"total": 0, "used": 0, "free": 1024})(),
    )

    assert store.reclaim(conn) == 0, "must not vacuum onto a full disk"
    assert store.reclaimable_bytes(conn) > 0, "space stays on the free list"
