# Telemetry Collection Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a headless collector that records real laptop telemetry — psutil system metrics, per-process samples, and Windows Event Log entries — into a local SQLite store, so the three-day baseline clock starts immediately.

**Architecture:** A separate headless process samples system metrics every 30 s and per-process metrics every 5 min (bursting to 30 s under load), writing to `%LOCALAPPDATA%\RCA\telemetry.db` in WAL mode. Event Log channels are polled every 5 min using a per-channel record-ID watermark, advanced in the same transaction as the event insert. The desktop app is a later consumer of this store and is not touched by this plan.

**Tech Stack:** Python 3.13, psutil 6.0, pywin32, SQLite (stdlib `sqlite3`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-27-real-telemetry-rca-design.md`

## Global Constraints

- **No synthetic data anywhere.** No `np.random`, `torch.randn`, or fabricated
  values in any code path, including demo blocks and test fixtures. Tests use
  live psutil readings or literal strings, never generated telemetry.
- **Cadence:** system 30 s, per-process 300 s, per-process burst 30 s, Event Log
  poll 300 s.
- **Burst thresholds:** `cpu_pct > 80`, `mem_pct > 85`, `disk_busy_pct > 80`.
- **Gap threshold:** `1.5 x cadence` = 45 s. Defined relative to cadence, never
  hardcoded as 45.
- **Top-N processes:** 15 by RSS union 15 by CPU delta (~28 rows/tick measured).
- **Retention:** `proc_samples` 30 days, `events` 365 days, `samples` forever.
- **Process identity is `(pid, create_time)`**, never `pid` alone.
- **Rates use measured monotonic elapsed time**, never assumed cadence.
- **Privacy:** never capture window titles. Event message text is opt-in and
  defaults to off. No network access from the collector.
- **Platform:** Windows only. Non-Windows must skip cleanly, not crash.
- Python version floor: 3.10 (uses `X | None` syntax).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/telemetry/config.py` | Paths, cadences, thresholds. No logic. |
| `src/telemetry/store.py` | SQLite schema, connection, all read/write APIs, purge. |
| `src/telemetry/rates.py` | `CounterTracker` — monotonic elapsed + counter deltas. Pure. |
| `src/telemetry/sampler.py` | System snapshot and process snapshot from psutil. |
| `src/telemetry/redaction.py` | Text redaction patterns. Pure, no I/O. |
| `src/telemetry/eventlog.py` | Event Log query, watermark, gap recording. |
| `src/telemetry/collector.py` | The loop: scheduling, burst logic, mutex, consent gate. |
| `src/telemetry/schedule.py` | Task Scheduler register/unregister via `schtasks`. |
| `src/telemetry/__main__.py` | CLI entry point. |

Tests mirror this one-to-one under `tests/telemetry/`.

---

### Task 1: Config and store schema

**Files:**
- Create: `src/telemetry/__init__.py`
- Create: `src/telemetry/config.py`
- Create: `src/telemetry/logsetup.py`
- Create: `src/telemetry/store.py`
- Create: `tests/conftest.py`
- Create: `tests/telemetry/__init__.py`
- Test: `tests/telemetry/test_store.py`

**Note:** this repo has no `pyproject.toml` and is not installed; existing tests
insert `../src` into `sys.path` by hand at the top of each file. A single
`tests/conftest.py` does it once for every test in the suite, so no test file
below needs path boilerplate.

**Interfaces:**
- Consumes: nothing.
- Produces: `logsetup.get_logger(name)`, `config.SYSTEM_CADENCE_S`, `config.PROCESS_CADENCE_S`,
  `config.PROCESS_BURST_CADENCE_S`, `config.EVENT_POLL_S`,
  `config.GAP_FACTOR`, `config.gap_threshold_s()`, `config.TOP_N`,
  `config.BURST_CPU_PCT`, `config.BURST_MEM_PCT`, `config.BURST_DISK_BUSY_PCT`,
  `config.PROC_RETENTION_DAYS`, `config.EVENT_RETENTION_DAYS`,
  `config.EVENT_CHANNELS`, `config.db_path()`, `config.SCHEMA_VERSION`;
  `store.connect(path)`, `store.init_schema(conn)`, `store.get_meta(conn, key, default=None)`,
  `store.set_meta(conn, key, value)`.

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:

```python
"""Make `src/` importable for the whole suite.

This repo is not pip-installed and has no pyproject.toml, so without this every
test file would need its own sys.path insertion.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
```

Create `tests/telemetry/__init__.py` as an empty file, then
`tests/telemetry/test_store.py`:

```python
import sqlite3

from telemetry import config, store


def test_init_schema_creates_all_tables(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    names = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"samples", "proc_samples", "events", "collection_gaps", "meta"} <= names


def test_init_schema_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.init_schema(conn)
    assert store.get_meta(conn, "schema_version") == str(config.SCHEMA_VERSION)


def test_wal_mode_enabled(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_meta_roundtrip_and_default(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    assert store.get_meta(conn, "absent", "fallback") == "fallback"
    store.set_meta(conn, "k", "v1")
    store.set_meta(conn, "k", "v2")
    assert store.get_meta(conn, "k") == "v2"


def test_events_unique_is_channel_scoped(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    ins = (
        "INSERT INTO events(ts, channel, record_id, provider, event_id, level)"
        " VALUES (?,?,?,?,?,?)"
    )
    conn.execute(ins, (1, "System", 1, "p", 41, "Critical"))
    # same record_id on a different channel must be allowed
    conn.execute(ins, (1, "Application", 1, "p", 1000, "Error"))
    # same record_id on the same channel must not
    try:
        conn.execute(ins, (2, "System", 1, "p", 41, "Critical"))
        raise AssertionError("expected IntegrityError")
    except sqlite3.IntegrityError:
        pass


def test_gap_threshold_derives_from_cadence():
    assert config.gap_threshold_s() == config.SYSTEM_CADENCE_S * config.GAP_FACTOR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry'`

- [ ] **Step 3: Write the implementation**

`src/telemetry/__init__.py`:

```python
"""Real laptop telemetry collection. No synthetic data is generated here."""
```

`src/telemetry/config.py`:

```python
"""Collection constants. Values here are the spec's, not tunables to guess at."""
import os
from pathlib import Path

SCHEMA_VERSION = 1

SYSTEM_CADENCE_S = 30
PROCESS_CADENCE_S = 300
PROCESS_BURST_CADENCE_S = 30
EVENT_POLL_S = 300

# A gap is any interval longer than 1.5x the sampling cadence. Expressed as a
# factor so it stays correct if the cadence changes: one dropped tick at 30s
# yields a 60s interval and must count as a gap.
GAP_FACTOR = 1.5

BURST_CPU_PCT = 80.0
BURST_MEM_PCT = 85.0
BURST_DISK_BUSY_PCT = 80.0

TOP_N = 15

PROC_RETENTION_DAYS = 30
EVENT_RETENTION_DAYS = 365

EVENT_CHANNELS = ("System", "Application")

MUTEX_NAME = "Local\\RCATelemetryCollector"
TASK_NAME = "RCA Telemetry Collector"

LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 2


def gap_threshold_s() -> float:
    return SYSTEM_CADENCE_S * GAP_FACTOR


def app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / "RCA" if base else Path.home() / ".rca"


def db_path() -> Path:
    return app_dir() / "telemetry.db"


def log_path() -> Path:
    return app_dir() / "collector.log"


def stop_flag_path() -> Path:
    return app_dir() / "stop.flag"
```

`src/telemetry/logsetup.py`:

```python
"""Local rotating log.

The collector runs headless under Task Scheduler with nowhere to print, so a
swallowed exception is an invisible one. Disk-full, database-locked and Event
Log failures all land here.
"""
import logging
from logging.handlers import RotatingFileHandler

from . import config

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if not _configured:
        config.app_dir().mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            config.log_path(),
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root = logging.getLogger("telemetry")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _configured = True
    return logger
```

`src/telemetry/store.py`:

```python
"""SQLite persistence for collected telemetry."""
import sqlite3
from pathlib import Path

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts                     INTEGER PRIMARY KEY,
    elapsed_ms             INTEGER,
    cpu_pct                REAL,
    cpu_pct_max_core       REAL,
    cpu_freq_mhz           REAL,
    cpu_freq_ratio         REAL,
    mem_pct                REAL,
    mem_available_mb       REAL,
    swap_pct               REAL,
    swap_used_bytes        INTEGER,
    swap_used_delta        INTEGER,
    disk_read_bps          REAL,
    disk_write_bps         REAL,
    disk_busy_pct          REAL,
    disk_free_pct          REAL,
    net_sent_bps           REAL,
    net_recv_bps           REAL,
    process_count          INTEGER,
    battery_pct            REAL,
    battery_drain_rate     REAL,
    power_plugged          INTEGER,
    on_battery             INTEGER,
    user_idle_sec          REAL,
    foreground_app         TEXT,
    cpu_busy_s_delta       REAL,
    mem_used_bytes         INTEGER,
    disk_read_bytes_delta  INTEGER,
    disk_write_bytes_delta INTEGER
);

CREATE TABLE IF NOT EXISTS proc_samples (
    ts               INTEGER NOT NULL,
    pid              INTEGER NOT NULL,
    create_time      REAL    NOT NULL,
    name             TEXT,
    cpu_pct          REAL,
    cpu_time_delta_s REAL,
    rss              INTEGER,
    io_read_delta    INTEGER,
    io_write_delta   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_proc_ts ON proc_samples(ts);

CREATE TABLE IF NOT EXISTS events (
    ts               INTEGER NOT NULL,
    channel          TEXT    NOT NULL,
    record_id        INTEGER NOT NULL,
    provider         TEXT,
    event_id         INTEGER,
    level            TEXT,
    message_redacted TEXT,
    UNIQUE(channel, record_id)
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS collection_gaps (
    channel     TEXT    NOT NULL,
    start_ts    INTEGER,
    end_ts      INTEGER,
    detected_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    set_meta(conn, "schema_version", str(config.SCHEMA_VERSION))


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/__init__.py src/telemetry/config.py src/telemetry/store.py tests/telemetry/
git commit -m "feat(telemetry): add SQLite store schema and collection config"
```

---

### Task 2: CounterTracker for monotonic rates

**Files:**
- Create: `src/telemetry/rates.py`
- Test: `tests/telemetry/test_rates.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rates.CounterTracker` with
  `tick(counters: dict[str, float], now: float | None = None) -> tuple[int | None, dict[str, float | None]]`
  returning `(elapsed_ms, deltas)`, and `reset() -> None`.
  `elapsed_ms` is `None` on the first tick after construction or reset.
  A delta is `None` when there is no previous value or the counter went
  backwards (a reset).

- [ ] **Step 1: Write the failing test**

```python
from telemetry.rates import CounterTracker


def test_first_tick_returns_none_elapsed_and_none_deltas():
    t = CounterTracker()
    elapsed, deltas = t.tick({"a": 100.0}, now=10.0)
    assert elapsed is None
    assert deltas == {"a": None}


def test_second_tick_returns_measured_elapsed_and_delta():
    t = CounterTracker()
    t.tick({"a": 100.0}, now=10.0)
    elapsed, deltas = t.tick({"a": 150.0}, now=12.5)
    assert elapsed == 2500
    assert deltas["a"] == 50.0


def test_elapsed_uses_measured_time_not_assumed_cadence():
    """A late tick must not be treated as if it arrived on schedule."""
    t = CounterTracker()
    t.tick({"bytes": 0.0}, now=0.0)
    _, on_time = t.tick({"bytes": 3000.0}, now=30.0)
    t.reset()
    t.tick({"bytes": 0.0}, now=0.0)
    late_elapsed, late = t.tick({"bytes": 3000.0}, now=90.0)
    assert on_time["bytes"] == late["bytes"] == 3000.0
    # same delta, but the caller divides by a 3x larger interval
    assert late_elapsed == 90000


def test_counter_reset_yields_none_not_negative():
    t = CounterTracker()
    t.tick({"a": 500.0}, now=0.0)
    _, deltas = t.tick({"a": 10.0}, now=1.0)
    assert deltas["a"] is None


def test_new_key_appearing_later_has_none_delta():
    t = CounterTracker()
    t.tick({"a": 1.0}, now=0.0)
    _, deltas = t.tick({"a": 2.0, "b": 99.0}, now=1.0)
    assert deltas["a"] == 1.0
    assert deltas["b"] is None


def test_reset_clears_history():
    t = CounterTracker()
    t.tick({"a": 1.0}, now=0.0)
    t.reset()
    elapsed, deltas = t.tick({"a": 2.0}, now=1.0)
    assert elapsed is None
    assert deltas["a"] is None


def test_zero_or_negative_interval_yields_none_deltas():
    t = CounterTracker()
    t.tick({"a": 1.0}, now=5.0)
    elapsed, deltas = t.tick({"a": 9.0}, now=5.0)
    assert elapsed == 0
    assert deltas["a"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_rates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.rates'`

- [ ] **Step 3: Write the implementation**

```python
"""Counter differencing against measured monotonic time.

Rates must never be computed against the assumed cadence: a delayed or slow
tick would inflate every rate on that row.
"""
import time


class CounterTracker:
    def __init__(self) -> None:
        self._last_mono: float | None = None
        self._last: dict[str, float] = {}

    def reset(self) -> None:
        self._last_mono = None
        self._last = {}

    def tick(
        self, counters: dict[str, float], now: float | None = None
    ) -> tuple[int | None, dict[str, float | None]]:
        if now is None:
            now = time.monotonic()

        if self._last_mono is None:
            self._last_mono = now
            self._last = dict(counters)
            return None, {k: None for k in counters}

        elapsed_ms = int(round((now - self._last_mono) * 1000))
        deltas: dict[str, float | None] = {}
        for key, value in counters.items():
            prev = self._last.get(key)
            if prev is None or value < prev or elapsed_ms <= 0:
                deltas[key] = None
            else:
                deltas[key] = value - prev

        self._last_mono = now
        self._last = dict(counters)
        return elapsed_ms, deltas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_rates.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/rates.py tests/telemetry/test_rates.py
git commit -m "feat(telemetry): add CounterTracker for monotonic rate calculation"
```

---

### Task 3: System sampler

**Files:**
- Create: `src/telemetry/sampler.py`
- Test: `tests/telemetry/test_sampler.py`

**Interfaces:**
- Consumes: `rates.CounterTracker`.
- Produces: `sampler.SAMPLE_COLUMNS` (tuple of str, the `samples` columns
  excluding `ts`), `sampler.sample_system_raw() -> dict`,
  `sampler.build_sample_row(raw, elapsed_ms, deltas) -> dict`,
  `sampler.user_idle_sec() -> float | None`,
  `sampler.foreground_app() -> str | None`.
  `sample_system_raw()` returns keys `gauges` (dict) and `counters` (dict);
  `counters` is what gets fed to `CounterTracker.tick`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from telemetry import sampler, store
from telemetry.rates import CounterTracker


def test_raw_sample_has_gauges_and_counters():
    raw = sampler.sample_system_raw()
    assert set(raw) == {"gauges", "counters"}
    for key in ("cpu_pct", "mem_pct", "process_count"):
        assert key in raw["gauges"]
    for key in ("cpu_busy_s", "disk_read_bytes", "disk_write_bytes",
                "net_sent_bytes", "net_recv_bytes", "swap_used_bytes",
                "disk_busy_ms"):
        assert key in raw["counters"]


def test_first_row_has_null_rates_but_real_gauges():
    tracker = CounterTracker()
    raw = sampler.sample_system_raw()
    elapsed, deltas = tracker.tick(raw["counters"])
    row = sampler.build_sample_row(raw, elapsed, deltas)
    assert row["elapsed_ms"] is None
    assert row["disk_read_bps"] is None
    assert row["cpu_busy_s_delta"] is None
    assert row["mem_pct"] is not None          # gauge, always available
    assert row["process_count"] > 0


def test_second_row_has_computed_rates():
    tracker = CounterTracker()
    first = sampler.sample_system_raw()
    tracker.tick(first["counters"])
    second = sampler.sample_system_raw()
    elapsed, deltas = tracker.tick(second["counters"])
    row = sampler.build_sample_row(second, elapsed, deltas)
    assert row["elapsed_ms"] >= 0
    assert row["disk_read_bps"] is None or row["disk_read_bps"] >= 0
    assert row["net_recv_bps"] is None or row["net_recv_bps"] >= 0


def test_row_keys_match_store_columns_exactly(tmp_path):
    """A mismatch here is a silent INSERT failure at runtime."""
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    cols.discard("ts")
    assert set(sampler.SAMPLE_COLUMNS) == cols

    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, None, {k: None for k in raw["counters"]})
    assert set(row) == cols


def test_foreground_app_is_never_a_window_title():
    """Privacy constraint: executable name only."""
    name = sampler.foreground_app()
    if name is not None:
        assert name.lower().endswith(".exe") or "." not in name
        assert len(name) < 128


def test_user_idle_sec_is_non_negative():
    idle = sampler.user_idle_sec()
    if idle is not None:
        assert idle >= 0.0


def test_disk_busy_pct_is_bounded():
    tracker = CounterTracker()
    tracker.tick(sampler.sample_system_raw()["counters"])
    raw = sampler.sample_system_raw()
    elapsed, deltas = tracker.tick(raw["counters"])
    row = sampler.build_sample_row(raw, elapsed, deltas)
    if row["disk_busy_pct"] is not None:
        assert 0.0 <= row["disk_busy_pct"] <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.sampler'`

- [ ] **Step 3: Write the implementation**

```python
"""psutil sampling.

Field choices were verified against psutil 6.0.0 on Windows:
 - swap sin/sout are always 0 on Windows, so paging is tracked via used bytes
 - there is no cheap system-wide thread count (11.3 s), so it is not collected
 - sensors_battery() returns None on machines without a battery
 - cpu_freq().min reads 0.0 on Windows; only .max is a usable denominator
"""
import ctypes
import os
from ctypes import wintypes

import psutil

SAMPLE_COLUMNS = (
    "elapsed_ms",
    "cpu_pct", "cpu_pct_max_core", "cpu_freq_mhz", "cpu_freq_ratio",
    "mem_pct", "mem_available_mb",
    "swap_pct", "swap_used_bytes", "swap_used_delta",
    "disk_read_bps", "disk_write_bps", "disk_busy_pct", "disk_free_pct",
    "net_sent_bps", "net_recv_bps",
    "process_count",
    "battery_pct", "battery_drain_rate", "power_plugged",
    "on_battery", "user_idle_sec", "foreground_app",
    "cpu_busy_s_delta", "mem_used_bytes",
    "disk_read_bytes_delta", "disk_write_bytes_delta",
)

_IS_WINDOWS = hasattr(ctypes, "windll")


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def user_idle_sec() -> float | None:
    """Seconds since last keyboard/mouse input. Duration only, never content."""
    if not _IS_WINDOWS:
        return None
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    # GetTickCount and dwTime are both 32-bit and wrap together at ~49.7 days;
    # a wrap shows up as a negative difference, which we clamp rather than store.
    delta_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0.0, delta_ms / 1000.0)


def foreground_app() -> str | None:
    """Executable name of the foreground window's process.

    Deliberately never reads the window title: titles leak document names,
    URLs and message contents, and the executable name is all attribution needs.
    """
    if not _IS_WINDOWS:
        return None
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return None
    try:
        return psutil.Process(pid.value).name()
    except (psutil.Error, OSError):
        return None


def sample_system_raw() -> dict:
    """Cheap system-wide read. Measured at ~8 ms."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    ct = psutil.cpu_times()
    disk = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    per_core = psutil.cpu_percent(percpu=True)

    try:
        freq = psutil.cpu_freq()
    except (OSError, NotImplementedError):
        freq = None
    try:
        battery = psutil.sensors_battery()
    except (OSError, NotImplementedError):
        battery = None
    # Explicitly the system drive: disk_partitions()[0] is not reliably C:.
    try:
        system_drive = os.environ.get("SystemDrive", "C:") + "\\"
        disk_free_pct = 100.0 - psutil.disk_usage(system_drive).percent
    except OSError:
        disk_free_pct = None

    freq_mhz = freq.current if freq else None
    freq_ratio = None
    if freq and freq.max:
        freq_ratio = freq.current / freq.max

    gauges = {
        "cpu_pct": psutil.cpu_percent(),
        "cpu_pct_max_core": max(per_core) if per_core else None,
        "cpu_freq_mhz": freq_mhz,
        "cpu_freq_ratio": freq_ratio,
        "mem_pct": vm.percent,
        "mem_available_mb": vm.available / (1024 * 1024),
        "mem_used_bytes": vm.used,
        "swap_pct": sw.percent,
        "swap_used_bytes": sw.used,
        "disk_free_pct": disk_free_pct,
        "process_count": len(psutil.pids()),
        "battery_pct": battery.percent if battery else None,
        "power_plugged": int(battery.power_plugged) if battery else None,
        "on_battery": int(not battery.power_plugged) if battery else None,
        "user_idle_sec": user_idle_sec(),
        "foreground_app": foreground_app(),
    }

    counters = {
        "cpu_busy_s": ct.user + ct.system,
        "disk_read_bytes": float(disk.read_bytes) if disk else 0.0,
        "disk_write_bytes": float(disk.write_bytes) if disk else 0.0,
        "disk_busy_ms": float(disk.read_time + disk.write_time) if disk else 0.0,
        "net_sent_bytes": float(net.bytes_sent) if net else 0.0,
        "net_recv_bytes": float(net.bytes_recv) if net else 0.0,
        "swap_used_bytes": float(sw.used),
        "battery_pct_counter": float(battery.percent) if battery else 0.0,
    }
    return {"gauges": gauges, "counters": counters}


def _per_second(delta: float | None, elapsed_ms: int | None) -> float | None:
    if delta is None or not elapsed_ms:
        return None
    return delta * 1000.0 / elapsed_ms


def build_sample_row(
    raw: dict, elapsed_ms: int | None, deltas: dict[str, float | None]
) -> dict:
    """Assemble one `samples` row. Keys match SAMPLE_COLUMNS exactly."""
    g = raw["gauges"]

    busy_ms = deltas.get("disk_busy_ms")
    disk_busy_pct = None
    if busy_ms is not None and elapsed_ms:
        disk_busy_pct = min(100.0, max(0.0, busy_ms * 100.0 / elapsed_ms))

    # Battery drain is only meaningful while discharging.
    drain = None
    batt_delta = deltas.get("battery_pct_counter")
    if batt_delta is not None and elapsed_ms and not g.get("power_plugged"):
        drain = max(0.0, -batt_delta) * 3600_000.0 / elapsed_ms

    return {
        "elapsed_ms": elapsed_ms,
        "cpu_pct": g["cpu_pct"],
        "cpu_pct_max_core": g["cpu_pct_max_core"],
        "cpu_freq_mhz": g["cpu_freq_mhz"],
        "cpu_freq_ratio": g["cpu_freq_ratio"],
        "mem_pct": g["mem_pct"],
        "mem_available_mb": g["mem_available_mb"],
        "swap_pct": g["swap_pct"],
        "swap_used_bytes": g["swap_used_bytes"],
        "swap_used_delta": deltas.get("swap_used_bytes"),
        "disk_read_bps": _per_second(deltas.get("disk_read_bytes"), elapsed_ms),
        "disk_write_bps": _per_second(deltas.get("disk_write_bytes"), elapsed_ms),
        "disk_busy_pct": disk_busy_pct,
        "disk_free_pct": g["disk_free_pct"],
        "net_sent_bps": _per_second(deltas.get("net_sent_bytes"), elapsed_ms),
        "net_recv_bps": _per_second(deltas.get("net_recv_bytes"), elapsed_ms),
        "process_count": g["process_count"],
        "battery_pct": g["battery_pct"],
        "battery_drain_rate": drain,
        "power_plugged": g["power_plugged"],
        "on_battery": g["on_battery"],
        "user_idle_sec": g["user_idle_sec"],
        "foreground_app": g["foreground_app"],
        "cpu_busy_s_delta": deltas.get("cpu_busy_s"),
        "mem_used_bytes": g["mem_used_bytes"],
        "disk_read_bytes_delta": deltas.get("disk_read_bytes"),
        "disk_write_bytes_delta": deltas.get("disk_write_bytes"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_sampler.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/sampler.py tests/telemetry/test_sampler.py
git commit -m "feat(telemetry): add system sampler with verified psutil fields"
```

---

### Task 4: Process sampler

**Files:**
- Modify: `src/telemetry/sampler.py` (append `ProcessSampler`)
- Test: `tests/telemetry/test_process_sampler.py`

**Interfaces:**
- Consumes: `config.TOP_N`.
- Produces: `sampler.ProcessSampler` with
  `sample(top_n: int, elapsed_s: float | None) -> list[dict]`.
  Each dict has keys `pid`, `create_time`, `name`, `cpu_pct`,
  `cpu_time_delta_s`, `rss`, `io_read_delta`, `io_write_delta`.
  Returns `[]` on the first call (no previous counters to difference).

- [ ] **Step 1: Write the failing test**

```python
import psutil
import pytest

from telemetry.sampler import ProcessSampler


def test_first_sample_is_empty_no_baseline():
    s = ProcessSampler()
    assert s.sample(top_n=5, elapsed_s=1.0) == []


def test_second_sample_returns_rows_with_expected_keys():
    s = ProcessSampler()
    s.sample(top_n=5, elapsed_s=None)
    rows = s.sample(top_n=5, elapsed_s=1.0)
    assert rows, "expected at least one process row"
    expected = {"pid", "create_time", "name", "cpu_pct",
                "cpu_time_delta_s", "rss", "io_read_delta", "io_write_delta"}
    for row in rows:
        assert set(row) == expected


def test_row_count_bounded_by_two_times_top_n():
    s = ProcessSampler()
    s.sample(top_n=15, elapsed_s=None)
    rows = s.sample(top_n=15, elapsed_s=1.0)
    assert len(rows) <= 30


def test_identity_is_pid_and_create_time():
    s = ProcessSampler()
    s.sample(top_n=15, elapsed_s=None)
    rows = s.sample(top_n=15, elapsed_s=1.0)
    keys = [(r["pid"], r["create_time"]) for r in rows]
    assert len(keys) == len(set(keys)), "identity must be unique per row"


def test_pid_reuse_does_not_fabricate_a_delta():
    """A recycled PID with a new create_time must start from zero, not from
    the dead process's counters."""
    s = ProcessSampler()
    s._prev = {(4242, 100.0): {"cpu_time_s": 999.0, "io_read": 10**9, "io_write": 10**9}}
    delta = s._delta_for(
        key=(4242, 200.0), cpu_time_s=1.0, io_read=5, io_write=5
    )
    assert delta["cpu_time_delta_s"] == 0.0
    assert delta["io_read_delta"] == 0
    assert delta["io_write_delta"] == 0


def test_counter_reset_clamps_to_zero_not_negative():
    s = ProcessSampler()
    key = (1234, 50.0)
    s._prev = {key: {"cpu_time_s": 100.0, "io_read": 500, "io_write": 500}}
    delta = s._delta_for(key=key, cpu_time_s=1.0, io_read=1, io_write=1)
    assert delta["cpu_time_delta_s"] == 0.0
    assert delta["io_read_delta"] == 0
    assert delta["io_write_delta"] == 0


def test_exited_processes_are_dropped_from_state():
    s = ProcessSampler()
    s.sample(top_n=5, elapsed_s=None)
    s._prev[(999999, 1.0)] = {"cpu_time_s": 1.0, "io_read": 0, "io_write": 0}
    s.sample(top_n=5, elapsed_s=1.0)
    assert (999999, 1.0) not in s._prev


def test_cpu_pct_is_non_negative():
    s = ProcessSampler()
    s.sample(top_n=15, elapsed_s=None)
    rows = s.sample(top_n=15, elapsed_s=1.0)
    for row in rows:
        assert row["cpu_pct"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_process_sampler.py -v`
Expected: FAIL with `ImportError: cannot import name 'ProcessSampler'`

- [ ] **Step 3: Write the implementation**

Append to `src/telemetry/sampler.py`:

```python
class ProcessSampler:
    """Per-process sampling with (pid, create_time) identity.

    Costs ~900 ms because any per-process attribute forces an OpenProcess per
    PID, so the collector runs this at 5 min rather than every tick.
    """

    def __init__(self) -> None:
        self._prev: dict[tuple[int, float], dict] = {}
        self._ncores = psutil.cpu_count() or 1

    def _delta_for(
        self, key: tuple[int, float], cpu_time_s: float, io_read: int, io_write: int
    ) -> dict:
        prev = self._prev.get(key)
        if prev is None:
            # Either a newly started process or a recycled PID whose
            # create_time differs. Both start from zero.
            return {"cpu_time_delta_s": 0.0, "io_read_delta": 0, "io_write_delta": 0}
        return {
            "cpu_time_delta_s": max(0.0, cpu_time_s - prev["cpu_time_s"]),
            "io_read_delta": max(0, io_read - prev["io_read"]),
            "io_write_delta": max(0, io_write - prev["io_write"]),
        }

    def sample(self, top_n: int, elapsed_s: float | None) -> list[dict]:
        current: dict[tuple[int, float], dict] = {}
        rows: list[dict] = []
        had_prev = bool(self._prev)

        attrs = ["name", "create_time", "cpu_times", "memory_info", "io_counters"]
        for proc in psutil.process_iter(attrs):
            info = proc.info
            create_time = info.get("create_time")
            mem = info.get("memory_info")
            times = info.get("cpu_times")
            if create_time is None or mem is None or times is None:
                continue

            key = (proc.pid, create_time)
            cpu_time_s = times.user + times.system
            io = info.get("io_counters")
            io_read = io.read_bytes if io else 0
            io_write = io.write_bytes if io else 0

            current[key] = {
                "cpu_time_s": cpu_time_s, "io_read": io_read, "io_write": io_write
            }
            if not had_prev:
                continue

            delta = self._delta_for(key, cpu_time_s, io_read, io_write)
            cpu_pct = 0.0
            if elapsed_s and elapsed_s > 0:
                cpu_pct = delta["cpu_time_delta_s"] * 100.0 / (elapsed_s * self._ncores)

            rows.append({
                "pid": proc.pid,
                "create_time": create_time,
                "name": info.get("name"),
                "cpu_pct": cpu_pct,
                "rss": mem.rss,
                **delta,
            })

        # Replacing rather than updating drops processes that have exited.
        self._prev = current

        if not rows:
            return []

        by_cpu = sorted(rows, key=lambda r: r["cpu_time_delta_s"], reverse=True)[:top_n]
        by_rss = sorted(rows, key=lambda r: r["rss"], reverse=True)[:top_n]
        chosen: dict[tuple[int, float], dict] = {}
        for row in by_cpu + by_rss:
            chosen[(row["pid"], row["create_time"])] = row
        return list(chosen.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_process_sampler.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/sampler.py tests/telemetry/test_process_sampler.py
git commit -m "feat(telemetry): add process sampler with pid-reuse-safe identity"
```

---

### Task 5: Store write and read APIs

**Files:**
- Modify: `src/telemetry/store.py` (append write/read functions)
- Test: `tests/telemetry/test_store_io.py`

**Interfaces:**
- Consumes: `sampler.SAMPLE_COLUMNS`, `config.gap_threshold_s()`.
- Produces: `store.insert_sample(conn, ts, row)`,
  `store.insert_proc_samples(conn, ts, rows)`,
  `store.find_gaps(conn, threshold_s) -> list[tuple[int, int]]`,
  `store.purge_proc_samples(conn, older_than_ts) -> int`,
  `store.purge_events(conn, older_than_ts) -> int`,
  `store.record_collection_gap(conn, channel, start_ts, end_ts, detected_at)`,
  `store.sample_count(conn) -> int`.

- [ ] **Step 1: Write the failing test**

```python
from telemetry import config, sampler, store


def _open(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    return conn


def test_insert_sample_roundtrip(tmp_path):
    conn = _open(tmp_path)
    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, 30000, {k: 1.0 for k in raw["counters"]})
    store.insert_sample(conn, ts=1000, row=row)
    assert store.sample_count(conn) == 1
    got = conn.execute("SELECT mem_pct FROM samples WHERE ts = 1000").fetchone()
    assert got[0] == row["mem_pct"]


def test_insert_sample_is_idempotent_on_same_ts(tmp_path):
    conn = _open(tmp_path)
    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, None, {k: None for k in raw["counters"]})
    store.insert_sample(conn, ts=5, row=row)
    store.insert_sample(conn, ts=5, row=row)
    assert store.sample_count(conn) == 1


def test_insert_proc_samples(tmp_path):
    conn = _open(tmp_path)
    rows = [{
        "pid": 1, "create_time": 2.0, "name": "a.exe", "cpu_pct": 1.0,
        "cpu_time_delta_s": 0.5, "rss": 100, "io_read_delta": 1, "io_write_delta": 2,
    }]
    store.insert_proc_samples(conn, ts=10, rows=rows)
    assert conn.execute("SELECT COUNT(*) FROM proc_samples").fetchone()[0] == 1


def test_find_gaps_detects_single_dropped_tick(tmp_path):
    """One dropped tick at 30s cadence is a 60s interval and IS a gap."""
    conn = _open(tmp_path)
    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, 30000, {k: 1.0 for k in raw["counters"]})
    for ts in (0, 30, 90, 120):        # 30->90 is a dropped tick
        store.insert_sample(conn, ts=ts, row=row)
    gaps = store.find_gaps(conn, config.gap_threshold_s())
    assert (30, 90) in gaps


def test_find_gaps_ignores_on_time_ticks(tmp_path):
    conn = _open(tmp_path)
    raw = sampler.sample_system_raw()
    row = sampler.build_sample_row(raw, 30000, {k: 1.0 for k in raw["counters"]})
    for ts in (0, 30, 60, 90):
        store.insert_sample(conn, ts=ts, row=row)
    assert store.find_gaps(conn, config.gap_threshold_s()) == []


def test_purge_proc_samples_respects_cutoff(tmp_path):
    conn = _open(tmp_path)
    rows = [{
        "pid": 1, "create_time": 2.0, "name": "a.exe", "cpu_pct": 1.0,
        "cpu_time_delta_s": 0.5, "rss": 100, "io_read_delta": 1, "io_write_delta": 2,
    }]
    store.insert_proc_samples(conn, ts=100, rows=rows)
    store.insert_proc_samples(conn, ts=900, rows=rows)
    assert store.purge_proc_samples(conn, older_than_ts=500) == 1
    remaining = conn.execute("SELECT ts FROM proc_samples").fetchall()
    assert [r[0] for r in remaining] == [900]


def test_record_collection_gap(tmp_path):
    conn = _open(tmp_path)
    store.record_collection_gap(conn, "System", 10, 20, detected_at=25)
    row = conn.execute(
        "SELECT channel, start_ts, end_ts, detected_at FROM collection_gaps"
    ).fetchone()
    assert row == ("System", 10, 20, 25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_store_io.py -v`
Expected: FAIL with `AttributeError: module 'telemetry.store' has no attribute 'insert_sample'`

- [ ] **Step 3: Write the implementation**

Add `from .sampler import SAMPLE_COLUMNS` to the **imports at the top** of
`src/telemetry/store.py` (alongside `from . import config`), then append the
rest below. `sampler` does not import `store`, so there is no cycle.

```python
# top of file, with the other imports:
#     from .sampler import SAMPLE_COLUMNS

_SAMPLE_INSERT = (
    "INSERT OR IGNORE INTO samples (ts, {cols}) VALUES (?, {marks})".format(
        cols=", ".join(SAMPLE_COLUMNS),
        marks=", ".join("?" for _ in SAMPLE_COLUMNS),
    )
)

_PROC_INSERT = (
    "INSERT INTO proc_samples"
    " (ts, pid, create_time, name, cpu_pct, cpu_time_delta_s,"
    "  rss, io_read_delta, io_write_delta)"
    " VALUES (?,?,?,?,?,?,?,?,?)"
)


def insert_sample(conn: sqlite3.Connection, ts: int, row: dict) -> None:
    conn.execute(_SAMPLE_INSERT, (ts, *(row[c] for c in SAMPLE_COLUMNS)))


def insert_proc_samples(conn: sqlite3.Connection, ts: int, rows: list[dict]) -> None:
    conn.executemany(_PROC_INSERT, [
        (ts, r["pid"], r["create_time"], r["name"], r["cpu_pct"],
         r["cpu_time_delta_s"], r["rss"], r["io_read_delta"], r["io_write_delta"])
        for r in rows
    ])


def sample_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


def find_gaps(conn: sqlite3.Connection, threshold_s: float) -> list[tuple[int, int]]:
    """Consecutive sample timestamps separated by more than threshold_s.

    Sleep, hibernate, shutdown, collector crashes and single dropped ticks all
    present identically here, which is intended: no analysis window may span one.
    """
    rows = conn.execute(
        "SELECT ts, LEAD(ts) OVER (ORDER BY ts) AS nxt FROM samples"
    ).fetchall()
    return [
        (ts, nxt) for ts, nxt in rows
        if nxt is not None and (nxt - ts) > threshold_s
    ]


def purge_proc_samples(conn: sqlite3.Connection, older_than_ts: int) -> int:
    cur = conn.execute("DELETE FROM proc_samples WHERE ts < ?", (older_than_ts,))
    return cur.rowcount


def purge_events(conn: sqlite3.Connection, older_than_ts: int) -> int:
    cur = conn.execute("DELETE FROM events WHERE ts < ?", (older_than_ts,))
    return cur.rowcount


def record_collection_gap(
    conn: sqlite3.Connection, channel: str,
    start_ts: int | None, end_ts: int | None, detected_at: int,
) -> None:
    conn.execute(
        "INSERT INTO collection_gaps(channel, start_ts, end_ts, detected_at)"
        " VALUES (?,?,?,?)",
        (channel, start_ts, end_ts, detected_at),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_store_io.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/store.py tests/telemetry/test_store_io.py
git commit -m "feat(telemetry): add store write/read APIs and gap detection"
```

---

### Task 6: Redaction

**Files:**
- Create: `src/telemetry/redaction.py`
- Test: `tests/telemetry/test_redaction.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `redaction.redact(text: str, username: str | None = None, max_len: int = 512) -> str`.

- [ ] **Step 1: Write the failing test**

```python
from telemetry.redaction import redact


def test_user_path_on_c_drive():
    out = redact(r"failed opening C:\Users\yashash\Documents\tax.pdf")
    assert "yashash" not in out
    assert "<redacted>" in out


def test_user_path_on_other_drive_letters():
    out = redact(r"D:\Users\alice\notes.txt and E:\Users\bob\x.log")
    assert "alice" not in out and "bob" not in out


def test_unc_path():
    out = redact(r"copy failed from \\fileserver\payroll\q3.xlsx")
    assert "fileserver" not in out
    assert "payroll" not in out


def test_url():
    out = redact("posted to https://internal.example.com/secret?token=abc")
    assert "example.com" not in out
    assert "<url redacted>" in out


def test_email():
    out = redact("notify yashashgdg@gmail.com about it")
    assert "yashashgdg@gmail.com" not in out


def test_literal_username_anywhere():
    out = redact("profile for yashash could not load", username="yashash")
    assert "yashash" not in out


def test_truncated_to_max_len():
    assert len(redact("x" * 5000)) <= 512


def test_plain_text_is_left_alone():
    msg = "The device is not ready for use"
    assert redact(msg) == msg


def test_empty_and_none_safe():
    assert redact("") == ""
    assert redact(None) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_redaction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.redaction'`

- [ ] **Step 3: Write the implementation**

```python
"""Best-effort redaction for opted-in Event Log text.

This cannot catch application-specific identifiers or document names embedded
in arbitrary error strings. That limitation is stated in the opt-in dialog;
anything surviving here reaches exported reports.
"""
import re

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_USER_DIR = re.compile(r"\b([A-Za-z]):\\Users\\[^\\\s\"']+", re.IGNORECASE)
_UNC = re.compile(r"\\\\[^\\\s\"']+\\[^\\\s\"']+")


def redact(text: str | None, username: str | None = None, max_len: int = 512) -> str:
    if not text:
        return ""
    out = _URL.sub("<url redacted>", text)
    out = _EMAIL.sub("<email redacted>", out)
    out = _USER_DIR.sub(lambda m: f"{m.group(1)}:\\Users\\<redacted>", out)
    out = _UNC.sub(r"\\\\<redacted>", out)
    if username:
        out = re.sub(re.escape(username), "<redacted>", out, flags=re.IGNORECASE)
    return out[:max_len]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_redaction.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/redaction.py tests/telemetry/test_redaction.py
git commit -m "feat(telemetry): add best-effort text redaction for event data"
```

---

### Task 7: Event Log reader

**Files:**
- Create: `src/telemetry/eventlog.py`
- Test: `tests/telemetry/test_eventlog.py`

**Interfaces:**
- Consumes: `store.set_meta`, `store.get_meta`, `store.record_collection_gap`,
  `redaction.redact`.
- Produces: `eventlog.EventLogReader(channel: str)` with
  `read_new(conn, capture_messages: bool = False, limit: int = 500) -> int`
  (returns rows inserted), `eventlog.parse_event_xml(xml: str) -> dict | None`,
  `eventlog.LEVEL_NAMES`, `eventlog.watermark_key(channel) -> str`.

**Note on message text:** instead of formatting publisher message templates
(which needs `EvtOpenPublisherMetadata` and often fails for third-party
providers), the opted-in text is the concatenation of the event's `EventData`
values. Those carry the substituted parameters — paths, process names — which
is both the useful part and the sensitive part, so it stays behind the same
opt-in.

- [ ] **Step 1: Write the failing test**

```python
import sys

import pytest

from telemetry import store
from telemetry.eventlog import EventLogReader, parse_event_xml, watermark_key

pywin32 = pytest.importorskip("win32evtlog")
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

SAMPLE_XML = """<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'>
<System><Provider Name='Microsoft-Windows-Kernel-Power'/>
<EventID>41</EventID><Level>1</Level>
<TimeCreated SystemTime='2026-07-27T14:33:05.1234567Z'/>
<EventRecordID>90210</EventRecordID><Channel>System</Channel></System>
<EventData><Data Name='BugcheckCode'>0</Data>
<Data Name='Path'>C:\\Users\\yashash\\x.sys</Data></EventData></Event>"""


def test_parse_extracts_system_fields():
    got = parse_event_xml(SAMPLE_XML)
    assert got["provider"] == "Microsoft-Windows-Kernel-Power"
    assert got["event_id"] == 41
    assert got["record_id"] == 90210
    assert got["level"] == "Critical"
    assert got["ts"] > 0


def test_parse_collects_event_data_values():
    got = parse_event_xml(SAMPLE_XML)
    assert "BugcheckCode" not in got["data"]     # names dropped, values kept
    assert "0" in got["data"]


def test_parse_returns_none_on_garbage():
    assert parse_event_xml("<not-an-event/>") is None


def test_watermark_key_is_per_channel():
    assert watermark_key("System") != watermark_key("Application")


def test_read_new_inserts_and_advances_watermark(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    reader = EventLogReader("System")
    inserted = reader.read_new(conn, limit=20)
    assert inserted >= 0
    if inserted:
        wm = int(store.get_meta(conn, watermark_key("System")))
        top = conn.execute(
            "SELECT MAX(record_id) FROM events WHERE channel='System'"
        ).fetchone()[0]
        assert wm == top


def test_second_read_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    reader = EventLogReader("System")
    reader.read_new(conn, limit=20)
    before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    reader.read_new(conn, limit=20)
    after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert after >= before
    dupes = conn.execute(
        "SELECT channel, record_id, COUNT(*) c FROM events"
        " GROUP BY channel, record_id HAVING c > 1"
    ).fetchall()
    assert dupes == []


def test_messages_absent_unless_opted_in(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    EventLogReader("System").read_new(conn, capture_messages=False, limit=20)
    vals = conn.execute("SELECT message_redacted FROM events").fetchall()
    assert all(v[0] is None for v in vals)


def test_watermark_reset_records_a_collection_gap(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    # A watermark far beyond any real record means the log was cleared/wrapped.
    store.set_meta(conn, watermark_key("System"), str(10**15))
    EventLogReader("System").read_new(conn, limit=20)
    gaps = conn.execute(
        "SELECT channel FROM collection_gaps WHERE channel='System'"
    ).fetchall()
    assert gaps, "expected a collection_gaps row after watermark reset"


def test_invalidated_watermark_resets_to_current_end_not_zero(tmp_path):
    """Resetting to 0 would replay the entire retained log as newly observed."""
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    reader = EventLogReader("System")
    newest = reader.newest_record_id()
    if newest is None:
        pytest.skip("System channel is empty on this machine")
    store.set_meta(conn, watermark_key("System"), str(10**15))
    reader.read_new(conn, limit=20)
    assert int(store.get_meta(conn, watermark_key("System"))) == newest
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_transient_failure_preserves_watermark(tmp_path, monkeypatch):
    """A temporary access error must not rewind ingestion."""
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    reader = EventLogReader("System")
    store.set_meta(conn, watermark_key("System"), "12345")

    def boom(*args, **kwargs):
        raise OSError("access denied")

    monkeypatch.setattr(reader, "newest_record_id", lambda: 99999)
    monkeypatch.setattr(reader, "_query", boom)

    assert reader.read_new(conn, limit=20) == 0
    assert store.get_meta(conn, watermark_key("System")) == "12345"
    assert conn.execute("SELECT COUNT(*) FROM collection_gaps").fetchone()[0] == 0


def test_unreadable_channel_preserves_watermark(tmp_path):
    """newest_record_id() failing is not proof of invalidation."""
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.set_meta(conn, watermark_key("System"), "777")
    reader = EventLogReader("Nonexistent-Channel-Xyz")
    store.set_meta(conn, watermark_key("Nonexistent-Channel-Xyz"), "777")
    assert reader.read_new(conn, limit=5) == 0
    assert store.get_meta(conn, watermark_key("Nonexistent-Channel-Xyz")) == "777"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_eventlog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.eventlog'`

- [ ] **Step 3: Write the implementation**

```python
"""Windows Event Log ingestion.

The watermark is the last EventRecordID seen per channel. Record IDs are
monotonic per channel, not globally, so both the watermark and the uniqueness
constraint are channel-scoped.

The watermark advances in the SAME transaction as the event insert. Committing
them separately guarantees either replayed or dropped events on an unclean
shutdown.
"""
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from . import store
from .redaction import redact

try:
    import win32evtlog
except ImportError:  # non-Windows or pywin32 absent
    win32evtlog = None

_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}

LEVEL_NAMES = {
    0: "Information", 1: "Critical", 2: "Error",
    3: "Warning", 4: "Information", 5: "Verbose",
}


def watermark_key(channel: str) -> str:
    return f"evtlog_watermark_{channel.lower()}"


def _parse_ts(raw: str) -> int:
    # SystemTime has 7 fractional digits; datetime accepts at most 6.
    cleaned = raw.replace("Z", "+00:00")
    if "." in cleaned:
        head, _, tail = cleaned.partition(".")
        frac, _, offset = tail.partition("+")
        cleaned = f"{head}.{frac[:6]}+{offset}" if offset else f"{head}.{frac[:6]}"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_event_xml(xml: str) -> dict | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None

    system = root.find("e:System", _NS)
    if system is None:
        return None

    provider = system.find("e:Provider", _NS)
    event_id = system.find("e:EventID", _NS)
    record_id = system.find("e:EventRecordID", _NS)
    created = system.find("e:TimeCreated", _NS)
    level = system.find("e:Level", _NS)
    if event_id is None or record_id is None or created is None:
        return None

    values = [
        (d.text or "").strip()
        for d in root.findall("e:EventData/e:Data", _NS)
        if (d.text or "").strip()
    ]

    return {
        "ts": _parse_ts(created.get("SystemTime")),
        "record_id": int(record_id.text),
        "provider": provider.get("Name") if provider is not None else None,
        "event_id": int(event_id.text),
        "level": LEVEL_NAMES.get(int(level.text) if level is not None else 0, "Information"),
        "data": " | ".join(values),
    }


class EventLogReader:
    def __init__(self, channel: str) -> None:
        self.channel = channel

    def _query(self, after_record_id: int):
        query = f"*[System[(EventRecordID > {after_record_id})]]"
        return win32evtlog.EvtQuery(
            self.channel,
            win32evtlog.EvtQueryChannelPath | win32evtlog.EvtQueryForwardDirection,
            query,
            None,
        )

    def newest_record_id(self) -> int | None:
        """Highest EventRecordID currently in the channel, or None if empty."""
        try:
            handle = win32evtlog.EvtQuery(
                self.channel,
                win32evtlog.EvtQueryChannelPath | win32evtlog.EvtQueryReverseDirection,
                "*",
                None,
            )
            items = win32evtlog.EvtNext(handle, 1)
        except Exception:
            return None
        if not items:
            return None
        record = parse_event_xml(
            win32evtlog.EvtRender(items[0], win32evtlog.EvtRenderEventXml)
        )
        return record["record_id"] if record else None

    def read_new(self, conn, capture_messages: bool = False, limit: int = 500) -> int:
        if win32evtlog is None:
            return 0

        watermark = int(store.get_meta(conn, watermark_key(self.channel), "0"))

        newest = self.newest_record_id()
        if newest is None:
            # Channel empty, or unreadable right now. Neither is proof that the
            # watermark is invalid, so leave it alone and try again next poll.
            _log.info("channel %s returned no newest record; watermark kept at %d",
                      self.channel, watermark)
            return 0

        # A watermark ahead of the newest record is proof the log was cleared,
        # wrapped, or recreated. This does NOT raise -- the query is perfectly
        # valid and simply matches nothing -- so it must be checked explicitly,
        # or the channel would silently stop ingesting forever.
        if watermark > newest:
            self._reset_to_end(conn, previous=watermark, newest=newest)
            return 0

        try:
            handle = self._query(watermark)
        except Exception:
            # Transient: access denied, service restarting, handle exhaustion.
            # Preserve the watermark. Resetting here would replay the entire
            # retained log as if newly observed.
            _log.exception("transient query failure on %s; watermark preserved at %d",
                           self.channel, watermark)
            return 0

        parsed: list[dict] = []
        while len(parsed) < limit:
            try:
                items = win32evtlog.EvtNext(handle, 64)
            except Exception:
                break
            if not items:
                break
            for item in items:
                xml = win32evtlog.EvtRender(item, win32evtlog.EvtRenderEventXml)
                record = parse_event_xml(xml)
                if record:
                    parsed.append(record)

        if not parsed:
            return 0

        highest = max(r["record_id"] for r in parsed)
        rows = [
            (
                r["ts"], self.channel, r["record_id"], r["provider"],
                r["event_id"], r["level"],
                redact(r["data"]) if capture_messages else None,
            )
            for r in parsed
        ]

        conn.execute("BEGIN")
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO events"
                " (ts, channel, record_id, provider, event_id, level, message_redacted)"
                " VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            store.set_meta(conn, watermark_key(self.channel), str(highest))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return len(rows)

    def _reset_to_end(self, conn, previous: int, newest: int) -> None:
        """Recover from a proven-invalid watermark.

        Resets to the CURRENT END of the log, never to 0: starting from 0 would
        re-ingest every retained record as if newly observed, backdating a
        month of events into the middle of an unrelated analysis window.
        Everything between the last event we stored and now is genuinely lost,
        so it is recorded as a coverage gap.
        """
        now = int(time.time())
        last_ts = conn.execute(
            "SELECT MAX(ts) FROM events WHERE channel = ?", (self.channel,)
        ).fetchone()[0]
        conn.execute("BEGIN")
        try:
            store.set_meta(conn, watermark_key(self.channel), str(newest))
            store.record_collection_gap(
                conn, self.channel, start_ts=last_ts, end_ts=now, detected_at=now
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        _log.warning(
            "watermark for %s was %d but newest record is %d; log was cleared or"
            " wrapped. Reset to %d and recorded a coverage gap.",
            self.channel, previous, newest, newest,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_eventlog.py -v`
Expected: 8 passed (skipped entirely on non-Windows)

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/eventlog.py tests/telemetry/test_eventlog.py
git commit -m "feat(telemetry): add Event Log reader with transactional watermark"
```

---

### Task 8: Collector loop

**Files:**
- Create: `src/telemetry/collector.py`
- Test: `tests/telemetry/test_collector.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `collector.Collector(conn, capture_messages=False)` with
  `tick_system() -> dict | None`, `should_burst(row) -> bool`,
  `maybe_tick_processes(now, row) -> int`, `maybe_poll_events(now) -> int`,
  `run_once(now=None) -> None`, `run_forever() -> None`;
  `collector.consent_granted(conn) -> bool`, `collector.grant_consent(conn)`,
  `collector.acquire_singleton() -> bool`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from telemetry import config, store
from telemetry.collector import Collector, consent_granted, grant_consent


def _conn(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    return conn


def test_consent_defaults_to_false(tmp_path):
    assert consent_granted(_conn(tmp_path)) is False


def test_grant_consent_persists(tmp_path):
    conn = _conn(tmp_path)
    grant_consent(conn)
    assert consent_granted(conn) is True


def test_first_tick_writes_a_row_with_null_rates(tmp_path):
    conn = _conn(tmp_path)
    c = Collector(conn)
    c.run_once(now=1000)
    row = conn.execute(
        "SELECT elapsed_ms, disk_read_bps, mem_pct FROM samples"
    ).fetchone()
    assert row[0] is None and row[1] is None
    assert row[2] is not None


def test_second_tick_has_elapsed_and_rates(tmp_path):
    conn = _conn(tmp_path)
    c = Collector(conn)
    c.run_once(now=1000)
    c.run_once(now=1030)
    row = conn.execute(
        "SELECT elapsed_ms FROM samples WHERE ts = 1030"
    ).fetchone()
    assert row[0] is not None and row[0] >= 0


def test_burst_triggers_on_high_cpu():
    assert Collector.should_burst({"cpu_pct": 95.0, "mem_pct": 10.0,
                                   "disk_busy_pct": 0.0}) is True


def test_burst_triggers_on_high_memory():
    assert Collector.should_burst({"cpu_pct": 1.0, "mem_pct": 90.0,
                                   "disk_busy_pct": 0.0}) is True


def test_no_burst_when_idle():
    assert Collector.should_burst({"cpu_pct": 5.0, "mem_pct": 20.0,
                                   "disk_busy_pct": 1.0}) is False


def test_burst_tolerates_missing_values():
    assert Collector.should_burst({"cpu_pct": None, "mem_pct": None,
                                   "disk_busy_pct": None}) is False


def test_processes_not_sampled_before_cadence_elapses(tmp_path):
    conn = _conn(tmp_path)
    c = Collector(conn)
    c.run_once(now=1000)
    c.run_once(now=1030)          # 30s later, process cadence is 300s
    count = conn.execute("SELECT COUNT(*) FROM proc_samples").fetchone()[0]
    assert count == 0


def test_processes_sampled_once_cadence_elapses(tmp_path):
    conn = _conn(tmp_path)
    c = Collector(conn)
    c.run_once(now=1000)
    c.run_once(now=1000 + config.PROCESS_CADENCE_S)
    c.run_once(now=1000 + 2 * config.PROCESS_CADENCE_S)
    count = conn.execute("SELECT COUNT(*) FROM proc_samples").fetchone()[0]
    assert count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_collector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.collector'`

- [ ] **Step 3: Write the implementation**

```python
"""The collection loop.

Deliberately never calls the detector: an always-on mode is added later as a
second reader of the store, not as a change to how data is produced.
"""
import ctypes
import time

from . import config, sampler, store
from .eventlog import EventLogReader
from .rates import CounterTracker
from .sampler import ProcessSampler

_ERROR_ALREADY_EXISTS = 183


def consent_granted(conn) -> bool:
    return store.get_meta(conn, "consent_granted", "0") == "1"


def grant_consent(conn) -> None:
    store.set_meta(conn, "consent_granted", "1")
    store.set_meta(conn, "consent_granted_at", str(int(time.time())))


def acquire_singleton() -> bool:
    """False if another collector already holds the mutex.

    Without this a stale Task Scheduler entry plus a manual run would both
    write, duplicating timestamps and doubling rates.
    """
    if not hasattr(ctypes, "windll"):
        return True
    handle = ctypes.windll.kernel32.CreateMutexW(None, True, config.MUTEX_NAME)
    if not handle:
        return False
    return ctypes.windll.kernel32.GetLastError() != _ERROR_ALREADY_EXISTS


class Collector:
    def __init__(self, conn, capture_messages: bool = False) -> None:
        self.conn = conn
        self.capture_messages = capture_messages
        self._tracker = CounterTracker()
        self._procs = ProcessSampler()
        self._readers = [EventLogReader(c) for c in config.EVENT_CHANNELS]
        self._last_proc_ts: float | None = None
        self._last_event_ts: float | None = None
        self._last_purge_ts: float | None = None

    @staticmethod
    def should_burst(row: dict) -> bool:
        return (
            (row.get("cpu_pct") or 0.0) > config.BURST_CPU_PCT
            or (row.get("mem_pct") or 0.0) > config.BURST_MEM_PCT
            or (row.get("disk_busy_pct") or 0.0) > config.BURST_DISK_BUSY_PCT
        )

    def tick_system(self, now: float) -> dict:
        raw = sampler.sample_system_raw()
        elapsed_ms, deltas = self._tracker.tick(raw["counters"], now=now)
        row = sampler.build_sample_row(raw, elapsed_ms, deltas)
        store.insert_sample(self.conn, ts=int(now), row=row)
        return row

    def maybe_tick_processes(self, now: float, row: dict) -> int:
        cadence = (
            config.PROCESS_BURST_CADENCE_S if self.should_burst(row)
            else config.PROCESS_CADENCE_S
        )
        if self._last_proc_ts is not None and (now - self._last_proc_ts) < cadence:
            return 0
        elapsed_s = None if self._last_proc_ts is None else now - self._last_proc_ts
        self._last_proc_ts = now
        rows = self._procs.sample(top_n=config.TOP_N, elapsed_s=elapsed_s)
        if rows:
            store.insert_proc_samples(self.conn, ts=int(now), rows=rows)
        return len(rows)

    def maybe_poll_events(self, now: float) -> int:
        if self._last_event_ts is not None and (now - self._last_event_ts) < config.EVENT_POLL_S:
            return 0
        self._last_event_ts = now
        total = 0
        for reader in self._readers:
            try:
                total += reader.read_new(self.conn, self.capture_messages)
            except Exception:
                # A failing channel must not take the whole collector down.
                continue
        return total

    def maybe_purge(self, now: float) -> None:
        if self._last_purge_ts is not None and (now - self._last_purge_ts) < 86400:
            return
        self._last_purge_ts = now
        store.purge_proc_samples(self.conn, int(now) - config.PROC_RETENTION_DAYS * 86400)
        store.purge_events(self.conn, int(now) - config.EVENT_RETENTION_DAYS * 86400)

    def run_once(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        row = self.tick_system(now)
        self.maybe_tick_processes(now, row)
        self.maybe_poll_events(now)
        self.maybe_purge(now)

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:
                # Never let one bad tick end collection.
                pass
            drift = time.monotonic() - started
            time.sleep(max(0.0, config.SYSTEM_CADENCE_S - drift))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/telemetry/test_collector.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/collector.py tests/telemetry/test_collector.py
git commit -m "feat(telemetry): add collector loop with burst sampling and consent gate"
```

---

### Task 9: CLI and Task Scheduler registration

**Files:**
- Create: `src/telemetry/schedule.py`
- Create: `src/telemetry/__main__.py`
- Test: `tests/telemetry/test_schedule.py`
- Modify: `README.md` (add a "Telemetry collector" section)

**Interfaces:**
- Consumes: `config.TASK_NAME`, `collector.*`.
- Produces: `schedule.build_command(python_exe, module) -> str`,
  `schedule.register(command) -> bool`, `schedule.unregister() -> bool`,
  `schedule.is_registered() -> bool`.
  CLI verbs: `run`, `install`, `uninstall`, `status`, `accept-consent`,
  `delete-all-data`.

- [ ] **Step 1: Write the failing test**

```python
import sys

import pytest

from telemetry import config, schedule


def test_build_command_quotes_paths_with_spaces():
    cmd = schedule.build_command(r"C:\Program Files\Python\python.exe", "telemetry")
    assert cmd.startswith('"C:\\Program Files\\Python\\python.exe"')
    assert "-m telemetry run" in cmd


def test_build_command_uses_exe_directly_when_frozen():
    cmd = schedule.build_command(r"C:\Apps\RCA-Collector.exe", None)
    assert cmd == '"C:\\Apps\\RCA-Collector.exe" run'


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_is_registered_returns_bool_without_raising():
    assert isinstance(schedule.is_registered(), bool)


def test_task_name_is_stable():
    assert config.TASK_NAME == "RCA Telemetry Collector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/telemetry/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'telemetry.schedule'`

- [ ] **Step 3: Write the implementation**

`src/telemetry/schedule.py`:

```python
"""Task Scheduler registration via schtasks.

schtasks is used rather than the COM API: it needs no extra dependency, and
registration is a one-line command that is easy to verify by hand.
"""
import subprocess
import sys

from . import config


def build_command(executable: str, module: str | None) -> str:
    if module is None:
        return f'"{executable}" run'
    return f'"{executable}" -m {module} run'


def default_command() -> str:
    if getattr(sys, "frozen", False):
        return build_command(sys.executable, None)
    return build_command(sys.executable, "telemetry")


def register(command: str | None = None) -> bool:
    command = command or default_command()
    result = subprocess.run(
        ["schtasks", "/Create", "/TN", config.TASK_NAME, "/TR", command,
         "/SC", "ONLOGON", "/F"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def unregister() -> bool:
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", config.TASK_NAME, "/F"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def is_registered() -> bool:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", config.TASK_NAME],
        capture_output=True, text=True,
    )
    return result.returncode == 0
```

`src/telemetry/__main__.py`:

```python
"""Collector CLI.

    python -m telemetry accept-consent
    python -m telemetry install
    python -m telemetry run
    python -m telemetry status
    python -m telemetry uninstall
    python -m telemetry delete-all-data
"""
import argparse
import sys

from . import config, schedule, store
from .collector import Collector, acquire_singleton, consent_granted, grant_consent

CONSENT_TEXT = """
This tool records, on this machine only:
  - system metrics every 30 seconds (CPU, memory, disk, network, battery)
  - the top processes by CPU and memory every 5 minutes (name, PID, usage)
  - Windows System and Application event metadata (provider, ID, level, time)

Window titles are never captured. Event message text is NOT stored unless you
enable it separately. Nothing is ever sent over the network.

Stored at: {path}
Delete everything at any time with:  python -m telemetry delete-all-data
"""


def _open():
    conn = store.connect(config.db_path())
    store.init_schema(conn)
    return conn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="telemetry")
    parser.add_argument(
        "command",
        choices=["run", "install", "uninstall", "status",
                 "accept-consent", "delete-all-data"],
    )
    parser.add_argument(
        "--capture-messages", action="store_true",
        help="store redacted event data text (off by default; see privacy notes)",
    )
    args = parser.parse_args(argv)
    conn = _open()

    if args.command == "accept-consent":
        print(CONSENT_TEXT.format(path=config.db_path()))
        grant_consent(conn)
        print("Consent recorded. Run 'python -m telemetry install' to start collecting.")
        return 0

    if args.command == "status":
        count = store.sample_count(conn)
        days = count * config.SYSTEM_CADENCE_S / 86400
        print(f"consent:    {'granted' if consent_granted(conn) else 'NOT GRANTED'}")
        print(f"scheduled:  {schedule.is_registered()}")
        print(f"samples:    {count} (~{days:.2f} days)")
        print(f"database:   {config.db_path()}")
        return 0

    if args.command == "install":
        if not consent_granted(conn):
            print("Consent not granted. Run 'accept-consent' first.", file=sys.stderr)
            return 1
        ok = schedule.register()
        print("Registered." if ok else "Registration failed.")
        return 0 if ok else 1

    if args.command == "uninstall":
        ok = schedule.unregister()
        print("Unregistered." if ok else "Unregistration failed.")
        return 0 if ok else 1

    if args.command == "delete-all-data":
        schedule.unregister()
        conn.close()
        path = config.db_path()
        for suffix in ("", "-wal", "-shm"):
            candidate = path.with_name(path.name + suffix)
            if candidate.exists():
                candidate.unlink()
        print(f"Deleted {path} and unregistered the scheduled task.")
        return 0

    # run
    if not consent_granted(conn):
        print("Consent not granted. Run 'accept-consent' first.", file=sys.stderr)
        return 1
    if not acquire_singleton():
        print("Another collector is already running.", file=sys.stderr)
        return 0
    Collector(conn, capture_messages=args.capture_messages).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/telemetry/ -v`
Expected: all pass

- [ ] **Step 5: Verify the collector actually runs end to end**

```bash
python -m telemetry accept-consent
python -m telemetry run
```

Let it run for ~3 minutes, stop with Ctrl-C, then:

```bash
python -m telemetry status
```

Expected: `samples:` shows roughly 6 rows. Confirm the first row has
`elapsed_ms IS NULL` and later rows do not:

```bash
python -c "from telemetry import config, store; c=store.connect(config.db_path()); print(c.execute('SELECT ts, elapsed_ms, cpu_pct, mem_pct FROM samples ORDER BY ts LIMIT 5').fetchall())"
```

- [ ] **Step 6: Add the README section**

Add to `README.md`:

```markdown
## Telemetry collector

The RCA pipeline runs on real telemetry from this machine. Collection is
opt-in and everything stays local.

    python -m telemetry accept-consent   # shows what is collected, records consent
    python -m telemetry install          # registers a logon task
    python -m telemetry status           # consent, schedule, days collected
    python -m telemetry uninstall        # stops future collection
    python -m telemetry delete-all-data  # removes the database entirely

Training needs about **3 days** of collected samples. `status` reports
progress.

Window titles are never captured. Event message text is stored only with
`--capture-messages`, and is redacted on a best-effort basis. Nothing is sent
over the network.
```

- [ ] **Step 7: Commit**

```bash
git add src/telemetry/schedule.py src/telemetry/__main__.py tests/telemetry/test_schedule.py README.md
git commit -m "feat(telemetry): add collector CLI and Task Scheduler registration"
```

---

## Verification checklist

After Task 9, confirm before starting Plan 2:

- [ ] `python -m pytest tests/telemetry/ -v` — all pass
- [ ] `python -m telemetry status` reports consent granted and task registered
- [ ] Sample count grows over 10 minutes without the collector process exceeding
      ~30 MB RSS (`Get-Process python | Select-Object WorkingSet64`)
- [ ] `SELECT COUNT(*) FROM proc_samples` is non-zero after 5+ minutes
- [ ] No `np.random`, `torch.randn`, or fabricated values anywhere in
      `src/telemetry/`:
      `grep -rn "np\.random\|torch\.randn\|random\." src/telemetry/` returns empty
- [ ] Starting a second collector exits immediately rather than double-writing
- [ ] `message_redacted` is NULL for every row when run without
      `--capture-messages`
- [ ] The database is not world-readable. `%LOCALAPPDATA%` grants access to the
      owning user and administrators by default and the store inherits that, so
      no explicit ACL call is needed — but confirm it rather than assume:
      `icacls "%LOCALAPPDATA%\RCA\telemetry.db"` should list only the current
      user, SYSTEM, and Administrators.

## Deferred to later plans

- Reading the store for training and inference (Plan 2)
- Baseline filtering, model artifact, incident segmentation (Plan 2)
- Topology prior, Granger guardrails, attribution (Plan 3)
- Desktop UI, consent dialog in the GUI, `packaging/excludes.txt` regeneration
  for the new `psutil`/`pywin32` dependencies (Plan 4)
