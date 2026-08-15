"""SQLite persistence for collected telemetry."""

import sqlite3
import time
from pathlib import Path
from typing import Iterable

from . import config
from .logsetup import get_logger

_LOGGER = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts INTEGER PRIMARY KEY, elapsed_ms INTEGER,
    cpu_pct REAL, cpu_pct_max_core REAL, cpu_freq_mhz REAL, cpu_freq_ratio REAL,
    mem_pct REAL, mem_available_mb REAL, swap_pct REAL, swap_used_bytes INTEGER,
    swap_used_delta INTEGER, disk_read_bps REAL, disk_write_bps REAL,
    disk_busy_pct REAL, disk_free_pct REAL, net_sent_bps REAL, net_recv_bps REAL,
    process_count INTEGER, battery_pct REAL, battery_drain_rate REAL,
    power_plugged INTEGER, on_battery INTEGER, user_idle_sec REAL,
    foreground_app TEXT, cpu_busy_s_delta REAL, mem_used_bytes INTEGER,
    disk_read_bytes_delta INTEGER, disk_write_bytes_delta INTEGER
);
CREATE TABLE IF NOT EXISTS proc_samples (
    ts INTEGER NOT NULL, pid INTEGER NOT NULL, create_time REAL NOT NULL,
    name TEXT, cpu_pct REAL, cpu_time_delta_s REAL, rss INTEGER,
    io_read_delta INTEGER, io_write_delta INTEGER,
    UNIQUE(ts, pid, create_time)
);
CREATE INDEX IF NOT EXISTS idx_proc_ts ON proc_samples(ts);
CREATE TABLE IF NOT EXISTS events (
    ts INTEGER NOT NULL, channel TEXT NOT NULL, record_id INTEGER NOT NULL,
    provider TEXT, event_id INTEGER, level TEXT, message_redacted TEXT,
    UNIQUE(channel, record_id)
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS collection_gaps (
    channel TEXT NOT NULL, start_ts INTEGER, end_ts INTEGER, detected_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


#: Sqlite messages that mean the file itself is unusable, as opposed to merely
#: busy. Matched on text because sqlite3 raises the same exception class for
#: both, and telling them apart is the whole point -- see connect().
_CORRUPTION_MARKERS = (
    "file is not a database",
    "database disk image is malformed",
    "file is encrypted",
    "malformed database schema",
)


def is_corruption(error: sqlite3.Error) -> bool:
    """Whether this error means the file is broken rather than contended.

    `sqlite3.OperationalError` -- which is what "database is locked" raises --
    is a **subclass** of `sqlite3.DatabaseError`. Catching the parent and
    treating it as corruption would move a perfectly healthy database aside
    every time the collector happened to hold a write lock, which is a far
    worse outcome than the crash this is meant to prevent.
    """
    if isinstance(error, sqlite3.OperationalError):
        # "unable to open database file" is permissions or a missing
        # directory; "database is locked" is contention. Neither is damage.
        return any(marker in str(error).lower() for marker in _CORRUPTION_MARKERS)
    return isinstance(error, sqlite3.DatabaseError)


def quarantine(path: Path | str) -> Path | None:
    """Move a damaged database aside and return where it went.

    Deliberately renamed rather than deleted. The file is unreadable to us,
    but it is still the only copy of the user's history, and a later sqlite
    build or a recovery tool may get more out of it than we can.
    """
    path = Path(path)
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        path.replace(target)
        # WAL and shared-memory sidecars describe the old file; leaving them
        # behind would corrupt the replacement on its first open.
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
    except OSError:
        return None
    _LOGGER.error("Database was unreadable; moved it to %s and started fresh.", target)
    return target


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the telemetry database, recovering from a corrupted file.

    An unclean shutdown can leave the database unreadable, and every caller
    routes through here, so this is the one place that can turn "the app
    opens to a traceback" into "the app starts, says what happened, and keeps
    collecting". Recovery is one attempt only: if the fresh file is also
    unreadable the problem is the disk or the directory, and retrying in a
    loop would just destroy whatever is still there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _open(path)
    except sqlite3.Error as error:
        if not is_corruption(error):
            raise                      # locked, read-only, no such directory
        if quarantine(path) is None:
            raise
        return _open(path)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        # A corrupt file connects lazily and only fails on first use, so the
        # connection has to be closed before the file can be moved on Windows.
        conn.close()
        raise
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _add_missing_sample_columns(conn)
    set_meta(conn, "schema_version", str(config.SCHEMA_VERSION))


def _add_missing_sample_columns(conn: sqlite3.Connection) -> list[str]:
    """Add newly-introduced sample columns to an existing database.

    CREATE TABLE IF NOT EXISTS silently leaves an older table alone, so a new
    channel would never be written on a machine that has already been
    collecting. Existing rows keep NULL for the new column, which is why new
    channels stay out of the model's feature set until they have history --
    otherwise cleaning would discard every row collected before they existed.
    """
    from .sampler import SAMPLE_COLUMN_TYPES

    existing = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    added = []
    for column, sql_type in SAMPLE_COLUMN_TYPES.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {column} {sql_type}")
            added.append(column)
    return added


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value),
    )


def insert_sample(conn: sqlite3.Connection, ts: int, row: dict[str, object]) -> None:
    columns = tuple(row)
    conn.execute(
        f"INSERT OR IGNORE INTO samples (ts, {', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in range(len(columns) + 1))})",
        (ts, *(row[column] for column in columns)),
    )


def insert_proc_samples(conn: sqlite3.Connection, ts: int, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = ("pid", "create_time", "name", "cpu_pct", "cpu_time_delta_s", "rss", "io_read_delta", "io_write_delta")
    conn.executemany(
        "INSERT OR IGNORE INTO proc_samples "
        f"(ts, {', '.join(columns)}) VALUES ({', '.join('?' for _ in range(len(columns) + 1))})",
        [(ts, *(row[column] for column in columns)) for row in rows],
    )


def sample_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


def find_gaps(conn: sqlite3.Connection, threshold_s: float | None = None) -> list[tuple[int, int]]:
    threshold_s = config.gap_threshold_s() if threshold_s is None else threshold_s
    rows = conn.execute("SELECT ts, LEAD(ts) OVER (ORDER BY ts) FROM samples").fetchall()
    return [(ts, next_ts) for ts, next_ts in rows if next_ts is not None and next_ts - ts > threshold_s]


def purge_proc_samples(conn: sqlite3.Connection, older_than_ts: int) -> int:
    return conn.execute("DELETE FROM proc_samples WHERE ts < ?", (older_than_ts,)).rowcount


def purge_events(conn: sqlite3.Connection, older_than_ts: int) -> int:
    return conn.execute("DELETE FROM events WHERE ts < ?", (older_than_ts,)).rowcount


def record_collection_gap(
    conn: sqlite3.Connection, channel: str, start_ts: int | None,
    end_ts: int | None, detected_at: int,
) -> None:
    conn.execute(
        "INSERT INTO collection_gaps(channel, start_ts, end_ts, detected_at) VALUES (?, ?, ?, ?)",
        (channel, start_ts, end_ts, detected_at),
    )
