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
    except OSError:
        _LOGGER.exception("Database is unreadable and could not be moved aside.")
        return None

    # Past this point the user's database has already moved. The sidecars are
    # cleaned in their own guard: bundling them with the rename meant a
    # sharing violation on the WAL -- an antivirus or indexer holding it for a
    # moment is enough -- returned None *after* the move, so the caller
    # re-raised, the app crashed anyway, a stale WAL describing a different
    # database was left to poison the replacement, and the one log line saying
    # where the history went never executed.
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        try:
            sidecar.unlink(missing_ok=True)
        except OSError:
            # Move it beside the database it belongs to rather than leaving it
            # to be adopted by the fresh file.
            try:
                sidecar.replace(target.with_name(target.name + suffix))
            except OSError:
                _LOGGER.error(
                    "Could not remove %s; delete it by hand before restarting.",
                    sidecar,
                )
    _prune_quarantines(path, keep=1)
    _LOGGER.error("Database was unreadable; moved it to %s and started fresh.", target)
    return target


def quarantine_files(path: Path | str) -> list[Path]:
    """Damaged databases set aside beside ``path``, newest last."""
    path = Path(path)
    return sorted(path.parent.glob(f"{path.name}.corrupt-*"))


def _prune_quarantines(path: Path, keep: int) -> None:
    """Keep only the most recent damaged copies.

    A failing disk plus a supervisor that restarts the collector up to twelve
    times can leave twelve full-size copies of a multi-gigabyte database in a
    directory whose whole purpose is to stay bounded. One is enough to hand to
    a recovery tool; the rest are the disk-filling problem this module exists
    to prevent, wearing the costume of a safety measure.
    """
    stale = quarantine_files(path)[:-keep] if keep else quarantine_files(path)
    for old in stale:
        for suffix in ("", "-wal", "-shm"):
            try:
                old.with_name(old.name + suffix).unlink(missing_ok=True)
            except OSError:
                pass


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
        # Overwrite freed cell content with zeros instead of leaving it in the
        # page. Without this, expiring foreground_app unlinks the string and
        # leaves it readable in the file: measured, 745 copies of a test
        # application name survived a purge that reported 2,000 rows blanked,
        # while the consent dialog told the user it had been erased. An
        # in-place shrink also frees no pages, so VACUUM never compacts them
        # away either. The cost is extra writes on delete, which is nothing at
        # a 30-second cadence.
        conn.execute("PRAGMA secure_delete=ON")
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


def purge_samples(conn: sqlite3.Connection, older_than_ts: int) -> int:
    """Expire metric history.

    This table had no retention at all while proc_samples expired at 30 days
    and events at 365, so the largest permanent contributor to the database
    was the one nobody had bounded. Measured before this existed: 3.33 MB/day
    with no ceiling, or roughly 1.2 GB in the first year.
    """
    return conn.execute("DELETE FROM samples WHERE ts < ?", (older_than_ts,)).rowcount


def purge_foreground_app(conn: sqlite3.Connection, older_than_ts: int) -> int:
    """Erase the focus record while keeping the readings it sat beside.

    A column, not a row: the numeric metrics on the same sample are machine
    state and stay for the full window, but the application name is a record
    of a person and does not need to. Blanking in place keeps the row's
    timestamp, so coverage, gap detection and the model's feature set are all
    untouched -- foreground_app is not in MODELLED_COLUMNS and never was.
    """
    return conn.execute(
        "UPDATE samples SET foreground_app = NULL "
        "WHERE ts < ? AND foreground_app IS NOT NULL",
        (older_than_ts,),
    ).rowcount


#: Below this there is nothing worth rewriting the database for. 2,000 pages
#: is about 8 MB at the usual 4 KB page size -- enough that a user would see
#: the difference, small enough that the daily check almost always skips.
VACUUM_MIN_FREE_PAGES = 2_000

#: ...but an absolute floor is the wrong shape once the database is large.
#: At the measured 3.33 MB/day, steady-state purging frees roughly 800 pages a
#: day, so an 8 MB threshold fires every two or three days -- and on a 1.2 GB
#: database that means rewriting the whole file to recover 0.7% of it, which
#: is precisely the churn the floor above exists to avoid. Requiring the free
#: space to be worth a tenth of the file keeps the cost proportional to the
#: benefit at every size.
VACUUM_MIN_FREE_FRACTION = 0.10


def reclaimable_bytes(conn: sqlite3.Connection) -> int:
    """Space deleted rows are holding that the filesystem cannot see."""
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
    return int(page_size) * int(free_pages)


def _database_path(conn: sqlite3.Connection) -> Path | None:
    for _seq, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return Path(filename)
    return None


def reclaim(conn: sqlite3.Connection) -> int:
    """Return deleted space to the filesystem, and report how much.

    Deleting rows does not shrink a SQLite file. Freed pages go on an internal
    free list and are reused for future writes, so without this the retention
    rules above would delete half a million rows and free zero bytes on disk
    -- the purge would look like it had done nothing at all.

    Skipped unless there is enough to be worth a full rewrite, and refused
    outright when the disk cannot hold the temporary copy VACUUM needs. A tool
    that diagnoses a machine must not be the thing that fills its disk, which
    is the same rule the evaluation harness had to learn about memory.
    """
    free_bytes = reclaimable_bytes(conn)
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    if free_bytes < VACUUM_MIN_FREE_PAGES * page_size:
        return 0

    path = _database_path(conn)
    if path is not None and path.exists():
        # Both gates, so a small database is governed by the absolute floor and
        # a large one by the proportion.
        if free_bytes < path.stat().st_size * VACUUM_MIN_FREE_FRACTION:
            return 0
    if path is not None and path.exists():
        try:
            import shutil

            # VACUUM builds a complete second copy before swapping it in.
            # Starting one without room for it is how a cleanup task turns
            # into the outage it was meant to prevent.
            needed = path.stat().st_size * 2
            if shutil.disk_usage(path.parent).free < needed:
                _LOGGER.warning(
                    "Skipping VACUUM: needs %.0f MB free, disk has less.",
                    needed / 1e6,
                )
                return 0
        except OSError:
            return 0

    try:
        conn.execute("VACUUM")
        # VACUUM in WAL mode writes the entire new database into the WAL, so
        # without a checkpoint nothing reaches the filesystem and the total
        # footprint *doubles*. Measured on a 45 MB database: main stayed at
        # 45.0 MB, the WAL grew to 44.7 MB, and the Captured Data tab -- which
        # sums main + wal + shm -- would have shown the disk usage rising
        # immediately after the purge meant to reduce it. The collector holds
        # its connection for the whole logon session, so this never resolved
        # on its own.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        # Contention or a read-only file. Nothing is lost -- the space stays
        # on the free list and the next daily pass will try again.
        _LOGGER.exception("VACUUM failed; space stays on the free list.")
        return 0
    _LOGGER.info("Reclaimed %.1f MB of deleted rows.", free_bytes / 1e6)
    return free_bytes


def record_collection_gap(
    conn: sqlite3.Connection, channel: str, start_ts: int | None,
    end_ts: int | None, detected_at: int,
) -> None:
    conn.execute(
        "INSERT INTO collection_gaps(channel, start_ts, end_ts, detected_at) VALUES (?, ?, ?, ?)",
        (channel, start_ts, end_ts, detected_at),
    )
