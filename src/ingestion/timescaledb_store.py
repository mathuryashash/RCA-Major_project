"""TimescaleDB metric storage and query layer."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# psycopg2 availability flag
# ---------------------------------------------------------------------------
try:
    import psycopg2
    import psycopg2.extras

    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    PSYCOPG2_AVAILABLE = False
    logger.info("psycopg2 not installed — TimescaleDBStore will operate as no-op")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False


class TimescaleDBStore:
    """Read/write metric and event data to TimescaleDB.

    Graceful degradation: when psycopg2 is missing or the database is
    unreachable every public method returns an empty/default result and
    logs a warning.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "password",
        dbname: str = "rca_system",
    ):
        self._conn = None
        self._dsn = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": dbname,
        }

        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not installed — TimescaleDBStore disabled")
            return

        try:
            self._conn = psycopg2.connect(**self._dsn)
            self._conn.autocommit = False
            logger.info("TimescaleDBStore connected to %s:%s/%s", host, port, dbname)
        except Exception as exc:
            logger.warning("TimescaleDB connection failed (%s) — store disabled", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def write_batch(self, metrics: List) -> int:
        """Bulk-insert MetricPoint objects into the ``metrics`` hypertable.

        Returns the number of rows inserted (0 if the database is unavailable).
        """
        if self._conn is None:
            logger.debug("write_batch called but DB unavailable — no-op")
            return 0

        sql = (
            "INSERT INTO metrics (time, host, service, metric_name, value, unit) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        rows = [
            (m.timestamp, m.host, m.service, m.metric_name, m.value, m.unit)
            for m in metrics
        ]
        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
            self._conn.commit()
            logger.info("Wrote %d metric rows to TimescaleDB", len(rows))
            return len(rows)
        except Exception as exc:
            logger.warning("write_batch failed: %s", exc)
            self._safe_rollback()
            return 0

    def query_range(
        self,
        start: datetime,
        end: datetime,
        metric_names: Optional[List[str]] = None,
        service: Optional[str] = None,
    ):
        """Query metrics in [start, end] and return a wide-format DataFrame.

        Columns = metric_name, index = time.  Returns an empty DataFrame when
        the database or pandas is unavailable.
        """
        if self._conn is None or not PANDAS_AVAILABLE:
            logger.debug(
                "query_range: DB or pandas unavailable — returning empty DataFrame"
            )
            if PANDAS_AVAILABLE:
                return pd.DataFrame()
            return []

        conditions = ["time >= %s", "time <= %s"]
        params: list = [start, end]

        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            conditions.append(f"metric_name IN ({placeholders})")
            params.extend(metric_names)

        if service:
            conditions.append("service = %s")
            params.append(service)

        where = " AND ".join(conditions)
        sql = (
            f"SELECT time, metric_name, value FROM metrics WHERE {where} ORDER BY time"
        )

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=["time", "metric_name", "value"])
            if df.empty:
                return df
            # Pivot to wide format
            df = df.pivot_table(
                index="time", columns="metric_name", values="value", aggfunc="mean"
            )
            return df
        except Exception as exc:
            logger.warning("query_range failed: %s", exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def write_event(
        self,
        event_type: str,
        service: str,
        payload: dict,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Insert a single event into the ``events`` table.

        Returns True on success, False otherwise.
        """
        if self._conn is None:
            logger.debug("write_event called but DB unavailable — no-op")
            return False

        sql = (
            "INSERT INTO events (event_type, service, payload, correlation_id) "
            "VALUES (%s, %s, %s, %s)"
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    sql, (event_type, service, json.dumps(payload), correlation_id)
                )
            self._conn.commit()
            return True
        except Exception as exc:
            logger.warning("write_event failed: %s", exc)
            self._safe_rollback()
            return False

    def query_events(self, start: datetime, end: datetime) -> List[dict]:
        """Return events in [start, end] as a list of dicts."""
        if self._conn is None:
            logger.debug("query_events: DB unavailable — returning empty list")
            return []

        sql = (
            "SELECT id, time, event_type, service, payload, correlation_id "
            "FROM events WHERE time >= %s AND time <= %s ORDER BY time"
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (start, end))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("query_events failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Normalization params
    # ------------------------------------------------------------------

    def get_normalization_params(
        self, metric_name: str, service: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return (mean, std) for the given metric/service or (None, None)."""
        if self._conn is None:
            return (None, None)

        sql = (
            "SELECT rolling_mean, rolling_std FROM normalization_params "
            "WHERE metric_name = %s AND service = %s"
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (metric_name, service))
                row = cur.fetchone()
            if row:
                return (row[0], row[1])
            return (None, None)
        except Exception as exc:
            logger.warning("get_normalization_params failed: %s", exc)
            return (None, None)

    def update_normalization_params(
        self,
        metric_name: str,
        service: str,
        mean: float,
        std: float,
        window_days: int = 7,
    ) -> bool:
        """Upsert normalization params for a metric/service pair."""
        if self._conn is None:
            logger.debug("update_normalization_params: DB unavailable — no-op")
            return False

        sql = (
            "INSERT INTO normalization_params "
            "(metric_name, service, rolling_mean, rolling_std, window_days) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (metric_name, service) DO UPDATE SET "
            "rolling_mean = EXCLUDED.rolling_mean, "
            "rolling_std = EXCLUDED.rolling_std, "
            "window_days = EXCLUDED.window_days, "
            "computed_at = NOW()"
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (metric_name, service, mean, std, window_days))
            self._conn.commit()
            return True
        except Exception as exc:
            logger.warning("update_normalization_params failed: %s", exc)
            self._safe_rollback()
            return False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the database connection is alive."""
        if self._conn is None:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_rollback(self):
        try:
            if self._conn and not self._conn.closed:
                self._conn.rollback()
        except Exception:
            pass
