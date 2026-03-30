"""TimescaleDB metric storage and query layer."""

import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool

    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    psycopg2.pool = None  # type: ignore[assignment]
    PSYCOPG2_AVAILABLE = False
    logger.info("psycopg2 not installed — TimescaleDBStore will operate as no-op")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    PANDAS_AVAILABLE = False

DEFAULT_BATCH_SIZE = 100
DEFAULT_POOL_SIZE = 5
DEFAULT_POOL_MAX_CONN = 10


class TimescaleDBStore:
    """Read/write metric and event data to TimescaleDB with connection pooling.

    Features:
    - Thread-safe connection pooling via psycopg2.pool.ThreadedConnectionPool
    - Batch inserts for efficiency (configurable batch size)
    - Graceful degradation when TimescaleDB is unavailable
    - Async-safe: uses thread-safe connection pool
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "password",
        dbname: str = "rca_system",
        pool_size: int = DEFAULT_POOL_SIZE,
        pool_max_conn: int = DEFAULT_POOL_MAX_CONN,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self._dsn = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": dbname,
        }
        self._pool: Optional["psycopg2.pool.ThreadedConnectionPool"] = None
        self._batch_size = batch_size
        self._batch_buffer: List[Tuple] = []
        self._batch_lock = threading.Lock()
        self._closed = False

        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not installed — TimescaleDBStore disabled")
            return

        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=pool_size,
                maxconn=pool_max_conn,
                **self._dsn,
            )
            logger.info(
                "TimescaleDBStore connection pool created for %s:%s/%s (pool_size=%d)",
                host,
                port,
                dbname,
                pool_size,
            )
            self.create_hypertable()
        except Exception as exc:
            logger.warning(
                "TimescaleDB connection pool creation failed (%s) — store disabled", exc
            )
            self._pool = None

    def _get_connection(self):
        if self._pool is None or self._closed:
            return None
        try:
            return self._pool.getconn()
        except Exception as exc:
            logger.warning("Failed to get connection from pool: %s", exc)
            return None

    def _return_connection(self, conn):
        if self._pool and conn and not self._closed:
            try:
                self._pool.putconn(conn)
            except Exception:
                pass

    def write_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Write a single metric to TimescaleDB.

        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Metric timestamp (defaults to NOW())
            labels: Optional labels/tags for the metric

        Returns:
            True on success, False on failure
        """
        if self._pool is None or self._closed:
            logger.debug("write_metric called but DB unavailable — no-op")
            return False

        timestamp = timestamp or datetime.utcnow()
        labels_json = json.dumps(labels) if labels else "{}"

        conn = self._get_connection()
        if conn is None:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO metrics (time, metric_name, value, labels) "
                    "VALUES (%s, %s, %s, %s)",
                    (timestamp, metric_name, value, labels_json),
                )
            conn.commit()
            return True
        except Exception as exc:
            logger.warning("write_metric failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            self._return_connection(conn)

    def write_metrics_batch(
        self,
        metrics_list: List[Dict],
    ) -> int:
        """Batch write multiple metrics to TimescaleDB for efficiency.

        Args:
            metrics_list: List of dicts with keys: metric_name, value, timestamp, labels

        Returns:
            Number of metrics written (0 if DB unavailable)
        """
        if not metrics_list:
            return 0

        if self._pool is None or self._closed:
            logger.debug("write_metrics_batch called but DB unavailable — no-op")
            return 0

        rows = []
        for m in metrics_list:
            timestamp = m.get("timestamp") or datetime.utcnow()
            labels = m.get("labels") or {}
            rows.append(
                (
                    timestamp,
                    m["metric_name"],
                    float(m["value"]),
                    json.dumps(labels),
                )
            )

        conn = self._get_connection()
        if conn is None:
            return 0

        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    "INSERT INTO metrics (time, metric_name, value, labels) "
                    "VALUES (%s, %s, %s, %s)",
                    rows,
                    page_size=500,
                )
            conn.commit()
            logger.info("Wrote %d metrics to TimescaleDB", len(rows))
            return len(rows)
        except Exception as exc:
            logger.warning("write_metrics_batch failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return 0
        finally:
            self._return_connection(conn)

    def write_batch(self, metrics: List) -> int:
        """Bulk-insert MetricPoint objects into the ``metrics`` hypertable.

        Returns the number of rows inserted (0 if the database is unavailable).
        """
        if self._pool is None or self._closed:
            logger.debug("write_batch called but DB unavailable — no-op")
            return 0

        rows = [
            (
                m.timestamp,
                m.metric_name,
                m.value,
                json.dumps({"host": m.host, "service": m.service, "unit": m.unit}),
            )
            for m in metrics
        ]

        conn = self._get_connection()
        if conn is None:
            return 0

        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    "INSERT INTO metrics (time, metric_name, value, labels) "
                    "VALUES (%s, %s, %s, %s)",
                    rows,
                    page_size=500,
                )
            conn.commit()
            logger.info("Wrote %d metric rows to TimescaleDB", len(rows))
            return len(rows)
        except Exception as exc:
            logger.warning("write_batch failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return 0
        finally:
            self._return_connection(conn)

    def add_to_buffer(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a metric to the batch buffer. Flushes automatically when batch_size reached.

        This method is thread-safe and ideal for high-throughput scenarios.
        """
        if self._pool is None or self._closed:
            return

        timestamp = timestamp or datetime.utcnow()
        labels_json = json.dumps(labels) if labels else "{}"

        with self._batch_lock:
            self._batch_buffer.append((timestamp, metric_name, value, labels_json))
            if len(self._batch_buffer) >= self._batch_size:
                self._flush_buffer()

    def _flush_buffer(self) -> int:
        """Flush the batch buffer to the database. Called by add_to_buffer or close."""
        if not self._batch_buffer:
            return 0

        conn = self._get_connection()
        if conn is None:
            return 0

        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    "INSERT INTO metrics (time, metric_name, value, labels) "
                    "VALUES (%s, %s, %s, %s)",
                    self._batch_buffer,
                    page_size=500,
                )
            conn.commit()
            count = len(self._batch_buffer)
            logger.info("Flushed %d metrics to TimescaleDB", count)
            self._batch_buffer = []
            return count
        except Exception as exc:
            logger.warning("Buffer flush failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return 0
        finally:
            self._return_connection(conn)

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Query metrics for a specific metric name within a time range.

        Args:
            metric_name: Name of the metric to query
            start_time: Start of the time range
            end_time: End of the time range
            labels_filter: Optional labels to filter by

        Returns:
            List of metric dicts with time, metric_name, value, labels
        """
        if self._pool is None or self._closed:
            logger.debug("query_metrics: DB unavailable — returning empty list")
            return []

        sql = (
            "SELECT time, metric_name, value, labels FROM metrics "
            "WHERE metric_name = %s AND time >= %s AND time <= %s "
            "ORDER BY time"
        )
        params = [metric_name, start_time, end_time]

        conn = self._get_connection()
        if conn is None:
            return []

        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                results = []
                for row in cur.fetchall():
                    record = dict(zip(cols, row))
                    if labels_filter:
                        record_labels = (
                            json.loads(record["labels"])
                            if isinstance(record["labels"], str)
                            else record["labels"]
                        )
                        if not all(
                            record_labels.get(k) == v for k, v in labels_filter.items()
                        ):
                            continue
                    results.append(record)
                return results
        except Exception as exc:
            logger.warning("query_metrics failed: %s", exc)
            return []
        finally:
            self._return_connection(conn)

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
        if self._pool is None or self._closed or not PANDAS_AVAILABLE:
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
            conditions.append("labels->>'service' = %s")
            params.append(service)

        where = " AND ".join(conditions)
        sql = (
            f"SELECT time, metric_name, value FROM metrics WHERE {where} ORDER BY time"
        )

        conn = self._get_connection()
        if conn is None:
            return pd.DataFrame() if PANDAS_AVAILABLE else []

        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=["time", "metric_name", "value"])
            if df.empty:
                return df
            df = df.pivot_table(
                index="time", columns="metric_name", values="value", aggfunc="mean"
            )
            return df
        except Exception as exc:
            logger.warning("query_range failed: %s", exc)
            return pd.DataFrame()
        finally:
            self._return_connection(conn)

    def create_hypertable(self) -> bool:
        """Create the metrics hypertable if it doesn't exist.

        Returns True on success or if already exists, False on failure.
        """
        if self._pool is None or self._closed:
            return False

        conn = self._get_connection()
        if conn is None:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = 'metrics'"
                )
                if not cur.fetchone():
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS metrics (
                            time        TIMESTAMPTZ NOT NULL,
                            metric_name TEXT NOT NULL,
                            value       DOUBLE PRECISION,
                            labels      JSONB DEFAULT '{}'
                        )
                    """)
                    cur.execute(
                        "SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics (metric_name, time DESC)"
                    )
                    conn.commit()
                    logger.info("Created metrics hypertable and indexes")
            return True
        except Exception as exc:
            logger.warning("create_hypertable failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            self._return_connection(conn)

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
        if self._pool is None or self._closed:
            logger.debug("write_event called but DB unavailable — no-op")
            return False

        conn = self._get_connection()
        if conn is None:
            return False

        sql = (
            "INSERT INTO events (event_type, service, payload, correlation_id) "
            "VALUES (%s, %s, %s, %s)"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (event_type, service, json.dumps(payload), correlation_id)
                )
            conn.commit()
            return True
        except Exception as exc:
            logger.warning("write_event failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            self._return_connection(conn)

    def query_events(self, start: datetime, end: datetime) -> List[dict]:
        """Return events in [start, end] as a list of dicts."""
        if self._pool is None or self._closed:
            logger.debug("query_events: DB unavailable — returning empty list")
            return []

        conn = self._get_connection()
        if conn is None:
            return []

        sql = (
            "SELECT id, time, event_type, service, payload, correlation_id "
            "FROM events WHERE time >= %s AND time <= %s ORDER BY time"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (start, end))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("query_events failed: %s", exc)
            return []
        finally:
            self._return_connection(conn)

    def get_normalization_params(
        self, metric_name: str, service: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return (mean, std) for the given metric/service or (None, None)."""
        if self._pool is None or self._closed:
            return (None, None)

        conn = self._get_connection()
        if conn is None:
            return (None, None)

        sql = (
            "SELECT rolling_mean, rolling_std FROM normalization_params "
            "WHERE metric_name = %s AND service = %s"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (metric_name, service))
                row = cur.fetchone()
            if row:
                return (row[0], row[1])
            return (None, None)
        except Exception as exc:
            logger.warning("get_normalization_params failed: %s", exc)
            return (None, None)
        finally:
            self._return_connection(conn)

    def update_normalization_params(
        self,
        metric_name: str,
        service: str,
        mean: float,
        std: float,
        window_days: int = 7,
    ) -> bool:
        """Upsert normalization params for a metric/service pair."""
        if self._pool is None or self._closed:
            logger.debug("update_normalization_params: DB unavailable — no-op")
            return False

        conn = self._get_connection()
        if conn is None:
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
            with conn.cursor() as cur:
                cur.execute(sql, (metric_name, service, mean, std, window_days))
            conn.commit()
            return True
        except Exception as exc:
            logger.warning("update_normalization_params failed: %s", exc)
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            self._return_connection(conn)

    def health_check(self) -> bool:
        """Return True if the database connection pool is healthy."""
        if self._pool is None or self._closed:
            return False
        conn = self._get_connection()
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False
        finally:
            self._return_connection(conn)

    def close(self):
        """Close the connection pool and flush any pending metrics."""
        if self._closed:
            return

        self._closed = True
        self._flush_buffer()

        if self._pool:
            try:
                self._pool.closeall()
                logger.info("TimescaleDB connection pool closed")
            except Exception as exc:
                logger.warning("Error closing connection pool: %s", exc)
            self._pool = None
