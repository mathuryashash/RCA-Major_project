"""
Module 1B: Metric Ingestion
Scrapes Prometheus /api/v1/query_range, optionally CloudWatch GetMetricData.
Pushes raw metrics to Redis Streams, persists to TimescaleDB.
"""

import csv
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CloudWatch availability flag
# ---------------------------------------------------------------------------
try:
    import boto3

    CLOUDWATCH_AVAILABLE = True
except ImportError:
    CLOUDWATCH_AVAILABLE = False
    logger.info("boto3 not installed — CloudWatch collection disabled")

# ---------------------------------------------------------------------------
# Default Prometheus queries
# ---------------------------------------------------------------------------
DEFAULT_QUERIES = [
    "cpu_usage_percent",
    "memory_usage_mb",
    "disk_usage_percent",
    "query_latency_p95_ms",
    "active_connections",
    "error_rate_per_min",
]


# ---------------------------------------------------------------------------
# MetricPoint dataclass
# ---------------------------------------------------------------------------
@dataclass
class MetricPoint:
    """A single time-series metric observation."""

    timestamp: datetime
    host: str
    service: str
    metric_name: str
    value: float
    unit: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (ISO-8601 timestamp)."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MetricPoint":
        """Reconstruct from a dict (accepts ISO-8601 timestamp string)."""
        ts = d["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            timestamp=ts,
            host=d["host"],
            service=d["service"],
            metric_name=d["metric_name"],
            value=float(d["value"]),
            unit=d.get("unit", ""),
        )


# ---------------------------------------------------------------------------
# PrometheusCollector
# ---------------------------------------------------------------------------
class PrometheusCollector:
    """Scrapes Prometheus /api/v1/query_range for metric time-series."""

    def __init__(self, prometheus_url: str, scrape_interval: int = 60):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.scrape_interval = scrape_interval

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if Prometheus is reachable (/-/healthy or /api/v1/status/buildinfo)."""
        try:
            resp = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    def scrape(
        self,
        queries: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """Query Prometheus query_range for each metric query.

        Graceful degradation: if Prometheus is unreachable, logs a warning
        and returns an empty list rather than raising.
        """
        queries = queries or DEFAULT_QUERIES
        now = datetime.now(timezone.utc)
        end = end or now
        start = start or (end - timedelta(seconds=self.scrape_interval))

        points: List[MetricPoint] = []

        for query in queries:
            try:
                resp = requests.get(
                    f"{self.prometheus_url}/api/v1/query_range",
                    params={
                        "query": query,
                        "start": start.timestamp(),
                        "end": end.timestamp(),
                        "step": self.scrape_interval,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") != "success":
                    logger.warning(
                        "Prometheus returned non-success for %s: %s",
                        query,
                        data.get("error", "unknown"),
                    )
                    continue

                for result in data.get("data", {}).get("result", []):
                    metric_labels = result.get("metric", {})
                    host = metric_labels.get("instance", "unknown")
                    service = metric_labels.get(
                        "job", metric_labels.get("service", "unknown")
                    )

                    for ts_val in result.get("values", []):
                        ts_epoch, val_str = ts_val
                        points.append(
                            MetricPoint(
                                timestamp=datetime.fromtimestamp(
                                    float(ts_epoch), tz=timezone.utc
                                ),
                                host=host,
                                service=service,
                                metric_name=query,
                                value=float(val_str),
                                unit=_infer_unit(query),
                            )
                        )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Prometheus unreachable at %s — skipping %s",
                    self.prometheus_url,
                    query,
                )
            except Exception as exc:
                logger.warning("Error scraping Prometheus for %s: %s", query, exc)

        return points


# ---------------------------------------------------------------------------
# CloudWatchCollector
# ---------------------------------------------------------------------------
class CloudWatchCollector:
    """Collects metrics from AWS CloudWatch via boto3 GetMetricData.

    Graceful degradation: if boto3 is not installed or AWS credentials are
    not configured, all operations silently return empty results.
    """

    def __init__(self, namespaces: Optional[List[str]] = None):
        self.namespaces = namespaces or ["AWS/EC2", "AWS/RDS"]
        self._client = None

        if CLOUDWATCH_AVAILABLE:
            try:
                self._client = boto3.client("cloudwatch")
                # Quick credential test (list_metrics is very cheap)
                self._client.list_metrics(Namespace=self.namespaces[0], MaxRecords=1)
            except Exception as exc:
                logger.warning("CloudWatch client init failed: %s", exc)
                self._client = None

    # ------------------------------------------------------------------
    def scrape(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """Fetch CloudWatch metrics for configured namespaces.

        Returns an empty list when CloudWatch is unavailable.
        """
        if self._client is None:
            logger.debug("CloudWatch not available — returning empty list")
            return []

        now = datetime.now(timezone.utc)
        end = end or now
        start = start or (end - timedelta(minutes=5))

        points: List[MetricPoint] = []
        metric_map = {
            "AWS/EC2": [
                ("CPUUtilization", "percent"),
                ("NetworkIn", "bytes"),
                ("NetworkOut", "bytes"),
            ],
            "AWS/RDS": [
                ("CPUUtilization", "percent"),
                ("DatabaseConnections", "count"),
                ("ReadLatency", "seconds"),
            ],
        }

        for namespace in self.namespaces:
            for metric_name, unit in metric_map.get(namespace, []):
                try:
                    resp = self._client.get_metric_statistics(
                        Namespace=namespace,
                        MetricName=metric_name,
                        StartTime=start,
                        EndTime=end,
                        Period=60,
                        Statistics=["Average"],
                    )
                    for dp in resp.get("Datapoints", []):
                        points.append(
                            MetricPoint(
                                timestamp=dp["Timestamp"],
                                host=namespace,
                                service=namespace,
                                metric_name=metric_name,
                                value=dp.get("Average", 0.0),
                                unit=unit,
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "CloudWatch error for %s/%s: %s",
                        namespace,
                        metric_name,
                        exc,
                    )

        return points


# ---------------------------------------------------------------------------
# MetricIngestionService
# ---------------------------------------------------------------------------
class MetricIngestionService:
    """Orchestrates one scrape cycle and CSV-based metric loading."""

    def __init__(self, config: dict):
        metric_cfg = config.get("metric_sources", {})
        self.prometheus = PrometheusCollector(
            prometheus_url=metric_cfg.get("prometheus_url", "http://localhost:9090"),
            scrape_interval=metric_cfg.get("scrape_interval_seconds", 60),
        )
        self.cloudwatch = CloudWatchCollector()

    # ------------------------------------------------------------------
    def run_scrape_cycle(self) -> List[MetricPoint]:
        """Run one scrape cycle across all collectors and merge results."""
        points: List[MetricPoint] = []
        points.extend(self.prometheus.scrape())
        points.extend(self.cloudwatch.scrape())
        logger.info("Scrape cycle collected %d metric points", len(points))
        return points

    # ------------------------------------------------------------------
    def load_csv_metrics(self, csv_path: str) -> List[MetricPoint]:
        """Load metrics from a CSV file (for testing / synthetic data).

        Expected columns: timestamp, host, service, metric_name, value, unit
        """
        points: List[MetricPoint] = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row["timestamp"]
                    if isinstance(ts, str):
                        # Try ISO-8601 first, fall back to common format
                        try:
                            ts = datetime.fromisoformat(ts)
                        except ValueError:
                            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    points.append(
                        MetricPoint(
                            timestamp=ts,
                            host=row.get("host", "localhost"),
                            service=row.get("service", "unknown"),
                            metric_name=row["metric_name"],
                            value=float(row["value"]),
                            unit=row.get("unit", ""),
                        )
                    )
        except FileNotFoundError:
            logger.warning("CSV file not found: %s", csv_path)
        except Exception as exc:
            logger.warning("Error loading CSV metrics from %s: %s", csv_path, exc)

        logger.info("Loaded %d metric points from %s", len(points), csv_path)
        return points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer_unit(metric_name: str) -> str:
    """Best-effort unit inference from the metric name suffix."""
    name = metric_name.lower()
    if "percent" in name:
        return "percent"
    if name.endswith("_mb"):
        return "MB"
    if name.endswith("_ms"):
        return "ms"
    if "connections" in name:
        return "count"
    if "rate" in name:
        return "per_min"
    return ""
