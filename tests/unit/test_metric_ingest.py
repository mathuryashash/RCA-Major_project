"""
Unit tests for Module 1B — Metric Ingestion.

Covers MetricPoint, PrometheusCollector, CloudWatchCollector,
MetricIngestionService, RedisBuffer, and TimescaleDBStore.
All external services are mocked so these tests run without network access.
"""

import csv
import os
import json
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.metric_ingest import (
    MetricPoint,
    PrometheusCollector,
    CloudWatchCollector,
    MetricIngestionService,
    CLOUDWATCH_AVAILABLE,
)
from src.ingestion.redis_buffer import RedisBuffer, REDIS_AVAILABLE
from src.ingestion.timescaledb_store import TimescaleDBStore, PSYCOPG2_AVAILABLE


# -----------------------------------------------------------------------
# 1. MetricPoint dataclass
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestMetricPoint:
    def test_metric_point_creation(self):
        """MetricPoint can be instantiated and fields are accessible."""
        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        mp = MetricPoint(
            timestamp=ts,
            host="prod-01",
            service="api-gateway",
            metric_name="cpu_usage_percent",
            value=78.5,
            unit="percent",
        )
        assert mp.timestamp == ts
        assert mp.host == "prod-01"
        assert mp.service == "api-gateway"
        assert mp.metric_name == "cpu_usage_percent"
        assert mp.value == 78.5
        assert mp.unit == "percent"

    def test_metric_point_to_dict_roundtrip(self):
        """to_dict / from_dict survive a round-trip."""
        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        original = MetricPoint(ts, "h", "s", "m", 1.0, "u")
        restored = MetricPoint.from_dict(original.to_dict())
        assert restored.timestamp == original.timestamp
        assert restored.value == original.value


# -----------------------------------------------------------------------
# 2. PrometheusCollector
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestPrometheusCollector:
    def test_prometheus_collector_init(self):
        """Collector stores URL and interval correctly."""
        pc = PrometheusCollector("http://prom:9090", scrape_interval=30)
        assert pc.prometheus_url == "http://prom:9090"
        assert pc.scrape_interval == 30

    def test_prometheus_health_check_unreachable(self):
        """health_check returns False when Prometheus cannot be reached."""
        pc = PrometheusCollector("http://unreachable:9090")
        with patch(
            "src.ingestion.metric_ingest.requests.get", side_effect=ConnectionError
        ):
            assert pc.health_check() is False

    def test_prometheus_scrape_unreachable(self):
        """scrape returns an empty list when Prometheus is unreachable."""
        pc = PrometheusCollector("http://unreachable:9090")
        import requests as req_mod

        with patch(
            "src.ingestion.metric_ingest.requests.get",
            side_effect=req_mod.exceptions.ConnectionError,
        ):
            result = pc.scrape()
            assert result == []

    def test_prometheus_scrape_parses_response(self):
        """scrape correctly parses a well-formed Prometheus JSON response."""
        pc = PrometheusCollector("http://prom:9090", scrape_interval=60)

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.raise_for_status = MagicMock()
        fake_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"instance": "host-1", "job": "web"},
                        "values": [
                            [1742036400.0, "55.2"],
                            [1742036460.0, "60.1"],
                        ],
                    }
                ],
            },
        }

        with patch(
            "src.ingestion.metric_ingest.requests.get", return_value=fake_response
        ):
            points = pc.scrape(queries=["cpu_usage_percent"])

        assert len(points) == 2
        assert points[0].host == "host-1"
        assert points[0].service == "web"
        assert points[0].metric_name == "cpu_usage_percent"
        assert points[0].value == 55.2


# -----------------------------------------------------------------------
# 3. CloudWatchCollector
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestCloudWatchCollector:
    def test_cloudwatch_graceful_degradation(self):
        """CloudWatchCollector.scrape does not crash when boto3 is absent."""
        # Force _client to None regardless of real boto3 availability
        cw = CloudWatchCollector.__new__(CloudWatchCollector)
        cw.namespaces = ["AWS/EC2"]
        cw._client = None

        result = cw.scrape()
        assert result == []


# -----------------------------------------------------------------------
# 4. MetricIngestionService — CSV loading
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestMetricIngestionService:
    def test_ingestion_service_csv_load(self, tmp_dir):
        """load_csv_metrics correctly reads a well-formed CSV."""
        csv_path = os.path.join(tmp_dir, "metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "host",
                    "service",
                    "metric_name",
                    "value",
                    "unit",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": "2026-03-15T10:00:00+00:00",
                    "host": "prod-01",
                    "service": "web",
                    "metric_name": "cpu_usage_percent",
                    "value": "72.5",
                    "unit": "percent",
                }
            )
            writer.writerow(
                {
                    "timestamp": "2026-03-15T10:01:00+00:00",
                    "host": "prod-01",
                    "service": "web",
                    "metric_name": "memory_usage_mb",
                    "value": "1024.0",
                    "unit": "MB",
                }
            )

        config = {
            "metric_sources": {
                "prometheus_url": "http://localhost:9090",
                "scrape_interval_seconds": 60,
            }
        }
        svc = MetricIngestionService(config)
        points = svc.load_csv_metrics(csv_path)

        assert len(points) == 2
        assert points[0].metric_name == "cpu_usage_percent"
        assert points[0].value == 72.5
        assert points[1].metric_name == "memory_usage_mb"
        assert points[1].value == 1024.0

    def test_ingestion_service_csv_missing_file(self, tmp_dir):
        """load_csv_metrics returns empty list for a missing file."""
        config = {
            "metric_sources": {
                "prometheus_url": "http://localhost:9090",
                "scrape_interval_seconds": 60,
            }
        }
        svc = MetricIngestionService(config)
        result = svc.load_csv_metrics(os.path.join(tmp_dir, "nonexistent.csv"))
        assert result == []


# -----------------------------------------------------------------------
# 5. RedisBuffer graceful degradation
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestRedisBuffer:
    def test_redis_buffer_graceful_degradation(self):
        """RedisBuffer does not crash when Redis is unavailable."""
        # Patch redis.Redis to raise on connect so the buffer falls back
        if REDIS_AVAILABLE:
            with patch(
                "src.ingestion.redis_buffer.redis.Redis",
                side_effect=Exception("Connection refused"),
            ):
                buf = RedisBuffer(host="unreachable", port=9999)
        else:
            # redis package not installed — constructor itself is a no-op
            buf = RedisBuffer(host="unreachable", port=9999)

        assert buf.health_check() is False

        # push / consume should be no-ops returning safe defaults
        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        dummy = MetricPoint(ts, "h", "s", "m", 1.0, "u")
        assert buf.push([dummy]) == 0
        assert buf.consume() == []


# -----------------------------------------------------------------------
# 6. TimescaleDBStore graceful degradation
# -----------------------------------------------------------------------


@pytest.mark.unit
class TestTimescaleDBStore:
    def test_timescaledb_store_graceful_degradation(self):
        """TimescaleDBStore does not crash when DB is unavailable."""
        if PSYCOPG2_AVAILABLE:
            with patch(
                "src.ingestion.timescaledb_store.psycopg2.connect",
                side_effect=Exception("Connection refused"),
            ):
                store = TimescaleDBStore(host="unreachable", port=9999)
        else:
            store = TimescaleDBStore(host="unreachable", port=9999)

        assert store.health_check() is False

        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        dummy = MetricPoint(ts, "h", "s", "m", 1.0, "u")
        assert store.write_batch([dummy]) == 0
        assert store.query_events(ts, ts) == []
        assert store.get_normalization_params("m", "s") == (None, None)
        assert store.write_event("deploy", "s", {"k": "v"}) is False
        assert store.update_normalization_params("m", "s", 0.0, 1.0) is False
