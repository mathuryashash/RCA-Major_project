"""
Shared pytest fixtures for the Automated RCA System test suite.
"""

import os
import sys
import tempfile
import shutil

import pytest
import yaml

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def config_dir():
    return os.path.join(PROJECT_ROOT, "config")


@pytest.fixture
def template_dir():
    return os.path.join(PROJECT_ROOT, "templates", "reports")


@pytest.fixture
def data_dir():
    return os.path.join(PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Temp directory for test outputs
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after each test."""
    d = tempfile.mkdtemp(prefix="rca_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_yaml(config_dir):
    """Load the real config.yaml."""
    path = os.path.join(config_dir, "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def remediation_rules(config_dir):
    """Load remediation_rules.yaml."""
    path = os.path.join(config_dir, "remediation_rules.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def safety_rules(config_dir):
    """Load safety_rules.yaml."""
    path = os.path.join(config_dir, "safety_rules.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sample log lines (all 3 formats)
# ---------------------------------------------------------------------------

SAMPLE_PLAINTEXT_LOGS = [
    "2026-03-15 10:00:01 INFO Application started successfully",
    "2026-03-15 10:00:05 INFO Request handled: GET /api/health 200 5ms",
    "2026-03-15 10:05:30 WARN Slow query detected: SELECT * FROM users (487ms)",
    "2026-03-15 10:10:00 ERROR Connection pool at capacity: 100/100 connections active",
    "2026-03-15 10:10:05 ERROR Request timeout: waiting for DB connection (5000ms)",
    "2026-03-15 10:10:30 CRITICAL Service health check failed: error_rate=15.3%",
]

SAMPLE_SYSLOG_LINES = [
    "Mar 15 10:00:01 prod-server-01 kernel: [INFO] System boot complete",
    "Mar 15 10:05:30 prod-server-01 kernel: [WARN] CPU usage spike: java (pid 1234) consuming 95% CPU",
    "Mar 15 10:10:00 prod-server-01 kernel: [ERROR] OOM killer activated: process node (pid 5678) score 950",
]

SAMPLE_JSON_LOGS = [
    '{"timestamp": "2026-03-15T10:00:01", "level": "INFO", "message": "Service initialized"}',
    '{"ts": "2026-03-15T10:05:30", "level": "WARN", "msg": "High memory usage: 1.8GB"}',
    '{"timestamp": "2026-03-15T10:10:00", "level": "ERROR", "message": "Connection refused"}',
]


@pytest.fixture
def plaintext_logs():
    return SAMPLE_PLAINTEXT_LOGS[:]


@pytest.fixture
def syslog_lines():
    return SAMPLE_SYSLOG_LINES[:]


@pytest.fixture
def json_logs():
    return SAMPLE_JSON_LOGS[:]


# ---------------------------------------------------------------------------
# Sample RCA data for NLG tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rca_data():
    """Minimal valid rca_data dict for NLG testing."""
    return {
        "incident_id": "INC-TEST-001",
        "detected_at": "2026-03-15T10:10:00Z",
        "root_cause": "db_migration_applied",
        "confidence": 0.85,
        "anomaly_scores": {"database": 0.92, "application": 0.45},
        "summary": "Database migration caused query plan regression.",
        "causal_chain": [
            {
                "from": "database",
                "to": "application",
                "confidence": 0.88,
                "lag": 120.0,
                "from_time": "2026-03-15T10:00:00Z",
                "to_time": "2026-03-15T10:02:00Z",
            }
        ],
        "evidence": [
            {
                "source": "db.log",
                "detail": "Slow query detected: SELECT * FROM product_variants (487ms)",
                "score": 0.92,
                "timestamp": "2026-03-15T10:05:30Z",
            },
            {
                "source": "app.log",
                "detail": "Request timeout: waiting for DB connection (5000ms)",
                "score": 0.88,
                "timestamp": "2026-03-15T10:10:00Z",
            },
        ],
        "recommendations": [
            {
                "tier": "Tier 1",
                "description": "Flush query plan cache",
                "command": "SELECT pg_stat_reset();",
            },
            {
                "tier": "Tier 2",
                "description": "Roll back DB migration",
                "command": "alembic downgrade -1",
            },
            {
                "tier": "Tier 3",
                "description": "Require EXPLAIN ANALYZE benchmarks before migrations",
            },
        ],
        "prevention": {
            "immediate": ["Add index usage monitoring alert"],
            "short_term": ["Implement pre-migration benchmarks"],
            "long_term": ["Adopt blue/green deployment for migrations"],
        },
    }


# ---------------------------------------------------------------------------
# Generated failure log fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generated_failure_logs(tmp_dir):
    """Generate a single db_migration failure scenario in a temp directory."""
    import subprocess

    labels_dir = os.path.join(tmp_dir, "labels")
    logs_dir = os.path.join(tmp_dir, "logs")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "generate_failures.py"),
            "--scenario",
            "db_migration",
            "--count",
            "1",
            "--seed",
            "42",
            "--output-dir",
            logs_dir,
            "--labels-dir",
            labels_dir,
        ],
        check=True,
        capture_output=True,
    )
    return {"logs_dir": logs_dir, "labels_dir": labels_dir}
