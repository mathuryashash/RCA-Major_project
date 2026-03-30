"""
conftest.py — shared fixtures for integration tests.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_backend_connections():
    """Patch TimescaleDB and Redis connection attempts so PipelineOrchestrator
    init doesn't hang.  We patch at the driver level (psycopg2.connect and
    redis.Redis) so that HTTP traffic (e.g. HuggingFace model downloads for
    LogBERT) is unaffected.
    """
    with (
        patch(
            "src.ingestion.timescaledb_store.psycopg2.connect",
            side_effect=Exception("Connection refused"),
        ),
        patch(
            "src.ingestion.timescaledb_store.psycopg2.pool.ThreadedConnectionPool",
            side_effect=Exception("Connection refused"),
        ),
        patch(
            "src.ingestion.redis_buffer.redis.Redis",
            side_effect=Exception("Connection refused"),
        ),
    ):
        yield
