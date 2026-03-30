"""
conftest.py — shared fixtures for system tests.
"""

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def patch_backend_connections():
    """Patch TimescaleDB and Redis so PipelineOrchestrator init doesn't hang.
    Only patches psycopg2/redis drivers — HuggingFace downloads (for LogBERT)
    are unaffected.
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
