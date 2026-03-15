"""
Unit tests for src.ingestion.log_ingest.LogHandler._parse_line
"""

import os
import json
import pytest

from src.ingestion.log_ingest import LogHandler

# Import sample log fixtures directly for parametrize / direct use
from tests.conftest import SAMPLE_PLAINTEXT_LOGS, SAMPLE_SYSLOG_LINES


def _make_handler(tmp_dir, fmt="plaintext"):
    """Create a LogHandler with a dummy source_config pointing at a temp file."""
    dummy_log = os.path.join(tmp_dir, "test.log")
    # LogHandler.__init__ calls _get_file_size, which needs the path to exist
    with open(dummy_log, "w") as f:
        f.write("")
    source_config = {
        "label": f"test-{fmt}",
        "path": dummy_log,
        "format": fmt,
    }
    return LogHandler(source_config)


@pytest.mark.unit
class TestLogHandlerParseLine:
    def test_parse_plaintext_line(self, tmp_dir):
        """Parse a standard plaintext log line and verify all 4 fields."""
        handler = _make_handler(tmp_dir, "plaintext")
        line = "2026-03-15 10:00:01 INFO Application started"
        result = handler._parse_line(line)

        assert result is not None
        assert result["timestamp"] == "2026-03-15 10:00:01"
        assert result["level"] == "INFO"
        assert result["raw_message"] == "Application started"
        assert result["source_file"] == handler.path

    def test_parse_syslog_line(self, tmp_dir):
        """Parse a syslog-format line and verify fields."""
        handler = _make_handler(tmp_dir, "syslog")
        line = "Mar 15 10:00:01 prod-server-01 kernel: System boot complete"
        result = handler._parse_line(line)

        assert result is not None
        assert result["timestamp"] == "Mar 15 10:00:01"
        assert result["raw_message"] == "System boot complete"
        assert result["source_file"] == handler.path

    def test_parse_json_line(self, tmp_dir):
        """Parse a JSON log line with 'timestamp' and 'message' keys."""
        handler = _make_handler(tmp_dir, "json")
        line = '{"timestamp": "2026-03-15T10:00:01", "level": "INFO", "message": "Started"}'
        result = handler._parse_line(line)

        assert result is not None
        assert result["timestamp"] == "2026-03-15T10:00:01"
        assert result["level"] == "INFO"
        assert result["raw_message"] == "Started"

    def test_parse_json_alt_keys(self, tmp_dir):
        """Parse a JSON log line using alternate keys 'ts' and 'msg'."""
        handler = _make_handler(tmp_dir, "json")
        line = '{"ts": "2026-03-15T10:00:01", "level": "WARN", "msg": "High memory"}'
        result = handler._parse_line(line)

        assert result is not None
        assert result["timestamp"] == "2026-03-15T10:00:01"
        assert result["level"] == "WARN"
        assert result["raw_message"] == "High memory"

    def test_parse_malformed_line(self, tmp_dir):
        """Parsing garbage input should not raise; returns a record gracefully."""
        handler = _make_handler(tmp_dir, "plaintext")
        result = handler._parse_line("!@#$%^&*() totally not a log line")

        # Should not crash — returns a dict (possibly with None timestamp)
        assert result is not None
        assert isinstance(result, dict)

    def test_all_plaintext_fixtures_parse(self, tmp_dir):
        """All SAMPLE_PLAINTEXT_LOGS should parse to records with non-None timestamp."""
        handler = _make_handler(tmp_dir, "plaintext")

        for line in SAMPLE_PLAINTEXT_LOGS:
            result = handler._parse_line(line)
            assert result is not None, f"parse returned None for: {line}"
            assert result["timestamp"] is not None, f"timestamp is None for: {line}"

    def test_all_syslog_fixtures_parse(self, tmp_dir):
        """All SAMPLE_SYSLOG_LINES should parse to records with non-None raw_message."""
        handler = _make_handler(tmp_dir, "syslog")

        for line in SAMPLE_SYSLOG_LINES:
            result = handler._parse_line(line)
            assert result is not None, f"parse returned None for: {line}"
            assert result["raw_message"] is not None, f"raw_message is None for: {line}"
