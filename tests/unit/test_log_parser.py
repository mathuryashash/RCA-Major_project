"""
Unit tests for src.preprocessing.log_parser.LogTemplateExtractor
"""

import os
import re
import pytest

from src.preprocessing.log_parser import LogTemplateExtractor

# Import sample logs from conftest so we can use them directly
from tests.conftest import SAMPLE_PLAINTEXT_LOGS


def _make_extractor(tmp_dir):
    """Create a fresh LogTemplateExtractor with persistence in tmp_dir."""
    path = os.path.join(tmp_dir, "drain3_test.bin")
    return LogTemplateExtractor(persistence_path=path)


def _strip_timestamp_level(line):
    """Strip the leading 'YYYY-MM-DD HH:MM:SS LEVEL ' prefix from a plaintext log."""
    match = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+ (.+)$", line)
    return match.group(1) if match else line


@pytest.mark.unit
class TestLogTemplateExtractor:
    def test_extract_template_returns_dict(self, tmp_dir):
        """extract_template should return a dict with keys: template, cluster_id, is_new."""
        extractor = _make_extractor(tmp_dir)
        result = extractor.extract_template("User 123 failed login from 10.0.0.1")

        assert isinstance(result, dict)
        assert "template" in result
        assert "cluster_id" in result
        assert "is_new" in result

    def test_extract_template_extracts_wildcard(self, tmp_dir):
        """Similar messages differing only in numbers should yield a template with <*> wildcards."""
        extractor = _make_extractor(tmp_dir)
        extractor.extract_template("User 123 failed login from 10.0.0.1")
        result = extractor.extract_template("User 456 failed login from 10.0.0.2")

        assert "<*>" in result["template"], (
            f"Expected wildcard <*> in template, got: {result['template']}"
        )

    def test_first_message_is_new(self, tmp_dir):
        """The very first log message fed to a fresh extractor should be marked is_new=True."""
        extractor = _make_extractor(tmp_dir)
        result = extractor.extract_template("Connection refused to postgres:5432")

        assert result["is_new"] is True

    def test_repeated_message_not_new(self, tmp_dir):
        """Feeding the exact same message twice: the second result should have is_new=False."""
        extractor = _make_extractor(tmp_dir)
        extractor.extract_template("Connection refused to postgres:5432")
        result = extractor.extract_template("Connection refused to postgres:5432")

        assert result["is_new"] is False

    def test_cluster_id_is_int(self, tmp_dir):
        """cluster_id should be an integer."""
        extractor = _make_extractor(tmp_dir)
        result = extractor.extract_template("Service started on port 8080")

        assert isinstance(result["cluster_id"], int)

    def test_multiple_templates(self, tmp_dir):
        """Feeding 3 different message patterns should produce at least 2 distinct cluster_ids."""
        extractor = _make_extractor(tmp_dir)
        messages = [
            "User 123 failed login from 10.0.0.1",
            "Connection refused to postgres:5432",
            "Disk usage at 95 percent on /dev/sda1",
        ]
        cluster_ids = set()
        for msg in messages:
            result = extractor.extract_template(msg)
            cluster_ids.add(result["cluster_id"])

        assert len(cluster_ids) >= 2, (
            f"Expected at least 2 distinct cluster_ids, got {cluster_ids}"
        )

    def test_parse_rate_100_percent(self, tmp_dir):
        """All 6 SAMPLE_PLAINTEXT_LOGS should parse to a non-empty template (PRD: 100% parse rate)."""
        extractor = _make_extractor(tmp_dir)

        for line in SAMPLE_PLAINTEXT_LOGS:
            message = _strip_timestamp_level(line)
            result = extractor.extract_template(message)
            assert result["template"], f"Empty template for message: {message}"

    def test_save_state_creates_file(self, tmp_dir):
        """Calling save_state should create the persistence file on disk."""
        persistence_path = os.path.join(tmp_dir, "subdir", "drain3_state.bin")
        extractor = LogTemplateExtractor(persistence_path=persistence_path)
        extractor.extract_template("Test message for persistence")
        extractor.save_state()

        assert os.path.exists(persistence_path), (
            f"Expected persistence file at {persistence_path}"
        )
