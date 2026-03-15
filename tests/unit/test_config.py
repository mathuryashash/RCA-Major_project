"""
Unit tests for src.common.config.load_config
"""

import os
import pytest
import yaml

from src.common.config import load_config


@pytest.mark.unit
class TestLoadConfig:
    def test_load_default_config(self, config_dir):
        """Load the real config/config.yaml and verify it returns a dict with expected keys."""
        config_path = os.path.join(config_dir, "config.yaml")
        result = load_config(config_path)

        assert result is not None
        assert isinstance(result, dict)
        expected_keys = {
            "log_sources",
            "metric_sources",
            "database",
            "anomaly_detection",
            "causal_inference",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_load_nonexistent_file(self):
        """Pass a path that doesn't exist.

        The implementation falls back to <project_root>/config/config.yaml based
        on the module's __file__ location.  When the fallback file exists (as it
        does in this repo), the function returns the real config dict rather than
        None.  We verify the fallback produces a valid dict.
        """
        result = load_config("/totally/bogus/path/no_such_file.yaml")
        # Fallback returns the real config – still a valid dict
        assert isinstance(result, dict)
        assert "log_sources" in result

    def test_load_custom_yaml(self, tmp_dir):
        """Write a temporary YAML file, load it, and verify its contents."""
        custom_data = {
            "service": "test-app",
            "version": 2,
            "features": ["a", "b", "c"],
        }
        custom_path = os.path.join(tmp_dir, "custom.yaml")
        with open(custom_path, "w", encoding="utf-8") as f:
            yaml.dump(custom_data, f)

        result = load_config(custom_path)

        assert result is not None
        assert result["service"] == "test-app"
        assert result["version"] == 2
        assert result["features"] == ["a", "b", "c"]

    def test_config_has_log_sources(self, config_dir):
        """Verify config contains 'log_sources' key with a non-empty list."""
        config_path = os.path.join(config_dir, "config.yaml")
        result = load_config(config_path)

        assert "log_sources" in result
        assert isinstance(result["log_sources"], list)
        assert len(result["log_sources"]) > 0

        # Each entry should have label, path, format
        for source in result["log_sources"]:
            assert "label" in source
            assert "path" in source
            assert "format" in source
