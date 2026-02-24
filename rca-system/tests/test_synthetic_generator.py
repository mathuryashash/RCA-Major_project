"""
Unit tests for the Synthetic Data Generator.

Run with: pytest tests/test_synthetic_generator.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator


class TestSyntheticMetricsGenerator:

    @pytest.fixture
    def generator(self):
        return SyntheticMetricsGenerator(seed=42)

    def test_normal_data_shape(self, generator):
        df = generator.generate_normal_behavior(duration_days=10)
        expected_samples = (10 * 24 * 60) // 5  # 10 days at 5-min intervals
        assert df.shape[0] == expected_samples
        assert df.shape[1] == 10  # 10 metric columns

    def test_normal_data_has_datetime_index(self, generator):
        df = generator.generate_normal_behavior(duration_days=5)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_normal_data_columns(self, generator):
        df = generator.generate_normal_behavior(duration_days=5)
        expected_cols = [
            'cpu_utilization', 'memory_usage_percent',
            'api_latency_p50_ms', 'api_latency_p95_ms', 'api_latency_p99_ms',
            'error_rate_percent', 'db_connections_active', 'cache_hit_rate',
            'request_throughput', 'disk_io_wait_percent'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_normal_data_ranges_cpu(self, generator):
        df = generator.generate_normal_behavior(duration_days=10)
        assert df['cpu_utilization'].between(0, 100).all(), "CPU out of range [0, 100]"

    def test_normal_data_ranges_memory(self, generator):
        df = generator.generate_normal_behavior(duration_days=10)
        assert df['memory_usage_percent'].between(0, 100).all()

    def test_normal_data_no_nans(self, generator):
        df = generator.generate_normal_behavior(duration_days=10)
        assert not df.isna().any().any(), "Normal data should not contain NaN"

    def test_failure_injection_database_slow_query(self, generator):
        normal = generator.generate_normal_behavior(duration_days=10)
        n = len(normal)
        failure, meta = generator.inject_failure_scenario(
            normal, 'database_slow_query', start_idx=n // 2, duration_samples=100
        )
        # Shape must be unchanged
        assert failure.shape == normal.shape
        # Metadata
        assert meta['failure_type'] == 'database_slow_query'
        assert meta['root_cause'] is not None
        assert len(meta['causal_chain']) > 0
        # Latency must have increased
        before_latency = normal.iloc[n // 2]['api_latency_p95_ms']
        after_latency = failure.iloc[n // 2 + 90]['api_latency_p95_ms']
        assert after_latency > before_latency * 2, "Latency should increase substantially"

    def test_failure_injection_memory_leak(self, generator):
        normal = generator.generate_normal_behavior(duration_days=10)
        n = len(normal)
        failure, meta = generator.inject_failure_scenario(
            normal, 'memory_leak', start_idx=n // 2, duration_samples=100
        )
        mid = n // 2
        # Memory at end should be significantly higher than at start of failure
        mem_start = failure.iloc[mid]['memory_usage_percent']
        mem_end   = failure.iloc[mid + 90]['memory_usage_percent']
        assert mem_end > mem_start, "Memory should increase during memory leak scenario"

    def test_failure_injection_all_types(self, generator):
        normal = generator.generate_normal_behavior(duration_days=10)
        n = len(normal)
        failure_types = [
            'database_slow_query', 'memory_leak', 'network_partition',
            'thread_pool_exhaustion', 'cpu_spike', 'disk_io_spike'
        ]
        for ft in failure_types:
            failure, meta = generator.inject_failure_scenario(
                normal, ft, start_idx=n // 2, duration_samples=50
            )
            assert failure.shape == normal.shape, f"Shape mismatch for {ft}"
            assert meta['failure_type'] == ft

    def test_invalid_failure_type_raises(self, generator):
        normal = generator.generate_normal_behavior(duration_days=5)
        with pytest.raises(ValueError, match="Unknown failure_type"):
            generator.inject_failure_scenario(normal, 'invalid_type', 100, 50)

    def test_invalid_start_idx_raises(self, generator):
        normal = generator.generate_normal_behavior(duration_days=5)
        with pytest.raises(ValueError, match="out of range"):
            generator.inject_failure_scenario(normal, 'cpu_spike', len(normal) + 100, 50)

    def test_evaluation_dataset_length(self, generator):
        scenarios = generator.generate_evaluation_dataset(
            num_scenarios=6, normal_days=5, failure_duration_samples=50
        )
        assert len(scenarios) == 6

    def test_evaluation_dataset_structure(self, generator):
        scenarios = generator.generate_evaluation_dataset(num_scenarios=3, normal_days=5)
        for s in scenarios:
            assert 'scenario_id' in s
            assert 'failure_type' in s
            assert 'normal_data' in s
            assert 'failure_data' in s
            assert 'metadata' in s

    def test_reproducibility_with_seed(self):
        g1 = SyntheticMetricsGenerator(seed=123)
        g2 = SyntheticMetricsGenerator(seed=123)
        df1 = g1.generate_normal_behavior(duration_days=5)
        df2 = g2.generate_normal_behavior(duration_days=5)
        assert df1.equals(df2), "Same seed should produce identical data"

    def test_different_seeds_differ(self):
        g1 = SyntheticMetricsGenerator(seed=1)
        g2 = SyntheticMetricsGenerator(seed=2)
        df1 = g1.generate_normal_behavior(duration_days=5)
        df2 = g2.generate_normal_behavior(duration_days=5)
        assert not df1.equals(df2), "Different seeds should produce different data"


class TestDataPreprocessor:
    """Tests for the DataPreprocessor (basic coverage)."""

    @pytest.fixture
    def normal_df(self):
        gen = SyntheticMetricsGenerator(seed=42)
        return gen.generate_normal_behavior(duration_days=10)

    def test_fit_transform_shape(self, normal_df):
        from src.preprocessing.data_normalizer import DataPreprocessor
        pp = DataPreprocessor(window_size=30, stride=5)
        arr = pp.fit_transform(normal_df)
        assert arr.shape == (len(normal_df), len(normal_df.columns))

    def test_create_windows_shape(self, normal_df):
        from src.preprocessing.data_normalizer import DataPreprocessor
        pp = DataPreprocessor(window_size=60, stride=10)
        pp.fit(normal_df)
        windows, indices, timestamps = pp.create_windows_from_df(normal_df)
        assert windows.ndim == 3
        assert windows.shape[1] == 60
        assert windows.shape[2] == len(normal_df.columns)

    def test_quality_report_structure(self, normal_df):
        from src.preprocessing.data_normalizer import DataPreprocessor
        pp = DataPreprocessor()
        report = pp.validate_data_quality(normal_df)
        assert 'quality_score' in report
        assert 'issues' in report
        assert 'missing_pct' in report
        assert report['quality_score'] > 80  # Clean synthetic data should score high

    def test_not_fitted_raises(self):
        from src.preprocessing.data_normalizer import DataPreprocessor
        from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator
        pp = DataPreprocessor()
        gen = SyntheticMetricsGenerator()
        df = gen.generate_normal_behavior(duration_days=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.transform(df)
