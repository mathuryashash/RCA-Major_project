"""
Synthetic Metrics Generator

Generates realistic system metrics with known failure patterns
for testing and validating the RCA pipeline.

This is critical for development without real production data —
it provides controlled experiments with known ground truth.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class SyntheticMetricsGenerator:
    """
    Generates synthetic time-series metrics for a microservice system
    with realistic correlations and injectable failure scenarios.

    Supports the following failure types:
    - 'database_slow_query': Gradually increasing query latency -> connection exhaustion -> error spike
    - 'memory_leak': Monotonic memory increase -> GC pressure -> OOM
    - 'network_partition': Sudden spike in latency and errors
    - 'thread_pool_exhaustion': API latency spike + DB connection spike
    - 'cpu_spike': Sudden CPU utilization increase
    - 'disk_io_spike': I/O wait surge affecting latency
    """

    def __init__(self, num_services: int = 5, sampling_interval_min: int = 5, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        self.num_services = num_services
        self.interval_min = sampling_interval_min
        self.services = [f"service_{i}" for i in range(num_services)]

        # Define inter-service dependencies (adjacency matrix)
        # service_0 -> service_1 -> service_2 (linear chain)
        # service_0 -> service_3 (branch)
        # service_3 -> service_4
        self.dependency_matrix = np.zeros((num_services, num_services))
        if num_services >= 2:
            self.dependency_matrix[0, 1] = 1  # service_0 causes service_1
        if num_services >= 3:
            self.dependency_matrix[1, 2] = 1  # service_1 causes service_2
        if num_services >= 4:
            self.dependency_matrix[0, 3] = 1  # service_0 causes service_3
        if num_services >= 5:
            self.dependency_matrix[3, 4] = 1  # service_3 causes service_4

    # ------------------------------------------------------------------
    # Normal metric generation helpers
    # ------------------------------------------------------------------

    def _generate_cpu_normal(self, num_samples: int) -> np.ndarray:
        """CPU follows diurnal pattern: low at night, peaks during business hours."""
        time_of_day = np.arange(num_samples) % (288)  # 24h / 5min intervals
        base = 40 + 20 * np.sin(2 * np.pi * time_of_day / 288)
        noise = np.random.normal(0, 3, num_samples)
        return np.clip(base + noise, 5, 95)

    def _generate_memory_normal(self, num_samples: int) -> np.ndarray:
        """Memory with slow oscillations simulating garbage collection cycles."""
        base = 60 + 0.005 * np.arange(num_samples)  # slight upward trend
        gc_pattern = np.sin(2 * np.pi * np.arange(num_samples) / 100) * 5
        noise = np.random.normal(0, 2, num_samples)
        return np.clip(base + gc_pattern + noise, 30, 95)

    def _generate_latency_p50(self, num_samples: int) -> np.ndarray:
        """P50 latency baseline: 40–70ms."""
        return np.clip(50 + np.random.normal(0, 5, num_samples), 20, 100)

    def _generate_latency_p95(self, num_samples: int) -> np.ndarray:
        """P95 approximately 2–3x P50."""
        p50 = self._generate_latency_p50(num_samples)
        return np.clip(p50 * 2.5 + np.random.normal(0, 10, num_samples), 50, 500)

    def _generate_latency_p99(self, num_samples: int) -> np.ndarray:
        """P99 approximately 5–10x P50."""
        p50 = self._generate_latency_p50(num_samples)
        return np.clip(p50 * 7 + np.random.normal(0, 20, num_samples), 100, 2000)

    def _generate_error_rate_normal(self, num_samples: int) -> np.ndarray:
        """Very low error rate during normal operation: ~0.5–1.5%."""
        return np.abs(np.random.normal(0.01, 0.005, num_samples))

    def _generate_db_connections(self, num_samples: int) -> np.ndarray:
        """Database connections following business hour patterns."""
        time_of_day = np.arange(num_samples) % 288
        business_hours = (time_of_day > 108) & (time_of_day < 240)
        base = np.where(business_hours, 65, 25)
        return np.clip(base + np.random.normal(0, 5, num_samples), 5, 100)

    def _generate_cache_hit_rate(self, num_samples: int) -> np.ndarray:
        """Cache hit rate typically 85–95%."""
        return np.clip(90 + np.random.normal(0, 3, num_samples), 75, 99)

    def _generate_throughput(self, num_samples: int) -> np.ndarray:
        """Request throughput: diurnal pattern, 500–1500 req/s."""
        time_of_day = np.arange(num_samples) % 288
        base = 1000 + 500 * np.sin(2 * np.pi * time_of_day / 288)
        return np.clip(base + np.random.normal(0, 50, num_samples), 100, 2000)

    def _generate_disk_io(self, num_samples: int) -> np.ndarray:
        """Disk I/O wait percentage: normally 1–10%."""
        return np.clip(np.random.normal(5, 2, num_samples), 1, 15)

    # ------------------------------------------------------------------
    # Main generation methods
    # ------------------------------------------------------------------

    def generate_normal_behavior(
        self,
        duration_days: int = 60,
        sampling_interval_minutes: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate N days of normal operational metrics.

        Returns a DataFrame with columns:
            timestamp, cpu_utilization, memory_usage_percent,
            api_latency_p50_ms, api_latency_p95_ms, api_latency_p99_ms,
            error_rate_percent, db_connections_active, cache_hit_rate,
            request_throughput, disk_io_wait_percent
        """
        interval = sampling_interval_minutes or self.interval_min
        num_samples = (duration_days * 24 * 60) // interval

        timestamps = pd.date_range(
            start='2024-01-01',
            periods=num_samples,
            freq=f'{interval}min'
        )

        data = {
            'timestamp': timestamps,
            'cpu_utilization': self._generate_cpu_normal(num_samples),
            'memory_usage_percent': self._generate_memory_normal(num_samples),
            'api_latency_p50_ms': self._generate_latency_p50(num_samples),
            'api_latency_p95_ms': self._generate_latency_p95(num_samples),
            'api_latency_p99_ms': self._generate_latency_p99(num_samples),
            'error_rate_percent': self._generate_error_rate_normal(num_samples),
            'db_connections_active': self._generate_db_connections(num_samples),
            'cache_hit_rate': self._generate_cache_hit_rate(num_samples),
            'request_throughput': self._generate_throughput(num_samples),
            'disk_io_wait_percent': self._generate_disk_io(num_samples),
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def inject_failure_scenario(
        self,
        df: pd.DataFrame,
        failure_type: str,
        start_idx: int,
        duration_samples: int,
        severity: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject a synthetic failure scenario into normal data.

        Args:
            df: Normal baseline DataFrame (from generate_normal_behavior)
            failure_type: One of the supported failure types
            start_idx: Index at which the failure starts
            duration_samples: How many samples the failure lasts
            severity: Multiplier for failure magnitude (0.0–2.0)

        Returns:
            (modified_df, metadata) where metadata contains ground truth:
            {
                'failure_type': str,
                'start_time': timestamp,
                'start_idx': int,
                'duration_samples': int,
                'severity': float,
                'root_cause': str,
                'causal_chain': list[str],
                'expected_anomalous_metrics': list[str]
            }
        """
        df_copy = df.copy()

        if start_idx < 0 or start_idx >= len(df_copy):
            raise ValueError(f"start_idx {start_idx} is out of range [0, {len(df_copy)})")

        end_idx = min(start_idx + duration_samples, len(df_copy))
        actual_duration = end_idx - start_idx

        metadata = {
            'failure_type': failure_type,
            'start_time': df_copy.index[start_idx],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration_samples': actual_duration,
            'severity': severity,
            'root_cause': None,
            'causal_chain': [],
            'expected_anomalous_metrics': []
        }

        if failure_type == 'database_slow_query':
            self._inject_database_slow_query(df_copy, start_idx, actual_duration, severity, metadata)

        elif failure_type == 'memory_leak':
            self._inject_memory_leak(df_copy, start_idx, actual_duration, severity, metadata)

        elif failure_type == 'network_partition':
            self._inject_network_partition(df_copy, start_idx, actual_duration, severity, metadata)

        elif failure_type == 'thread_pool_exhaustion':
            self._inject_thread_pool_exhaustion(df_copy, start_idx, actual_duration, severity, metadata)

        elif failure_type == 'cpu_spike':
            self._inject_cpu_spike(df_copy, start_idx, actual_duration, severity, metadata)

        elif failure_type == 'disk_io_spike':
            self._inject_disk_io_spike(df_copy, start_idx, actual_duration, severity, metadata)

        else:
            raise ValueError(
                f"Unknown failure_type '{failure_type}'. "
                "Valid options: database_slow_query, memory_leak, network_partition, "
                "thread_pool_exhaustion, cpu_spike, disk_io_spike"
            )

        return df_copy, metadata

    # ------------------------------------------------------------------
    # Failure scenario implementations
    # ------------------------------------------------------------------

    def _inject_database_slow_query(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Database schema migration causes query planner to pick inefficient index.

        Causal chain:
        1. Slow database queries (+400% latency)
        2. Connections held longer -> pool exhaustion (~30% through failure)
        3. API request timeouts -> error rate spike (~60% through failure)
        """
        ramp = np.linspace(1.0, 1 + 9 * severity, duration)  # Up to 10x slower

        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('api_latency_p50_ms')] *= ramp[i]
                df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] *= ramp[i]
                df.iloc[idx, df.columns.get_loc('api_latency_p99_ms')] *= ramp[i]

        # Connection pool exhaustion kicks in at ~30% through failure
        pool_exhaust_start = start_idx + int(duration * 0.3)
        for i in range(duration - int(duration * 0.3)):
            idx = pool_exhaust_start + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('db_connections_active')] = (
                    95 + np.random.normal(3, 1) * severity
                )

        # Error rate spikes at ~60% through failure
        error_start = start_idx + int(duration * 0.6)
        for i in range(start_idx + duration - error_start):
            idx = error_start + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('error_rate_percent')] = (
                    10 * severity + np.random.normal(0, 2)
                )

        metadata['root_cause'] = 'Database schema migration causing inefficient index use'
        metadata['causal_chain'] = [
            'schema_migration -> slow_queries',
            'slow_queries -> connection_pool_exhaustion',
            'connection_pool_exhaustion -> api_timeouts',
            'api_timeouts -> high_error_rate'
        ]
        metadata['expected_anomalous_metrics'] = [
            'api_latency_p50_ms', 'api_latency_p95_ms', 'api_latency_p99_ms',
            'db_connections_active', 'error_rate_percent'
        ]

    def _inject_memory_leak(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Unclosed connections/file handles leading to monotonic memory increase.

        Causal chain:
        1. Memory usage increases linearly (leak rate: ~30% of max over duration)
        2. GC pressure causes CPU increase (~70% through failure)
        3. Latency increases due to GC pauses
        """
        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                progress = i / max(duration - 1, 1)
                increase = 30 * progress * severity
                df.iloc[idx, df.columns.get_loc('memory_usage_percent')] = np.clip(
                    df.iloc[idx, df.columns.get_loc('memory_usage_percent')] + increase,
                    0, 99
                )

        # GC pressure causes CPU spike at ~70% through failure
        gc_start = start_idx + int(duration * 0.7)
        for i in range(start_idx + duration - gc_start):
            idx = gc_start + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('cpu_utilization')] = np.clip(
                    df.iloc[idx, df.columns.get_loc('cpu_utilization')] + 20 * severity,
                    0, 99
                )
                df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] *= (1 + 0.5 * severity)

        metadata['root_cause'] = 'Memory leak: unclosed connections not released from pool'
        metadata['causal_chain'] = [
            'buggy_deployment -> connection_leak',
            'connection_leak -> memory_growth',
            'memory_growth -> gc_pressure',
            'gc_pressure -> cpu_spike',
            'gc_pressure -> latency_increase'
        ]
        metadata['expected_anomalous_metrics'] = [
            'memory_usage_percent', 'cpu_utilization', 'api_latency_p95_ms'
        ]

    def _inject_network_partition(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Inter-region network partition causes sudden latency and error spike.

        Causal chain:
        1. Network partition -> immediate latency explosion
        2. Cross-region requests timeout -> high error rate simultaneously
        """
        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] = (
                    5000 * severity + np.random.normal(0, 500)
                )
                df.iloc[idx, df.columns.get_loc('api_latency_p99_ms')] = (
                    7000 * severity + np.random.normal(0, 500)
                )
                df.iloc[idx, df.columns.get_loc('error_rate_percent')] = (
                    40 * severity + np.random.normal(0, 5)
                )

        metadata['root_cause'] = 'Inter-region network partition'
        metadata['causal_chain'] = [
            'network_partition -> cross_region_timeout',
            'cross_region_timeout -> latency_explosion',
            'cross_region_timeout -> error_rate_spike'
        ]
        metadata['expected_anomalous_metrics'] = [
            'api_latency_p95_ms', 'api_latency_p99_ms', 'error_rate_percent'
        ]

    def _inject_thread_pool_exhaustion(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Background job consuming all worker threads, blocking API requests.

        Causal chain:
        1. Background ingestion job uses all threads
        2. API requests queue -> latency spike
        3. DB connections max out (blocking waits)
        4. Eventually errors spike from timeouts
        """
        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('api_latency_p50_ms')] = (
                    1000 * severity + np.random.normal(0, 100)
                )
                df.iloc[idx, df.columns.get_loc('db_connections_active')] = np.clip(
                    99 * severity + np.random.normal(1, 0.5), 0, 100
                )
                # Error rate joins after ~40% through
                if i > int(duration * 0.4):
                    df.iloc[idx, df.columns.get_loc('error_rate_percent')] = (
                        25 * severity + np.random.normal(0, 3)
                    )

        metadata['root_cause'] = 'Background job blocking API thread pool'
        metadata['causal_chain'] = [
            'background_job_started -> thread_pool_exhaustion',
            'thread_pool_exhaustion -> api_latency_spike',
            'thread_pool_exhaustion -> db_connection_saturation',
            'api_latency_spike -> request_timeouts',
            'request_timeouts -> error_rate_spike'
        ]
        metadata['expected_anomalous_metrics'] = [
            'api_latency_p50_ms', 'db_connections_active', 'error_rate_percent'
        ]

    def _inject_cpu_spike(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Runaway process (infinite loop in new deployment) consuming CPU.

        Causal chain:
        1. CPU utilization spikes to 85-95%
        2. Request processing slows down (latency increase)
        """
        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('cpu_utilization')] = np.clip(
                    85 * severity + np.random.normal(0, 3), 0, 99
                )
                df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] = np.clip(
                    df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] * (1 + 1.5 * severity),
                    0, 50000
                )

        metadata['root_cause'] = 'Runaway process consuming CPU (infinite loop in new deployment)'
        metadata['causal_chain'] = [
            'infinite_loop_deployed -> cpu_saturation',
            'cpu_saturation -> request_processing_slowdown',
            'request_processing_slowdown -> api_latency_increase'
        ]
        metadata['expected_anomalous_metrics'] = [
            'cpu_utilization', 'api_latency_p95_ms'
        ]

    def _inject_disk_io_spike(
        self, df: pd.DataFrame, start_idx: int, duration: int, severity: float, metadata: Dict
    ):
        """
        Excessive disk I/O (e.g., logging explosion, full disk scan).

        Causal chain:
        1. Disk I/O wait spikes
        2. Write-heavy workloads block request processing
        3. API latency increases
        """
        for i in range(duration):
            idx = start_idx + i
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc('disk_io_wait_percent')] = np.clip(
                    60 * severity + np.random.normal(0, 5), 0, 99
                )
                if i > int(duration * 0.2):
                    df.iloc[idx, df.columns.get_loc('api_latency_p95_ms')] *= (1 + 2.0 * severity)
                    df.iloc[idx, df.columns.get_loc('api_latency_p99_ms')] *= (1 + 2.0 * severity)

        metadata['root_cause'] = 'Disk I/O spike from logging explosion or disk scan'
        metadata['causal_chain'] = [
            'disk_io_explosion -> io_wait_spike',
            'io_wait_spike -> request_blocking',
            'request_blocking -> api_latency_increase'
        ]
        metadata['expected_anomalous_metrics'] = [
            'disk_io_wait_percent', 'api_latency_p95_ms', 'api_latency_p99_ms'
        ]

    # ------------------------------------------------------------------
    # Batch generation for evaluation
    # ------------------------------------------------------------------

    def generate_evaluation_dataset(
        self,
        num_scenarios: int = 20,
        normal_days: int = 30,
        failure_duration_samples: int = 100
    ) -> List[Dict]:
        """
        Generate a labeled evaluation dataset with multiple failure scenarios.

        Each scenario contains:
        - normal_data: Normal operational baseline
        - failure_data: Data with injected failure
        - metadata: Ground truth labels for evaluation

        Returns:
            List of scenario dicts for use in model evaluation
        """
        failure_types = [
            'database_slow_query',
            'memory_leak',
            'network_partition',
            'thread_pool_exhaustion',
            'cpu_spike',
            'disk_io_spike'
        ]

        scenarios = []
        normal_samples = (normal_days * 24 * 60) // self.interval_min

        for i in range(num_scenarios):
            failure_type = failure_types[i % len(failure_types)]
            severity = np.random.uniform(0.5, 1.5)

            # Regenerate with different seed per scenario for variety
            np.random.seed(self.seed + i)
            normal_data = self.generate_normal_behavior(duration_days=normal_days)

            # Start failure at 80% through the normal period
            start_idx = int(normal_samples * 0.8)
            failure_data, metadata = self.inject_failure_scenario(
                normal_data,
                failure_type=failure_type,
                start_idx=start_idx,
                duration_samples=failure_duration_samples,
                severity=severity
            )

            scenarios.append({
                'scenario_id': i,
                'failure_type': failure_type,
                'normal_data': normal_data,
                'failure_data': failure_data,
                'metadata': metadata,
                'normal_period_end_idx': start_idx
            })

        # Reset seed
        np.random.seed(self.seed)
        return scenarios

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save generated dataset to CSV."""
        df.to_csv(filepath)
        print(f"Dataset saved to {filepath} (shape: {df.shape})")

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load previously saved dataset."""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Dataset loaded from {filepath} (shape: {df.shape})")
        return df


# ---------------------------------------------------------------------------
# Usage example / smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== SyntheticMetricsGenerator Demo ===\n")

    generator = SyntheticMetricsGenerator(seed=42)

    # 1. Generate 60 days of normal data
    normal_data = generator.generate_normal_behavior(duration_days=60)
    print(f"Normal data shape: {normal_data.shape}")
    print(f"Columns: {list(normal_data.columns)}")
    print(f"Date range: {normal_data.index[0]} -> {normal_data.index[-1]}\n")

    # 2. Inject a database slow query failure at day 50
    failure_start = (50 * 24 * 60) // 5  # samples into the series
    failure_data, metadata = generator.inject_failure_scenario(
        normal_data,
        failure_type='database_slow_query',
        start_idx=failure_start,
        duration_samples=200,
        severity=0.8
    )

    print("Failure metadata:")
    print(json.dumps(
        {k: str(v) if not isinstance(v, (str, int, float, list)) else v
         for k, v in metadata.items()},
        indent=2
    ))

    before = normal_data.iloc[failure_start - 10]
    after = failure_data.iloc[failure_start + 180]

    print(f"\nAPI latency P95 BEFORE failure: {before['api_latency_p95_ms']:.1f}ms")
    print(f"API latency P95 AFTER failure:  {after['api_latency_p95_ms']:.1f}ms")
    print(f"Error rate BEFORE failure:       {before['error_rate_percent']:.3f}%")
    print(f"Error rate AFTER failure:        {after['error_rate_percent']:.3f}%")
    print(f"DB connections BEFORE failure:   {before['db_connections_active']:.1f}")
    print(f"DB connections AFTER failure:    {after['db_connections_active']:.1f}")

    # 3. Generate evaluation dataset
    print("\n=== Generating evaluation dataset ===")
    scenarios = generator.generate_evaluation_dataset(num_scenarios=6, normal_days=10)
    print(f"Generated {len(scenarios)} labeled scenarios")
    for s in scenarios:
        print(f"  Scenario {s['scenario_id']}: {s['failure_type']} "
              f"(severity={s['metadata']['severity']:.2f})")
