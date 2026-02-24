import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class SyntheticMetricsGenerator:
    """
    Generates realistic system metrics with known failure patterns
    to validate the Root Cause Analysis (RCA) pipeline.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_normal_behavior(
        self, 
        duration_days: int = 60,
        sampling_interval_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Generate normal operational metrics over a specified duration.
        """
        num_samples = (duration_days * 24 * 60) // sampling_interval_minutes
        
        # We start looking back from roughly "now" (simulated past)
        # Using a fixed start date for reproducibility in tests
        timestamps = pd.date_range(
            start='2024-01-01', 
            periods=num_samples, 
            freq=f'{sampling_interval_minutes}min'
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
        
        return pd.DataFrame(data)
    
    def _generate_cpu_normal(self, num_samples: int) -> np.ndarray:
        time_of_day = np.arange(num_samples) % 288  # 24h / 5min
        base = 40 + 20 * np.sin(2 * np.pi * time_of_day / 288)
        noise = np.random.normal(0, 3, num_samples)
        return np.clip(base + noise, 5, 95)
    
    def _generate_memory_normal(self, num_samples: int) -> np.ndarray:
        # Slow upward trend broken by GC drops
        base = 60 + 0.01 * np.arange(num_samples)
        gc_pattern = np.sin(2 * np.pi * np.arange(num_samples) / 100) * 5
        noise = np.random.normal(0, 2, num_samples)
        return np.clip(base + gc_pattern + noise, 30, 95)
    
    def _generate_latency_p50(self, num_samples: int) -> np.ndarray:
        return np.clip(50 + np.random.normal(0, 5, num_samples), 20, 100)
    
    def _generate_latency_p95(self, num_samples: int) -> np.ndarray:
        p50 = self._generate_latency_p50(num_samples)
        return p50 * 2.5 + np.random.normal(0, 10, num_samples)
    
    def _generate_latency_p99(self, num_samples: int) -> np.ndarray:
        p50 = self._generate_latency_p50(num_samples)
        return p50 * 7 + np.random.normal(0, 20, num_samples)
    
    def _generate_error_rate_normal(self, num_samples: int) -> np.ndarray:
        return np.abs(np.random.normal(0.01, 0.005, num_samples))
    
    def _generate_db_connections(self, num_samples: int) -> np.ndarray:
        time_of_day = np.arange(num_samples) % 288
        business_hours = (time_of_day > 108) & (time_of_day < 240)
        base = np.where(business_hours, 65, 25)
        noise = np.random.normal(0, 5, num_samples)
        return np.clip(base + noise, 5, 100)
    
    def _generate_cache_hit_rate(self, num_samples: int) -> np.ndarray:
        return 90 + np.random.normal(0, 3, num_samples)
    
    def _generate_throughput(self, num_samples: int) -> np.ndarray:
        time_of_day = np.arange(num_samples) % 288
        base = 1000 + 500 * np.sin(2 * np.pi * time_of_day / 288)
        noise = np.random.normal(0, 50, num_samples)
        return np.clip(base + noise, 100, 2000)
    
    def _generate_disk_io(self, num_samples: int) -> np.ndarray:
        return np.clip(np.random.normal(5, 2, num_samples), 1, 15)
    
    def inject_failure_scenario(
        self, 
        df: pd.DataFrame, 
        failure_type: str,
        start_idx: int,
        duration_samples: int,
        severity: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Injects a known failure condition into the normal metrics.
        Returns the modified DataFrame and the ground truth metadata.
        """
        df_copy = df.copy()
        
        # Cap the duration if it exceeds the end of the dataframe
        end_idx = min(start_idx + duration_samples, len(df_copy))
        duration_samples = end_idx - start_idx
        
        metadata = {
            'failure_type': failure_type,
            'start_time': df_copy.iloc[start_idx]['timestamp'],
            'start_idx': start_idx,
            'duration_samples': duration_samples,
            'severity': severity,
            'root_cause': None,
            'causal_chain': []
        }
        
        if failure_type == 'database_slow_query':
            for i in range(duration_samples):
                progress = i / duration_samples
                multiplier = 1 + (9 * progress * severity)  # Max 10x
                idx = start_idx + i
                df_copy.loc[idx, 'api_latency_p50_ms'] *= multiplier
                df_copy.loc[idx, 'api_latency_p95_ms'] *= multiplier
                df_copy.loc[idx, 'api_latency_p99_ms'] *= multiplier
            
            # Connection pool exhaustion happens 30% into the failure
            pool_exhaust_idx = start_idx + int(duration_samples * 0.3)
            if pool_exhaust_idx < end_idx:
                for i in range(pool_exhaust_idx, end_idx):
                    df_copy.loc[i, 'db_connections_active'] = 95 + np.random.normal(3, 1)
            
            # Errors spike 60% into the failure
            error_spike_idx = start_idx + int(duration_samples * 0.6)
            if error_spike_idx < end_idx:
                for i in range(error_spike_idx, end_idx):
                    df_copy.loc[i, 'error_rate_percent'] = 10 + np.random.normal(0, 2)
            
            metadata['root_cause'] = 'Database schema migration causing inefficient index'
            metadata['causal_chain'] = [
                'Slow database queries',
                'Connection pool exhaustion',
                'API timeouts',
                'High error rate'
            ]
            
        elif failure_type == 'memory_leak':
            for i in range(duration_samples):
                progress = i / duration_samples
                increase = 30 * progress * severity # 30% increase
                idx = start_idx + i
                df_copy.loc[idx, 'memory_usage_percent'] += increase
            
            metadata['root_cause'] = 'Unclosed file handles in new code deployment'
            metadata['causal_chain'] = [
                'Leaked connections accumulating',
                'Memory pressure increasing',
                'Garbage collection overhead',
                'OOM kill (eventual)'
            ]
            
        elif failure_type == 'cpu_spike':
            for i in range(duration_samples):
                idx = start_idx + i
                df_copy.loc[idx, 'cpu_utilization'] = 85 + np.random.normal(0, 3)
                df_copy.loc[idx, 'api_latency_p95_ms'] *= 1.5
            
            metadata['root_cause'] = 'Runaway process consuming CPU'
            metadata['causal_chain'] = [
                'Infinite loop in new code',
                'CPU utilization spike',
                'Request latency increases'
            ]
            
        return df_copy, metadata

if __name__ == "__main__":
    import os
    
    # Example execution to generate some data to disk
    generator = SyntheticMetricsGenerator()
    
    print("Generating 60 days of normal baseline metrics...")
    normal_data = generator.generate_normal_behavior(duration_days=60)
    
    # Inject a failure for the last ~17 hours
    failure_start = len(normal_data) - 200 
    print(f"Injecting 'database_slow_query' failure at index {failure_start}...")
    
    failed_data, meta = generator.inject_failure_scenario(
        normal_data,
        failure_type='database_slow_query',
        start_idx=failure_start,
        duration_samples=200,
        severity=0.8
    )
    
    # Save output
    os.makedirs('data/synthetic', exist_ok=True)
    failed_data.to_csv('data/synthetic/synthetic_metrics_with_db_failure.csv', index=False)
    
    with open('data/synthetic/synthetic_metrics_db_failure_metadata.json', 'w') as f:
        # Convert timestamp to string for JSON serialization
        meta['start_time'] = str(meta['start_time'])
        json.dump(meta, f, indent=2)
        
    print(f"Data saved to data/synthetic/. Metadata: {meta['root_cause']}")
