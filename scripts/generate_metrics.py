"""
Synthetic Metric Generator for the Automated RCA System (FR-04)

Generates correlated time-series metrics for each of the 10 failure scenarios.
Produces CSV files with per-minute metric data and optional Prometheus text output.

Designed as a companion to generate_failures.py — the anomaly injection window
(last 20% of the time range) aligns with when logs would show the failure.

Usage:
    python scripts/generate_metrics.py --scenario all --duration-hours 24 --seed 42
    python scripts/generate_metrics.py --scenario memory_leak --duration-hours 6 --seed 1
    python scripts/generate_metrics.py --list
    python scripts/generate_metrics.py --scenario all --prometheus
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIOS = [
    "db_migration",
    "memory_leak",
    "network_partition",
    "thread_pool",
    "dns_propagation",
    "cpu_saturation",
    "connection_pool_leak",
    "cache_stampede",
    "disk_exhaustion",
    "mq_backlog",
]

# All possible metric names and their baseline configs.
# Each entry: (baseline, noise_std, min_val, max_val, unit)
METRIC_DEFAULTS = {
    "cpu_usage_percent": (30.0, 3.0, 0.0, 100.0, "%"),
    "memory_usage_mb": (4000.0, 80.0, 0.0, 8192.0, "MB"),
    "disk_usage_percent": (45.0, 0.5, 0.0, 100.0, "%"),
    "network_bytes_in": (50e6, 5e6, 0.0, None, "bytes/s"),
    "network_bytes_out": (50e6, 5e6, 0.0, None, "bytes/s"),
    "query_latency_p95_ms": (15.0, 2.0, 1.0, None, "ms"),
    "active_connections": (50.0, 5.0, 0.0, 500.0, "count"),
    "thread_pool_active": (20.0, 3.0, 0.0, 200.0, "count"),
    "cache_hit_ratio": (0.95, 0.01, 0.0, 1.0, "ratio"),
    "queue_depth": (10.0, 2.0, 0.0, None, "count"),
    "error_rate_per_min": (0.5, 0.3, 0.0, None, "count/min"),
}

# Which metrics are relevant per scenario (subset of METRIC_DEFAULTS keys).
SCENARIO_METRICS: Dict[str, List[str]] = {
    "db_migration": [
        "cpu_usage_percent",
        "memory_usage_mb",
        "query_latency_p95_ms",
        "active_connections",
        "error_rate_per_min",
    ],
    "memory_leak": [
        "cpu_usage_percent",
        "memory_usage_mb",
        "error_rate_per_min",
    ],
    "network_partition": [
        "cpu_usage_percent",
        "network_bytes_in",
        "network_bytes_out",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "thread_pool": [
        "cpu_usage_percent",
        "thread_pool_active",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "dns_propagation": [
        "cpu_usage_percent",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "cpu_saturation": [
        "cpu_usage_percent",
        "memory_usage_mb",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "connection_pool_leak": [
        "cpu_usage_percent",
        "active_connections",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "cache_stampede": [
        "cpu_usage_percent",
        "cache_hit_ratio",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
    "disk_exhaustion": [
        "cpu_usage_percent",
        "disk_usage_percent",
        "error_rate_per_min",
    ],
    "mq_backlog": [
        "cpu_usage_percent",
        "queue_depth",
        "query_latency_p95_ms",
        "error_rate_per_min",
    ],
}

# Service labels mirror config.yaml log_sources labels.
SCENARIO_SERVICE: Dict[str, str] = {
    "db_migration": "database",
    "memory_leak": "application",
    "network_partition": "system",
    "thread_pool": "application",
    "dns_propagation": "system",
    "cpu_saturation": "system",
    "connection_pool_leak": "database",
    "cache_stampede": "application",
    "disk_exhaustion": "system",
    "mq_backlog": "application",
}


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def _diurnal(timestamps: np.ndarray) -> np.ndarray:
    """Return a diurnal multiplier (0-centred) that peaks at 14:00 and troughs at 04:00.

    Uses a sinusoid with a 24-hour period.  The peak hour is 14:00 local,
    which corresponds to a phase offset of (14 - 12) / 24 * 2pi.
    """
    hours = np.array([(t.hour + t.minute / 60.0) for t in timestamps])
    # sin peaks at pi/2 => want peak at hour 14 => shift so sin(2pi*(h-14)/24 + pi/2)
    return np.sin(2.0 * np.pi * (hours - 14.0) / 24.0 + np.pi / 2.0)


def _generate_baseline(
    metric_name: str,
    timestamps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a realistic baseline time-series for *metric_name*.

    Applies:
      - Gaussian noise on top of the base value.
      - Diurnal pattern for cpu_usage_percent and memory_usage_mb.
      - Slow linear drift for disk_usage_percent.
    """
    base, noise_std, min_val, max_val, _ = METRIC_DEFAULTS[metric_name]
    n = len(timestamps)
    values = np.full(n, base, dtype=np.float64)

    # Gaussian noise
    values += rng.normal(0, noise_std, size=n)

    # Diurnal pattern (sinusoidal)
    if metric_name in ("cpu_usage_percent", "memory_usage_mb"):
        diurnal = _diurnal(timestamps)
        if metric_name == "cpu_usage_percent":
            values += diurnal * 8.0  # +/- 8 pp swing
        else:
            values += diurnal * 300.0  # +/- 300 MB swing

    # Slow linear drift for disk
    if metric_name == "disk_usage_percent":
        drift = np.linspace(0, 2.0, n)  # +2 pp over the full window
        values += drift

    # Clamp
    if min_val is not None:
        values = np.maximum(values, min_val)
    if max_val is not None:
        values = np.minimum(values, max_val)

    return values


# ---------------------------------------------------------------------------
# Failure injection functions (last 20% of the window)
# ---------------------------------------------------------------------------


def _inject_db_migration(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """query_latency spikes 10x, active_connections saturates, error_rate spikes."""
    n_fault = int(fault_mask.sum())
    if "query_latency_p95_ms" in data:
        data["query_latency_p95_ms"][fault_mask] *= 10.0
        data["query_latency_p95_ms"][fault_mask] += rng.normal(0, 10, n_fault)
    if "active_connections" in data:
        # Ramp to 200 (pool max)
        ramp = np.linspace(data["active_connections"][fault_mask][0], 200.0, n_fault)
        data["active_connections"][fault_mask] = ramp + rng.normal(0, 2, n_fault)
        data["active_connections"] = np.clip(data["active_connections"], 0, 500)
    if "error_rate_per_min" in data:
        data["error_rate_per_min"][fault_mask] += rng.uniform(5, 30, n_fault)


def _inject_memory_leak(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """memory_usage grows linearly to near-OOM (7800+), cpu rises from GC pressure."""
    n_fault = int(fault_mask.sum())
    if "memory_usage_mb" in data:
        start_mem = data["memory_usage_mb"][fault_mask][0]
        data["memory_usage_mb"][fault_mask] = np.linspace(
            start_mem, 7800 + rng.uniform(0, 300), n_fault
        )
        data["memory_usage_mb"][fault_mask] += rng.normal(0, 20, n_fault)
        data["memory_usage_mb"] = np.clip(data["memory_usage_mb"], 0, 8192)
    if "cpu_usage_percent" in data:
        # GC pressure: proportional to how full memory is
        mem = data["memory_usage_mb"][fault_mask]
        gc_pressure = np.clip((mem - 6000) / 2000 * 30, 0, 40)  # up to +40 pp
        data["cpu_usage_percent"][fault_mask] += gc_pressure + rng.normal(0, 2, n_fault)
        data["cpu_usage_percent"] = np.clip(data["cpu_usage_percent"], 0, 100)


def _inject_network_partition(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """network_bytes drop to near-zero, error_rate spikes, latency -> timeout (30s)."""
    n_fault = int(fault_mask.sum())
    for key in ("network_bytes_in", "network_bytes_out"):
        if key in data:
            data[key][fault_mask] = rng.uniform(0, 1e5, n_fault)  # near zero
    if "query_latency_p95_ms" in data:
        data["query_latency_p95_ms"][fault_mask] = 30000.0 + rng.normal(0, 500, n_fault)
        data["query_latency_p95_ms"] = np.maximum(data["query_latency_p95_ms"], 1.0)
    if "error_rate_per_min" in data:
        data["error_rate_per_min"][fault_mask] += rng.uniform(20, 80, n_fault)


def _inject_thread_pool(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """thread_pool_active rises to max (200), latency increases, error_rate rises."""
    n_fault = int(fault_mask.sum())
    if "thread_pool_active" in data:
        start = data["thread_pool_active"][fault_mask][0]
        data["thread_pool_active"][fault_mask] = np.linspace(start, 200.0, n_fault)
        data["thread_pool_active"][fault_mask] += rng.normal(0, 1.5, n_fault)
        data["thread_pool_active"] = np.clip(data["thread_pool_active"], 0, 200)
    if "query_latency_p95_ms" in data:
        # Latency proportional to thread saturation
        pool = data["thread_pool_active"][fault_mask]
        extra_latency = (pool / 200.0) ** 2 * 500  # up to +500 ms
        data["query_latency_p95_ms"][fault_mask] += extra_latency
    if "error_rate_per_min" in data:
        data["error_rate_per_min"][fault_mask] += rng.uniform(3, 15, n_fault)


def _inject_dns_propagation(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Intermittent latency spikes (alternating normal/high), error_rate fluctuates."""
    n_fault = int(fault_mask.sum())
    # Alternating pattern: every other minute is spiked
    toggle = np.arange(n_fault) % 2 == 0
    if "query_latency_p95_ms" in data:
        spike = np.where(toggle, rng.uniform(200, 2000, n_fault), 0.0)
        data["query_latency_p95_ms"][fault_mask] += spike
    if "error_rate_per_min" in data:
        err_spike = np.where(
            toggle, rng.uniform(5, 25, n_fault), rng.uniform(0, 2, n_fault)
        )
        data["error_rate_per_min"][fault_mask] += err_spike


def _inject_cpu_saturation(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """cpu_usage rises to 95%+, latency increases proportionally."""
    n_fault = int(fault_mask.sum())
    if "cpu_usage_percent" in data:
        start_cpu = data["cpu_usage_percent"][fault_mask][0]
        data["cpu_usage_percent"][fault_mask] = np.linspace(start_cpu, 97.0, n_fault)
        data["cpu_usage_percent"][fault_mask] += rng.normal(0, 1.5, n_fault)
        data["cpu_usage_percent"] = np.clip(data["cpu_usage_percent"], 0, 100)
    if "query_latency_p95_ms" in data:
        cpu = data["cpu_usage_percent"][fault_mask]
        # Latency grows exponentially as CPU approaches 100%
        load_factor = np.clip((cpu - 60) / 40, 0, 1)
        extra = load_factor**3 * 3000  # up to +3000 ms at 100%
        data["query_latency_p95_ms"][fault_mask] += extra
    if "error_rate_per_min" in data:
        cpu = data["cpu_usage_percent"][fault_mask]
        err = np.clip((cpu - 80) / 20, 0, 1) * 20  # up to +20 at 100%
        data["error_rate_per_min"][fault_mask] += err + rng.normal(0, 1, n_fault)


def _inject_connection_pool_leak(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """active_connections grows steadily past normal, latency increases."""
    n_fault = int(fault_mask.sum())
    if "active_connections" in data:
        start_conn = data["active_connections"][fault_mask][0]
        # Grow to ~250 (well past the normal 50 baseline)
        data["active_connections"][fault_mask] = np.linspace(start_conn, 250.0, n_fault)
        data["active_connections"][fault_mask] += rng.normal(0, 3, n_fault)
        data["active_connections"] = np.clip(data["active_connections"], 0, 500)
    if "query_latency_p95_ms" in data:
        conns = data["active_connections"][fault_mask]
        extra = np.clip((conns - 80) / 170, 0, 1) ** 2 * 800  # up to +800 ms
        data["query_latency_p95_ms"][fault_mask] += extra
    if "error_rate_per_min" in data:
        conns = data["active_connections"][fault_mask]
        err = np.clip((conns - 150) / 100, 0, 1) * 15
        data["error_rate_per_min"][fault_mask] += err + rng.normal(0, 0.5, n_fault)


def _inject_cache_stampede(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """cache_hit_ratio drops to 0.1, latency spikes, cpu rises from recomputation."""
    n_fault = int(fault_mask.sum())
    if "cache_hit_ratio" in data:
        start_ratio = data["cache_hit_ratio"][fault_mask][0]
        data["cache_hit_ratio"][fault_mask] = np.linspace(start_ratio, 0.1, n_fault)
        data["cache_hit_ratio"][fault_mask] += rng.normal(0, 0.02, n_fault)
        data["cache_hit_ratio"] = np.clip(data["cache_hit_ratio"], 0.0, 1.0)
    if "query_latency_p95_ms" in data:
        # Latency inversely related to cache hit ratio
        if "cache_hit_ratio" in data:
            miss_rate = 1.0 - data["cache_hit_ratio"][fault_mask]
            extra = miss_rate * 500  # up to +500 ms at 0% hit
            data["query_latency_p95_ms"][fault_mask] += extra
        else:
            data["query_latency_p95_ms"][fault_mask] += rng.uniform(100, 500, n_fault)
    if "cpu_usage_percent" in data:
        # Recomputation load
        if "cache_hit_ratio" in data:
            miss_rate = 1.0 - data["cache_hit_ratio"][fault_mask]
            data["cpu_usage_percent"][fault_mask] += miss_rate * 40
        else:
            data["cpu_usage_percent"][fault_mask] += rng.uniform(10, 40, n_fault)
        data["cpu_usage_percent"] = np.clip(data["cpu_usage_percent"], 0, 100)
    if "error_rate_per_min" in data:
        data["error_rate_per_min"][fault_mask] += rng.uniform(3, 15, n_fault)


def _inject_disk_exhaustion(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """disk_usage grows to 98%+, error_rate spikes when writes fail."""
    n_fault = int(fault_mask.sum())
    if "disk_usage_percent" in data:
        start_disk = data["disk_usage_percent"][fault_mask][0]
        data["disk_usage_percent"][fault_mask] = np.linspace(start_disk, 99.0, n_fault)
        data["disk_usage_percent"][fault_mask] += rng.normal(0, 0.2, n_fault)
        data["disk_usage_percent"] = np.clip(data["disk_usage_percent"], 0, 100)
    if "error_rate_per_min" in data:
        disk = data["disk_usage_percent"][fault_mask]
        # Errors spike sharply above 95%
        err = np.clip((disk - 95) / 5, 0, 1) ** 2 * 50
        data["error_rate_per_min"][fault_mask] += err + rng.normal(0, 1, n_fault)
        data["error_rate_per_min"] = np.maximum(data["error_rate_per_min"], 0)


def _inject_mq_backlog(
    data: Dict[str, np.ndarray],
    fault_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """queue_depth grows exponentially, latency increases, error_rate rises."""
    n_fault = int(fault_mask.sum())
    if "queue_depth" in data:
        # Exponential growth from current to ~10000
        t_norm = np.linspace(0, 1, n_fault)
        start_q = data["queue_depth"][fault_mask][0]
        data["queue_depth"][fault_mask] = start_q * np.exp(
            t_norm * np.log(10000 / max(start_q, 1))
        )
        data["queue_depth"][fault_mask] += rng.normal(0, 5, n_fault)
        data["queue_depth"] = np.maximum(data["queue_depth"], 0)
    if "query_latency_p95_ms" in data:
        q = data["queue_depth"][fault_mask]
        extra = np.log1p(q) * 30  # logarithmic relationship
        data["query_latency_p95_ms"][fault_mask] += extra
    if "error_rate_per_min" in data:
        q = data["queue_depth"][fault_mask]
        err = np.clip(q / 5000, 0, 1) * 25
        data["error_rate_per_min"][fault_mask] += err + rng.normal(0, 1, n_fault)
        data["error_rate_per_min"] = np.maximum(data["error_rate_per_min"], 0)


# Registry mapping scenario name -> injection function
_INJECTORS = {
    "db_migration": _inject_db_migration,
    "memory_leak": _inject_memory_leak,
    "network_partition": _inject_network_partition,
    "thread_pool": _inject_thread_pool,
    "dns_propagation": _inject_dns_propagation,
    "cpu_saturation": _inject_cpu_saturation,
    "connection_pool_leak": _inject_connection_pool_leak,
    "cache_stampede": _inject_cache_stampede,
    "disk_exhaustion": _inject_disk_exhaustion,
    "mq_backlog": _inject_mq_backlog,
}


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def generate_scenario_metrics(
    scenario: str,
    duration_hours: float = 24.0,
    interval_seconds: int = 60,
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate a DataFrame of time-series metrics for a single failure scenario.

    Parameters
    ----------
    scenario : str
        One of the 10 supported scenario names (see ``SCENARIOS``).
    duration_hours : float
        Total duration of the time-series in hours (default 24).
    interval_seconds : int
        Sampling interval in seconds (default 60 — 1-minute resolution).
    seed : int
        Random seed for reproducibility.
    start_time : datetime, optional
        Start of the time window.  Defaults to ``2026-03-15 00:00:00``.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``timestamp``, ``metric_name``, ``value``, ``service``.
    """
    if scenario not in _INJECTORS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available: {', '.join(SCENARIOS)}"
        )

    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = datetime(2026, 3, 15, 0, 0, 0)

    n_points = int(duration_hours * 3600 / interval_seconds)
    timestamps = np.array(
        [start_time + timedelta(seconds=i * interval_seconds) for i in range(n_points)]
    )

    # Determine which metrics this scenario uses
    metric_names = SCENARIO_METRICS[scenario]
    service = SCENARIO_SERVICE[scenario]

    # Build baseline for each metric
    data: Dict[str, np.ndarray] = {}
    for m in metric_names:
        data[m] = _generate_baseline(m, timestamps, rng)

    # Fault mask: last 20% of data points
    fault_start = int(n_points * 0.8)
    fault_mask = np.zeros(n_points, dtype=bool)
    fault_mask[fault_start:] = True

    # Inject anomaly
    _INJECTORS[scenario](data, fault_mask, rng)

    # Final clamp pass using METRIC_DEFAULTS bounds
    for m in metric_names:
        _, _, min_val, max_val, _ = METRIC_DEFAULTS[m]
        if min_val is not None:
            data[m] = np.maximum(data[m], min_val)
        if max_val is not None:
            data[m] = np.minimum(data[m], max_val)

    # Build long-format DataFrame
    rows = []
    for m in metric_names:
        for i in range(n_points):
            rows.append(
                {
                    "timestamp": timestamps[i],
                    "metric_name": m,
                    "value": round(float(data[m][i]), 4),
                    "service": service,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prometheus text format exporter
# ---------------------------------------------------------------------------


def to_prometheus_text(df: pd.DataFrame) -> str:
    """Convert the *latest* data-point per metric in *df* to Prometheus text exposition format.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``generate_scenario_metrics`` (must have columns
        ``timestamp``, ``metric_name``, ``value``, ``service``).

    Returns
    -------
    str
        Prometheus-compatible ``/metrics`` text output.
    """
    latest = df.loc[df.groupby("metric_name")["timestamp"].idxmax()]
    lines: List[str] = []
    for _, row in latest.iterrows():
        name = row["metric_name"]
        prom_name = f"rca_{name}"
        _, _, _, _, unit = METRIC_DEFAULTS[name]
        lines.append(f"# HELP {prom_name} Synthetic RCA metric ({unit})")
        lines.append(f"# TYPE {prom_name} gauge")
        lines.append(f'{prom_name}{{service="{row["service"]}"}} {row["value"]}')
    lines.append("")  # trailing newline per spec
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _write_scenario_csv(df: pd.DataFrame, scenario: str, output_dir: str) -> str:
    """Write a per-scenario CSV and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{scenario}_metrics.csv")
    df.to_csv(path, index=False)
    return path


def _write_combined_csv(frames: List[pd.DataFrame], output_dir: str) -> str:
    """Write the combined CSV for all scenarios and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    path = os.path.join(output_dir, "combined_metrics.csv")
    combined.to_csv(path, index=False)
    return path


def run_generation(
    scenario_names: List[str],
    duration_hours: float,
    interval_seconds: int,
    seed: int,
    output_dir: str,
    prometheus: bool = False,
) -> None:
    """Generate metrics for the requested scenarios and write outputs.

    Parameters
    ----------
    scenario_names : list[str]
        Scenario names to generate.
    duration_hours : float
        Duration in hours.
    interval_seconds : int
        Sampling interval in seconds.
    seed : int
        Master random seed.  Each scenario gets ``seed + index`` for variety.
    output_dir : str
        Directory for CSV output.
    prometheus : bool
        If True, also write a ``prometheus_latest.txt`` file.
    """
    frames: List[pd.DataFrame] = []
    n_points = int(duration_hours * 3600 / interval_seconds)

    for idx, name in enumerate(scenario_names):
        scenario_seed = seed + idx
        print(f"  Generating '{name}' ({n_points} points, seed={scenario_seed}) ...")
        df = generate_scenario_metrics(
            scenario=name,
            duration_hours=duration_hours,
            interval_seconds=interval_seconds,
            seed=scenario_seed,
        )
        path = _write_scenario_csv(df, name, output_dir)
        n_metrics = df["metric_name"].nunique()
        print(f"    -> {path}  ({len(df)} rows, {n_metrics} metrics)")
        frames.append(df)

    combined_path = _write_combined_csv(frames, output_dir)
    total_rows = sum(len(f) for f in frames)
    print(f"\n  Combined CSV: {combined_path}  ({total_rows} rows)")

    if prometheus:
        # Write Prometheus text for the last scenario's latest data point
        prom_text = to_prometheus_text(pd.concat(frames, ignore_index=True))
        prom_path = os.path.join(output_dir, "prometheus_latest.txt")
        with open(prom_path, "w", encoding="utf-8") as f:
            f.write(prom_text)
        print(f"  Prometheus: {prom_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Metric Generator for the Automated RCA System (FR-04)"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default="all",
        help=f"Scenario name or 'all'. Available: {', '.join(SCENARIOS)}",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=24.0,
        help="Duration of time-series in hours (default: 24)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Sampling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/metrics",
        help="Output directory for CSV files (default: data/metrics)",
    )
    parser.add_argument(
        "--prometheus",
        action="store_true",
        help="Also write a Prometheus text exposition file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing metric CSVs before generating",
    )

    args = parser.parse_args()

    if args.list:
        print("Available failure scenarios (metrics):")
        for name in SCENARIOS:
            metrics = ", ".join(SCENARIO_METRICS[name])
            print(f"  {name:25s} — metrics: {metrics}")
        return

    if args.clear and os.path.isdir(args.output_dir):
        removed = 0
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".csv") or fname.endswith(".txt"):
                os.remove(os.path.join(args.output_dir, fname))
                removed += 1
        print(f"  Cleared {removed} file(s) from {args.output_dir}")

    if args.scenario == "all":
        names = list(SCENARIOS)
    else:
        names = [s.strip() for s in args.scenario.split(",")]
        for n in names:
            if n not in SCENARIO_METRICS:
                print(
                    f"Error: Unknown scenario '{n}'. Use --list to see available scenarios."
                )
                sys.exit(1)

    n_points = int(args.duration_hours * 3600 / args.interval_seconds)
    print(
        f"Metric Generator: {len(names)} scenario(s), "
        f"{args.duration_hours}h @ {args.interval_seconds}s intervals "
        f"({n_points} points), seed={args.seed}"
    )
    run_generation(
        scenario_names=names,
        duration_hours=args.duration_hours,
        interval_seconds=args.interval_seconds,
        seed=args.seed,
        output_dir=args.output_dir,
        prometheus=args.prometheus,
    )


if __name__ == "__main__":
    main()
