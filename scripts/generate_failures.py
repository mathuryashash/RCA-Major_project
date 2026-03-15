"""
Synthetic Failure Scenario Generator (FR-04)

Generates labelled failure scenarios for the Automated RCA System.
Produces log files (app.log, db.log, syslog.log) and JSONL ground-truth labels.

Usage:
    python scripts/generate_failures.py --scenario all --count 20 --seed 42
    python scripts/generate_failures.py --scenario db_migration --count 5 --seed 1
    python scripts/generate_failures.py --list
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    """Standard log timestamp."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _syslog_ts(dt: datetime) -> str:
    """Syslog-style timestamp: 'Mon DD HH:MM:SS'."""
    return dt.strftime("%b %d %H:%M:%S")


def write_log(path: str, dt: datetime, level: str, message: str, fmt: str = "standard"):
    """Append a log line to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "syslog":
        hostname = "prod-server-01"
        line = f"{_syslog_ts(dt)} {hostname} kernel: [{level}] {message}\n"
    else:
        line = f"{_ts(dt)} {level} {message}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _jitter(rng: random.Random, minutes: float) -> timedelta:
    """Random jitter up to +/- minutes."""
    return timedelta(seconds=rng.uniform(-minutes * 60, minutes * 60))


def _severity_multiplier(severity: str) -> float:
    return {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}.get(severity, 1.0)


SEVERITIES = ["low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# Scenario implementations
# ---------------------------------------------------------------------------


def scenario_db_migration(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 1: Database Schema Migration failure."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    mul = _severity_multiplier(severity)
    t = base

    table = rng.choice(
        ["product_variants", "order_items", "user_sessions", "inventory"]
    )
    rows = rng.randint(1_000_000, 80_000_000)

    write_log(db, t, "INFO", f"Applying migration: add_composite_index on {table}")
    t += timedelta(seconds=rng.randint(5, 30))
    write_log(db, t, "INFO", f"Migration complete. Rows affected: {rows}")

    # Healthy period
    for _ in range(rng.randint(3, 6)):
        t += timedelta(minutes=rng.randint(5, 20))
        write_log(
            app,
            t,
            "INFO",
            f"Request handled: GET /api/{table} 200 {rng.randint(30, 60)}ms",
        )
        write_log(
            db,
            t,
            "INFO",
            f"Query executed: SELECT * FROM {table} ({rng.randint(20, 50)}ms)",
        )

    # Degradation phase
    latencies = [int(100 * mul + i * 200 * mul) for i in range(rng.randint(5, 10))]
    for lat in latencies:
        t += timedelta(minutes=rng.randint(10, 40))
        write_log(
            db,
            t,
            "WARN",
            f"Slow query detected: SELECT * FROM {table} WHERE id IN (...) ({lat}ms)",
        )
        if lat > 500:
            write_log(
                app,
                t + timedelta(seconds=2),
                "WARN",
                f"Upstream response slow: database ({lat}ms)",
            )

    # Connection pool exhaustion
    pool = rng.randint(80, 100)
    t += timedelta(minutes=rng.randint(5, 15))
    write_log(
        db, t, "ERROR", f"Connection pool at capacity: {pool}/{pool} connections active"
    )
    write_log(
        db,
        t + timedelta(seconds=5),
        "ERROR",
        f"Rejecting new connection: pool exhausted",
    )

    # API failures
    t += timedelta(minutes=rng.randint(1, 5))
    write_log(
        app,
        t,
        "ERROR",
        f"Request timeout: waiting for DB connection ({int(5000 * mul)}ms)",
    )
    write_log(
        app,
        t + timedelta(seconds=10),
        "ERROR",
        f"5xx error rate elevated: {rng.uniform(5, 20):.1f}%",
    )
    write_log(
        app,
        t + timedelta(seconds=30),
        "CRITICAL",
        f"Service health check failed: error_rate={rng.uniform(10, 30):.1f}%",
    )

    return {
        "scenario": "db_migration",
        "label": "db_migration_applied",
        "causal_chain": [
            "db_migration",
            "query_latency",
            "connection_pool",
            "api_timeout",
        ],
        "affected_services": ["database", "application"],
    }


def scenario_memory_leak(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 2: Memory Leak from WebSocket handler."""
    app = os.path.join(out, "app.log")
    sys_log = os.path.join(out, "syslog.log")
    mul = _severity_multiplier(severity)
    t = base

    version = f"v{rng.randint(2, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
    write_log(app, t, "INFO", f"Deployment started: {version} (rolling restart)")
    t += timedelta(seconds=rng.randint(30, 120))
    write_log(app, t, "INFO", f"Deployment complete: {version} all pods healthy")

    # Memory growth over hours
    mem_mb = rng.randint(400, 600)
    for i in range(rng.randint(8, 15)):
        t += timedelta(hours=rng.uniform(0.5, 4))
        mem_mb += int(rng.randint(50, 200) * mul)
        write_log(
            app,
            t,
            "INFO",
            f"WebSocket connections: {rng.randint(100, 5000)} active, memory: {mem_mb}MB",
        )
        if mem_mb > 1200:
            write_log(
                app,
                t + timedelta(seconds=5),
                "WARN",
                f"Memory usage high: {mem_mb}MB / 2048MB ({mem_mb * 100 // 2048}%)",
            )
        if mem_mb > 1800:
            write_log(
                app,
                t + timedelta(seconds=10),
                "ERROR",
                f"Memory critically high: {mem_mb}MB, GC unable to reclaim",
            )
            write_log(
                sys_log,
                t + timedelta(seconds=12),
                "WARN",
                f"Memory pressure detected: available {2048 - mem_mb}MB",
                fmt="syslog",
            )

    # OOM
    t += timedelta(minutes=rng.randint(10, 60))
    pid = rng.randint(1000, 9999)
    write_log(
        sys_log,
        t,
        "ERROR",
        f"OOM killer activated: process node (pid {pid}) score 950",
        fmt="syslog",
    )
    write_log(
        app,
        t + timedelta(seconds=2),
        "CRITICAL",
        f"Pod restarted due to OOMKilled: container api-server",
    )
    write_log(
        app,
        t + timedelta(seconds=30),
        "INFO",
        f"Pod recovered: api-server ready, memory reset to 450MB",
    )

    return {
        "scenario": "memory_leak",
        "label": "memory_leak_code_deploy",
        "causal_chain": ["code_deploy", "memory_growth", "oom_kill", "pod_restart"],
        "affected_services": ["application", "system"],
    }


def scenario_network_partition(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 3: Network Partition (inter-region)."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    sys_log = os.path.join(out, "syslog.log")
    t = base

    region_a = rng.choice(["us-east-1", "us-west-2", "eu-west-1"])
    region_b = rng.choice(["ap-southeast-1", "eu-central-1", "us-east-2"])

    write_log(
        sys_log,
        t,
        "WARN",
        f"BGP route update detected: path to {region_b} changed via AS{rng.randint(1000, 9999)}",
        fmt="syslog",
    )
    t += timedelta(seconds=rng.randint(1, 10))
    write_log(
        sys_log,
        t,
        "ERROR",
        f"Network partition detected: {region_a} cannot reach {region_b}",
        fmt="syslog",
    )

    # Redis split-brain
    t += timedelta(seconds=rng.randint(5, 30))
    write_log(
        db, t, "ERROR", f"Redis cluster: lost connection to replica in {region_b}"
    )
    write_log(
        db,
        t + timedelta(seconds=2),
        "WARN",
        f"Redis cluster: split-brain detected, promoting local replica",
    )
    write_log(
        db,
        t + timedelta(seconds=5),
        "ERROR",
        f"Redis READONLY: replica cannot accept writes during partition",
    )

    # Application impact
    for i in range(rng.randint(3, 8)):
        t += timedelta(seconds=rng.randint(10, 60))
        write_log(
            app,
            t,
            "ERROR",
            f"Cache read failed: connection to {region_b} timed out after 3000ms",
        )
        write_log(
            app,
            t + timedelta(seconds=1),
            "WARN",
            f"Falling back to database for session data",
        )

    # Data inconsistency
    t += timedelta(minutes=rng.randint(1, 5))
    write_log(
        db,
        t,
        "ERROR",
        f"Data inconsistency detected: {rng.randint(10, 500)} conflicting keys after partition heal",
    )
    write_log(
        app,
        t + timedelta(seconds=10),
        "WARN",
        f"Stale data served to {rng.randint(100, 5000)} requests during partition",
    )

    return {
        "scenario": "network_partition",
        "label": "network_partition_bgp",
        "causal_chain": [
            "bgp_route_change",
            "network_partition",
            "redis_split_brain",
            "data_inconsistency",
        ],
        "affected_services": ["system", "database", "application"],
    }


def scenario_thread_pool(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 4: Thread Pool Exhaustion."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    mul = _severity_multiplier(severity)
    t = base

    job = rng.choice(
        ["data-export", "report-generation", "batch-import", "email-dispatch"]
    )
    max_threads = rng.choice([50, 100, 200])

    write_log(
        app,
        t,
        "INFO",
        f"Background job triggered: {job} with {int(rng.randint(500, 5000) * mul)} items",
    )

    # Thread pool filling
    for i in range(rng.randint(5, 10)):
        active = min(max_threads, int((i + 1) * max_threads / 8 * mul))
        t += timedelta(seconds=rng.randint(5, 30))
        write_log(
            app,
            t,
            "INFO" if active < max_threads * 0.7 else "WARN",
            f"Thread pool: {active}/{max_threads} active, queue depth: {rng.randint(0, 200)}",
        )

    # Exhaustion
    t += timedelta(seconds=rng.randint(10, 60))
    write_log(
        app,
        t,
        "ERROR",
        f"Thread pool exhausted: {max_threads}/{max_threads} threads busy, {rng.randint(50, 500)} requests queued",
    )
    write_log(
        db,
        t + timedelta(seconds=5),
        "WARN",
        f"Slow query: connection held for {rng.randint(10, 60)}s by idle thread",
    )

    # Cascading timeouts
    for i in range(rng.randint(3, 6)):
        t += timedelta(seconds=rng.randint(5, 30))
        write_log(
            app,
            t,
            "ERROR",
            f"Request timeout: {rng.choice(['GET /api/data', 'POST /api/upload', 'GET /health'])} ({rng.randint(5000, 30000)}ms)",
        )

    write_log(
        app,
        t + timedelta(seconds=30),
        "CRITICAL",
        f"Service degraded: thread pool exhaustion, {rng.randint(10, 50)}% requests failing",
    )

    return {
        "scenario": "thread_pool",
        "label": "thread_pool_background_job",
        "causal_chain": [
            "background_job_spike",
            "thread_pool_saturation",
            "request_queuing",
            "timeouts",
        ],
        "affected_services": ["application", "database"],
    }


def scenario_dns_propagation(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 5: DNS Propagation Delay."""
    app = os.path.join(out, "app.log")
    sys_log = os.path.join(out, "syslog.log")
    t = base

    domain = rng.choice(
        ["api.internal.corp", "db-primary.service.local", "cache.cluster.internal"]
    )
    old_ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    new_ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    ttl = rng.choice([300, 600, 1800, 3600])

    write_log(
        sys_log,
        t,
        "INFO",
        f"DNS record updated: {domain} A {old_ip} -> {new_ip} (TTL={ttl}s)",
        fmt="syslog",
    )

    # Stale DNS causing failures
    for i in range(rng.randint(5, 12)):
        t += timedelta(seconds=rng.randint(30, ttl // 4))
        if rng.random() < 0.6:
            write_log(
                app,
                t,
                "ERROR",
                f"Connection refused: {domain} ({old_ip}:{rng.choice([5432, 6379, 8080])})",
            )
        else:
            write_log(
                app,
                t,
                "WARN",
                f"DNS resolution returned stale record for {domain}: {old_ip} (expected {new_ip})",
            )

    # Mixed period
    t += timedelta(seconds=rng.randint(60, 300))
    write_log(
        sys_log,
        t,
        "WARN",
        f"DNS TTL misconfiguration: {domain} TTL={ttl}s exceeds recommended 60s",
        fmt="syslog",
    )
    write_log(
        app,
        t + timedelta(seconds=10),
        "WARN",
        f"Intermittent connectivity: {domain} resolving to both {old_ip} and {new_ip}",
    )

    # Recovery
    t += timedelta(seconds=ttl + rng.randint(10, 120))
    write_log(
        app,
        t,
        "INFO",
        f"DNS propagation complete: {domain} resolving to {new_ip} consistently",
    )

    return {
        "scenario": "dns_propagation",
        "label": "dns_ttl_misconfiguration",
        "causal_chain": [
            "dns_record_change",
            "stale_dns_cache",
            "connection_failures",
            "intermittent_errors",
        ],
        "affected_services": ["system", "application"],
    }


def scenario_cpu_saturation(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 6: CPU Saturation from runaway process."""
    app = os.path.join(out, "app.log")
    sys_log = os.path.join(out, "syslog.log")
    t = base

    process = rng.choice(["java", "python3", "node", "postgres"])
    pid = rng.randint(1000, 65000)
    cause = rng.choice(
        [
            "regex backtracking",
            "infinite loop in batch job",
            "crypto mining malware",
            "unoptimized query plan",
        ]
    )

    write_log(
        sys_log,
        t,
        "WARN",
        f"CPU usage spike: {process} (pid {pid}) consuming 95% CPU",
        fmt="syslog",
    )
    t += timedelta(seconds=rng.randint(5, 30))
    write_log(
        sys_log,
        t,
        "WARN",
        f"Load average: {rng.uniform(8, 32):.1f} {rng.uniform(6, 24):.1f} {rng.uniform(4, 16):.1f}",
        fmt="syslog",
    )

    # Service degradation
    for i in range(rng.randint(5, 10)):
        t += timedelta(seconds=rng.randint(10, 60))
        write_log(
            app,
            t,
            "WARN",
            f"Response latency elevated: p99={rng.randint(500, 5000)}ms (cause: CPU contention)",
        )
        if rng.random() < 0.4:
            write_log(
                sys_log,
                t,
                "WARN",
                f"Context switch rate elevated: {rng.randint(50000, 200000)}/s",
                fmt="syslog",
            )

    # Critical
    t += timedelta(minutes=rng.randint(1, 10))
    write_log(
        sys_log,
        t,
        "ERROR",
        f"CPU saturation: all cores at 100% for {rng.randint(30, 300)}s, suspected {cause}",
        fmt="syslog",
    )
    write_log(
        app,
        t + timedelta(seconds=5),
        "ERROR",
        f"Health check timeout: service unresponsive",
    )
    write_log(
        app,
        t + timedelta(seconds=15),
        "CRITICAL",
        f"Service marked unhealthy: removed from load balancer",
    )

    return {
        "scenario": "cpu_saturation",
        "label": "cpu_runaway_process",
        "causal_chain": [
            "runaway_process",
            "cpu_saturation",
            "service_degradation",
            "health_check_failure",
        ],
        "affected_services": ["system", "application"],
    }


def scenario_connection_pool_leak(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 7: Connection Pool Leak."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    mul = _severity_multiplier(severity)
    t = base

    pool_max = rng.choice([20, 50, 100])
    endpoint = rng.choice(
        ["/api/users", "/api/orders", "/api/products", "/api/analytics"]
    )

    write_log(
        app, t, "INFO", f"Connection pool initialized: max={pool_max}, idle={pool_max}"
    )

    # Gradual leak
    active = 0
    for i in range(rng.randint(8, 15)):
        t += timedelta(minutes=rng.randint(2, 10))
        leaked = int(rng.randint(2, 8) * mul)
        active = min(pool_max, active + leaked)
        idle = pool_max - active
        write_log(
            db,
            t,
            "INFO" if active < pool_max * 0.6 else "WARN",
            f"Connection pool status: active={active}, idle={idle}, max={pool_max}",
        )
        if active > pool_max * 0.7:
            write_log(
                app,
                t + timedelta(seconds=1),
                "WARN",
                f"Connection pool usage high: {active}/{pool_max} ({active * 100 // pool_max}%)",
            )

    # Exhaustion
    t += timedelta(minutes=rng.randint(5, 15))
    write_log(
        db,
        t,
        "ERROR",
        f"Connection pool exhausted: {pool_max}/{pool_max} active, 0 idle",
    )
    write_log(
        app,
        t + timedelta(seconds=2),
        "ERROR",
        f"Cannot acquire database connection for {endpoint}: pool exhausted, waited 30s",
    )

    # Cascading failures
    for _ in range(rng.randint(3, 6)):
        t += timedelta(seconds=rng.randint(5, 30))
        write_log(
            app,
            t,
            "ERROR",
            f"Database operation failed: connection pool timeout for {rng.choice([endpoint, '/api/health', '/api/status'])}",
        )

    write_log(
        app,
        t + timedelta(seconds=30),
        "CRITICAL",
        f"Service degraded: {rng.randint(40, 80)}% of requests failing due to connection pool exhaustion",
    )

    return {
        "scenario": "connection_pool_leak",
        "label": "connection_leak_bug",
        "causal_chain": [
            "connection_leak",
            "pool_exhaustion",
            "request_failures",
            "service_degradation",
        ],
        "affected_services": ["database", "application"],
    }


def scenario_cache_stampede(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 8: Cache Stampede."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    mul = _severity_multiplier(severity)
    t = base

    cache_key = rng.choice(
        ["product_catalog", "user_preferences", "pricing_rules", "homepage_feed"]
    )
    ttl = rng.choice([300, 600, 1800])

    write_log(
        app, t, "INFO", f"Cache serving: key={cache_key}, hit_rate=99.2%, TTL={ttl}s"
    )

    # Normal operation
    for _ in range(rng.randint(2, 4)):
        t += timedelta(minutes=rng.randint(1, 5))
        write_log(
            app, t, "INFO", f"Cache hit: {cache_key} (latency: {rng.randint(1, 5)}ms)"
        )

    # TTL expiry - stampede
    t += timedelta(seconds=ttl)
    write_log(app, t, "WARN", f"Cache miss: {cache_key} expired (TTL={ttl}s)")

    concurrent = int(rng.randint(50, 500) * mul)
    write_log(
        app,
        t + timedelta(milliseconds=100),
        "WARN",
        f"Thundering herd detected: {concurrent} concurrent requests for {cache_key}",
    )
    write_log(
        db,
        t + timedelta(seconds=1),
        "WARN",
        f"Query spike: {concurrent} simultaneous queries for {cache_key} data",
    )

    # DB overload
    t += timedelta(seconds=rng.randint(5, 30))
    write_log(
        db,
        t,
        "ERROR",
        f"Database overloaded: query queue depth {rng.randint(200, 1000)}, avg latency {rng.randint(2000, 10000)}ms",
    )
    write_log(
        db,
        t + timedelta(seconds=5),
        "WARN",
        f"Connection pool usage: {rng.randint(80, 100)}%",
    )

    # Cascading slow queries
    for _ in range(rng.randint(3, 6)):
        t += timedelta(seconds=rng.randint(5, 30))
        write_log(
            app,
            t,
            "ERROR",
            f"Slow response: {rng.choice(['GET /products', 'GET /feed', 'GET /pricing'])} {rng.randint(3000, 15000)}ms",
        )

    write_log(
        app,
        t + timedelta(seconds=30),
        "WARN",
        f"Cache re-populated: {cache_key} (rebuild took {rng.randint(5, 60)}s)",
    )

    return {
        "scenario": "cache_stampede",
        "label": "cache_ttl_expiry",
        "causal_chain": [
            "cache_ttl_expiry",
            "thundering_herd",
            "db_overload",
            "slow_responses",
        ],
        "affected_services": ["application", "database"],
    }


def scenario_disk_exhaustion(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 9: Disk Space Exhaustion."""
    app = os.path.join(out, "app.log")
    db = os.path.join(out, "db.log")
    sys_log = os.path.join(out, "syslog.log")
    mul = _severity_multiplier(severity)
    t = base

    disk_total = rng.choice([50, 100, 200])  # GB

    write_log(
        sys_log,
        t,
        "INFO",
        f"Disk usage: /var/log {rng.randint(40, 60)}% of {disk_total}GB",
        fmt="syslog",
    )
    write_log(
        sys_log,
        t + timedelta(seconds=5),
        "WARN",
        f"logrotate: configuration missing for /var/log/application/*.log",
        fmt="syslog",
    )

    # Gradual fill
    usage = rng.randint(60, 70)
    for i in range(rng.randint(6, 12)):
        t += timedelta(hours=rng.uniform(1, 6))
        usage = min(99, usage + int(rng.randint(3, 8) * mul))
        level = "INFO" if usage < 80 else ("WARN" if usage < 90 else "ERROR")
        write_log(
            sys_log,
            t,
            level,
            f"Disk usage: /var/log {usage}% of {disk_total}GB",
            fmt="syslog",
        )
        if usage > 85:
            write_log(
                app,
                t,
                "WARN",
                f"Log file size: /var/log/application/app.log is {rng.uniform(5, 50):.1f}GB",
            )

    # Disk full
    t += timedelta(hours=rng.uniform(0.5, 3))
    write_log(
        sys_log,
        t,
        "ERROR",
        f"Disk full: /var/log 100% of {disk_total}GB, no space left on device",
        fmt="syslog",
    )
    write_log(
        app,
        t + timedelta(seconds=5),
        "ERROR",
        f"Failed to write log: [Errno 28] No space left on device",
    )
    write_log(
        db,
        t + timedelta(seconds=10),
        "ERROR",
        f"WAL write failed: no space left on device (/var/log)",
    )
    write_log(
        db,
        t + timedelta(seconds=15),
        "CRITICAL",
        f"Database shutting down: cannot write WAL segments",
    )
    write_log(
        app,
        t + timedelta(seconds=20),
        "CRITICAL",
        f"Service unavailable: disk space exhaustion on /var/log",
    )

    return {
        "scenario": "disk_exhaustion",
        "label": "log_rotation_disabled",
        "causal_chain": [
            "log_rotation_disabled",
            "disk_fill",
            "write_failure",
            "service_crash",
        ],
        "affected_services": ["system", "application", "database"],
    }


def scenario_mq_backlog(
    base: datetime, rng: random.Random, out: str, severity: str
) -> Dict:
    """Scenario 10: Message Queue Backlog."""
    app = os.path.join(out, "app.log")
    mul = _severity_multiplier(severity)
    t = base

    queue = rng.choice(
        ["order-processing", "notification-dispatch", "data-pipeline", "event-stream"]
    )
    consumer = f"{queue}-consumer"

    write_log(
        app,
        t,
        "INFO",
        f"Consumer {consumer} healthy: processing {rng.randint(100, 1000)} msg/s",
    )

    # Consumer crash
    t += timedelta(minutes=rng.randint(5, 30))
    crash_reason = rng.choice(
        [
            "OutOfMemoryError in message deserializer",
            "Unhandled exception: NullPointerException",
            "Segfault in native library",
            "Connection to downstream service lost",
        ]
    )
    write_log(app, t, "ERROR", f"Consumer {consumer} crashed: {crash_reason}")
    write_log(
        app,
        t + timedelta(seconds=2),
        "WARN",
        f"Consumer {consumer} marked unhealthy, no heartbeat",
    )

    # Backlog growth
    backlog = 0
    for i in range(rng.randint(6, 12)):
        t += timedelta(minutes=rng.randint(1, 10))
        backlog += int(rng.randint(5000, 50000) * mul)
        level = "WARN" if backlog < 200000 else "ERROR"
        write_log(
            app,
            t,
            level,
            f"Queue {queue}: backlog={backlog}, consumers=0/{rng.randint(1, 5)}",
        )
        if backlog > 100000:
            write_log(
                app,
                t + timedelta(seconds=5),
                "WARN",
                f"Producer throttling: queue {queue} backpressure, publish latency {rng.randint(100, 5000)}ms",
            )

    # Upstream impact
    t += timedelta(minutes=rng.randint(5, 15))
    write_log(
        app,
        t,
        "ERROR",
        f"Producer timeout: cannot publish to {queue}, backlog={backlog}",
    )
    write_log(
        app,
        t + timedelta(seconds=10),
        "ERROR",
        f"Upstream service degraded: {rng.choice(['order-api', 'notification-service', 'data-ingestion'])} returning 503",
    )
    write_log(
        app,
        t + timedelta(seconds=30),
        "CRITICAL",
        f"Message queue {queue}: backlog={backlog}, estimated drain time: {backlog // 1000} minutes",
    )

    return {
        "scenario": "mq_backlog",
        "label": "consumer_service_crash",
        "causal_chain": [
            "consumer_crash",
            "queue_backlog",
            "producer_backpressure",
            "upstream_timeout",
        ],
        "affected_services": ["application"],
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS = {
    "db_migration": scenario_db_migration,
    "memory_leak": scenario_memory_leak,
    "network_partition": scenario_network_partition,
    "thread_pool": scenario_thread_pool,
    "dns_propagation": scenario_dns_propagation,
    "cpu_saturation": scenario_cpu_saturation,
    "connection_pool_leak": scenario_connection_pool_leak,
    "cache_stampede": scenario_cache_stampede,
    "disk_exhaustion": scenario_disk_exhaustion,
    "mq_backlog": scenario_mq_backlog,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenarios(
    scenario_names: List[str],
    count: int,
    seed: int,
    output_dir: str,
    labels_dir: str,
    realtime: bool = False,
):
    """Generate failure scenario logs and JSONL labels."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    rng = random.Random(seed)
    label_file = os.path.join(labels_dir, "ground_truth.jsonl")
    total_lines_before = _count_log_lines(output_dir)
    total_labels = 0

    for name in scenario_names:
        fn = SCENARIOS[name]
        print(f"  Generating '{name}' x{count} ...")
        for i in range(count):
            variation_seed = rng.randint(0, 2**31)
            var_rng = random.Random(variation_seed)
            severity = var_rng.choice(SEVERITIES)
            base_time = datetime(2026, 3, 15, 10, 0, 0) + timedelta(
                hours=var_rng.randint(-48, 48)
            )

            label = fn(base_time, var_rng, output_dir, severity)
            label["seed"] = variation_seed
            label["severity"] = severity
            label["timestamp"] = base_time.isoformat()

            with open(label_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(label) + "\n")
            total_labels += 1

            if realtime:
                time.sleep(0.5)

    total_lines_after = _count_log_lines(output_dir)
    new_lines = total_lines_after - total_lines_before
    print(f"\n  Generated {total_labels} scenarios, ~{new_lines} log lines")
    print(f"  Labels: {label_file}")
    print(f"  Logs:   {output_dir}/")


def _count_log_lines(directory: str) -> int:
    """Count total lines across all .log files in a directory."""
    total = 0
    if not os.path.exists(directory):
        return 0
    for fname in os.listdir(directory):
        if fname.endswith(".log"):
            path = os.path.join(directory, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                total += sum(1 for _ in f)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Failure Scenario Generator for the RCA System (FR-04)"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        default="all",
        help=f"Scenario name or 'all'. Available: {', '.join(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=20,
        help="Variations per scenario (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/logs", help="Log output directory"
    )
    parser.add_argument(
        "--labels-dir", default="data/labels", help="JSONL labels output directory"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Add delays between writes (for demo)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenarios and exit"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing logs before generating"
    )

    args = parser.parse_args()

    if args.list:
        print("Available failure scenarios:")
        for name, fn in SCENARIOS.items():
            print(f"  {name:25s} — {fn.__doc__.strip()}")
        return

    if args.clear:
        for fname in ["app.log", "db.log", "syslog.log"]:
            path = os.path.join(args.output_dir, fname)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Cleared {path}")
        label_path = os.path.join(args.labels_dir, "ground_truth.jsonl")
        if os.path.exists(label_path):
            os.remove(label_path)
            print(f"  Cleared {label_path}")

    if args.scenario == "all":
        names = list(SCENARIOS.keys())
    else:
        names = [s.strip() for s in args.scenario.split(",")]
        for n in names:
            if n not in SCENARIOS:
                print(
                    f"Error: Unknown scenario '{n}'. Use --list to see available scenarios."
                )
                sys.exit(1)

    print(
        f"Failure Generator: {len(names)} scenario(s), {args.count} variations each, seed={args.seed}"
    )
    run_scenarios(
        names, args.count, args.seed, args.output_dir, args.labels_dir, args.realtime
    )
    print("Done.")


if __name__ == "__main__":
    main()
