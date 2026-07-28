from telemetry import collector, store


def _raw():
    return {"gauges": {
        "cpu_pct": 1.0, "cpu_pct_max_core": 1.0, "cpu_freq_mhz": 1000.0, "cpu_freq_ratio": 0.5,
        "mem_pct": 40.0, "mem_available_mb": 100.0, "mem_used_bytes": 10, "swap_pct": 0.0,
        "swap_used_bytes": 0, "disk_free_pct": 50.0, "process_count": 2, "battery_pct": None,
        "power_plugged": None, "on_battery": None, "user_idle_sec": None, "foreground_app": None,
    }, "counters": {"cpu_busy_s": 1.0, "disk_read_bytes": 1.0, "disk_write_bytes": 1.0,
                       "disk_busy_ms": 1.0, "net_sent_bytes": 1.0, "net_recv_bytes": 1.0,
                       "swap_used_bytes": 0.0, "battery_pct_counter": 0.0}}


def test_consent_is_required_and_gap_resets_rates(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "collector.db")
    store.init_schema(conn)
    instance = collector.Collector(conn)
    monkeypatch.setattr(collector.sampler, "sample_system_raw", _raw)
    try:
        instance.run_once(now=1_000, mono_now=1.0)
    except PermissionError:
        pass
    else:
        raise AssertionError("collection must require consent")
    collector.grant_consent(conn)
    instance.run_once(now=1_000, mono_now=1.0)
    instance.run_once(now=1_030, mono_now=31.0)
    instance.run_once(now=1_120, mono_now=121.0)
    rows = conn.execute("SELECT ts, elapsed_ms FROM samples ORDER BY ts").fetchall()
    assert rows == [(1000, None), (1030, 30000), (1120, None)]


def test_burst_logic_handles_missing_values():
    assert collector.Collector.should_burst({"cpu_pct": 81})
    assert not collector.Collector.should_burst({"cpu_pct": None, "mem_pct": None, "disk_busy_pct": None})
