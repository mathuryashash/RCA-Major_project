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


def test_collector_survives_tick_failures_but_gives_up_eventually():
    """A transient fault must not end a run building an hours-long baseline.

    But a collector failing every single tick should stop rather than write a
    log entry every thirty seconds forever.
    """
    from telemetry import collector as collector_module

    class _AlwaysFails(collector_module.Collector):
        def __init__(self):                       # no database needed
            self.calls = 0

        def run_once(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("sampler exploded")

    broken = _AlwaysFails()
    import telemetry.config as config_module

    original = config_module.SYSTEM_CADENCE_S
    config_module.SYSTEM_CADENCE_S = 0            # do not sleep through the test
    try:
        broken.run_forever()
    finally:
        config_module.SYSTEM_CADENCE_S = original

    assert broken.calls == collector_module.MAX_CONSECUTIVE_FAILURES


def test_a_new_collector_honours_an_existing_pause(tmp_path, monkeypatch):
    """Pause must survive a collector being started by something else.

    The desktop app starts a collector every time it opens, and the logon
    entry starts one at every sign-in. run_forever() used to delete the stop
    flag on entry, so either of those silently undid a pause the user had
    asked for -- and the UI would still say "Paused". Only resume (or
    restart, from the UI) may lift it.
    """
    from telemetry import config as config_module

    monkeypatch.setattr(config_module, "app_dir", lambda: tmp_path)
    config_module.stop_flag_path().touch()

    class _Counts(collector.Collector):
        def __init__(self):
            self.calls = 0

        def run_once(self, *args, **kwargs):
            self.calls += 1

    instance = _Counts()
    monkeypatch.setattr(config_module, "SYSTEM_CADENCE_S", 0)
    instance.run_forever()
    assert instance.calls == 0, "a paused collector must not sample"
    assert config_module.stop_flag_path().exists(), "the pause must still be in place"
