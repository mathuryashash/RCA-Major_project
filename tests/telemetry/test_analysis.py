import pandas as pd

from telemetry.analysis import baseline_status, clean_baseline, contiguous_windows, modelled_features


def _samples(count=70):
    ts = pd.date_range("2026-01-01", periods=count, freq="30s", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "elapsed_ms": [None] + [30000] * (count - 1), "cpu_pct": 1.0, "mem_pct": 2.0, "battery_pct": None})


def test_baseline_excludes_post_start_and_event_leadup():
    samples = _samples(200)
    events = pd.DataFrame({"event_id": [41], "timestamp": [samples.iloc[150]["timestamp"]]})
    clean = clean_baseline(samples, events)
    assert len(clean) < len(samples) and clean.iloc[0]["elapsed_ms"] == 30000


def test_nullable_battery_does_not_disqualify_all_rows():
    samples = _samples()
    clean = clean_baseline(samples, pd.DataFrame())
    assert len(clean) == len(samples) - 1
    assert "battery_pct" not in modelled_features(clean)


def test_contiguous_windows_splits_gap():
    samples = _samples(130)
    samples.loc[65:, "timestamp"] += pd.Timedelta(minutes=3)
    assert len(contiguous_windows(samples, minimum_samples=60)) == 2


def test_attribution_surfaces_a_memory_hog_that_uses_no_cpu(tmp_path):
    """Ranking by CPU alone cannot name the cause of a memory incident.

    Measured on an injected memory fault: the harness held 1.15 GB in a
    process that slept between allocations, so its CPU was ~0 and the top ten
    by CPU were SearchIndexer, WmiPrvSE, Taskmgr and MsMpEng. Detection
    worked and attribution named four innocents. max_rss_bytes was already
    being selected -- it was simply never sorted on.
    """
    from telemetry import store
    from telemetry.analysis import load_process_attribution

    path = tmp_path / "t.db"
    conn = store.connect(path)
    store.init_schema(conn)

    base = 1_800_000_000
    noisy = [
        {"pid": i, "create_time": 1.0, "name": f"busy{i}.exe", "cpu_pct": 50.0 - i,
         "cpu_time_delta_s": 1.0, "rss": 10_000_000, "io_read_delta": 0, "io_write_delta": 0}
        for i in range(12)
    ]
    hog = {"pid": 999, "create_time": 1.0, "name": "hog.exe", "cpu_pct": 0.0,
           "cpu_time_delta_s": 0.0, "rss": 1_150_000_000, "io_read_delta": 0, "io_write_delta": 0}
    for tick in range(3):
        store.insert_proc_samples(conn, base + tick * 30, noisy + [hog])
    conn.commit()

    frame = load_process_attribution(
        pd.Timestamp(base, unit="s", tz="UTC"),
        pd.Timestamp(base + 120, unit="s", tz="UTC"),
        path=path,
    )
    names = list(frame["name"])
    assert "hog.exe" in names, f"the memory hog must be attributable, got {names}"
    # The CPU-heavy processes must not be evicted to make room for it.
    assert "busy0.exe" in names
    assert len(names) <= 10
