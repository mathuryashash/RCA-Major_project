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
