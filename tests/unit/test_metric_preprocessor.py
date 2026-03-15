"""
Unit tests for src.preprocessing.metric_preprocessor.MetricPreprocessor
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.metric_preprocessor import MetricPreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regular_df(
    n_rows: int = 200, n_cols: int = 3, freq: str = "1min"
) -> pd.DataFrame:
    """Create a regular metric DataFrame with no gaps."""
    rng = np.random.RandomState(42)
    idx = pd.date_range("2026-01-01", periods=n_rows, freq=freq)
    data = rng.randn(n_rows, n_cols) * 10 + 50  # mean≈50, std≈10
    cols = [f"metric_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _inject_gap(df: pd.DataFrame, start_offset: int, gap_minutes: int) -> pd.DataFrame:
    """Remove rows from *df* to create a gap of *gap_minutes* starting at *start_offset*."""
    drop_idx = df.index[start_offset : start_offset + gap_minutes]
    return df.drop(drop_idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFillGaps:
    """FR-12: Missing-data handling."""

    def test_fill_gaps_short(self):
        """Gaps < 5 min are forward-filled (no NaN remaining)."""
        df = _make_regular_df(120)
        # Remove 3 consecutive minutes → 3-min gap (< 5 min)
        gapped = _inject_gap(df, start_offset=20, gap_minutes=3)

        pp = MetricPreprocessor()
        filled = pp.fill_gaps(gapped)

        # The gap region in the reindexed frame should have no NaN
        gap_region = filled.iloc[20:23]
        assert not gap_region.isna().any().any(), (
            "Short gap (<5 min) was not forward-filled"
        )

    def test_fill_gaps_medium(self):
        """Gaps 5–30 min are linearly interpolated."""
        df = _make_regular_df(120)
        # Remove 15 consecutive minutes → 15-min gap
        gapped = _inject_gap(df, start_offset=30, gap_minutes=15)

        pp = MetricPreprocessor()
        filled = pp.fill_gaps(gapped)

        gap_region = filled.iloc[30:45]
        assert not gap_region.isna().any().any(), (
            "Medium gap (5-30 min) was not interpolated"
        )

        # Interpolated values should be *between* the boundary values
        for col in df.columns:
            lo = min(df.iloc[29][col], df.iloc[45][col])
            hi = max(df.iloc[29][col], df.iloc[45][col])
            # Allow small tolerance for floating point
            assert gap_region[col].between(lo - 1e-9, hi + 1e-9).all(), (
                f"Interpolated values for {col} fall outside boundary range"
            )

    def test_fill_gaps_long(self):
        """Gaps > 30 min remain NaN (sentinel for exclusion)."""
        df = _make_regular_df(200)
        # Remove 35 consecutive minutes → 35-min gap
        gapped = _inject_gap(df, start_offset=50, gap_minutes=35)

        pp = MetricPreprocessor()
        filled = pp.fill_gaps(gapped)

        gap_region = filled.iloc[50:85]
        assert gap_region.isna().any().any(), "Long gap (>30 min) should remain NaN"


@pytest.mark.unit
class TestNormalize:
    """FR-13: Z-score normalization."""

    def test_normalize_zscore(self):
        """Normalized columns should have approximately 0 mean, 1 std."""
        df = _make_regular_df(n_rows=10_080, n_cols=3)  # 7 days at 1-min

        pp = MetricPreprocessor()
        normed = pp.normalize(df, window_days=7)

        # Check the latter half (rolling window is fully populated)
        latter = normed.iloc[len(normed) // 2 :]
        for col in normed.columns:
            col_mean = latter[col].mean()
            col_std = latter[col].std()
            assert abs(col_mean) < 0.5, f"{col} mean {col_mean:.3f} not near 0"
            assert 0.3 < col_std < 2.0, f"{col} std {col_std:.3f} not near 1"

    def test_normalize_denormalize_roundtrip(self):
        """denormalize(normalize(x)) should recover the original values."""
        df = _make_regular_df(n_rows=1_000, n_cols=2)

        pp = MetricPreprocessor()
        normed = pp.normalize(df, window_days=7)
        recovered = pp.denormalize(normed)

        np.testing.assert_allclose(
            recovered.values,
            df.values,
            atol=1e-10,
            err_msg="Round-trip normalization failed",
        )


@pytest.mark.unit
class TestCreateWindows:
    """FR-14: Windowing."""

    def test_create_windows_shape(self):
        """Output shape must be (n_windows, window_size, n_metrics)."""
        df = _make_regular_df(n_rows=200, n_cols=4)

        pp = MetricPreprocessor()
        windows = pp.create_windows(df, window_size=60, overlap=0.5)

        assert windows.ndim == 3
        assert windows.shape[1] == 60
        assert windows.shape[2] == 4

    def test_create_windows_overlap(self):
        """50 % overlap should produce the correct number of windows."""
        n_rows, window_size, overlap = 200, 60, 0.5
        df = _make_regular_df(n_rows=n_rows, n_cols=2)

        stride = int(window_size * (1 - overlap))
        expected_n = (n_rows - window_size) // stride + 1

        pp = MetricPreprocessor()
        windows = pp.create_windows(df, window_size=window_size, overlap=overlap)

        assert windows.shape[0] == expected_n, (
            f"Expected {expected_n} windows, got {windows.shape[0]}"
        )

    def test_create_windows_nan_skip(self):
        """Windows containing NaN sentinels (from >30-min gaps) are excluded."""
        df = _make_regular_df(n_rows=200, n_cols=2)
        # Inject NaNs at rows 100-105 to simulate a long gap sentinel
        df.iloc[100:106] = np.nan

        pp = MetricPreprocessor()
        windows = pp.create_windows(df, window_size=60, overlap=0.5)

        # Every returned window must be NaN-free
        for i in range(windows.shape[0]):
            assert not np.isnan(windows[i]).any(), f"Window {i} contains NaN"

        # We should have *fewer* windows than if there were no NaNs
        windows_clean = pp.create_windows(
            _make_regular_df(n_rows=200, n_cols=2),
            window_size=60,
            overlap=0.5,
        )
        assert windows.shape[0] < windows_clean.shape[0], (
            "NaN-tainted windows were not excluded"
        )


@pytest.mark.unit
class TestHDF5IO:
    """HDF5 save / load round-trip."""

    def test_hdf5_save_load_roundtrip(self, tmp_dir):
        """save_hdf5 → load_hdf5 must produce an identical array."""
        h5py = pytest.importorskip("h5py")  # skip if h5py not installed

        pp = MetricPreprocessor()
        rng = np.random.RandomState(0)
        windows = rng.randn(10, 60, 3)
        path = os.path.join(tmp_dir, "test_windows.h5")

        metadata = {"metric_names": "cpu,mem,disk", "overlap": 0.5}
        pp.save_hdf5(windows, path, metadata=metadata)

        loaded_windows, loaded_meta = MetricPreprocessor.load_hdf5(path)

        np.testing.assert_array_equal(loaded_windows, windows)
        assert loaded_meta["metric_names"] == "cpu,mem,disk"
        assert float(loaded_meta["overlap"]) == 0.5


@pytest.mark.unit
class TestFullPipeline:
    """End-to-end: raw DataFrame → processed windows."""

    def test_full_process_pipeline(self, tmp_dir):
        """process() should return a valid 3-D array and optionally save HDF5."""
        h5py = pytest.importorskip("h5py")

        df = _make_regular_df(n_rows=200, n_cols=3)

        pp = MetricPreprocessor(config={"window_size": 60})
        out_path = os.path.join(tmp_dir, "pipeline_out.h5")
        windows = pp.process(df, output_path=out_path, window_size=60, overlap=0.5)

        # Shape checks
        assert windows.ndim == 3
        assert windows.shape[1] == 60
        assert windows.shape[2] == 3
        assert windows.shape[0] > 0

        # HDF5 file should exist and be loadable
        assert os.path.exists(out_path)
        loaded, meta = MetricPreprocessor.load_hdf5(out_path)
        np.testing.assert_array_equal(loaded, windows)
        assert int(meta["n_windows"]) == windows.shape[0]
