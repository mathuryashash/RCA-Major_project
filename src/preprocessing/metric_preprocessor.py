"""
Module 2B: Metric Preprocessing
Handles missing data, z-score normalization, and windowing for time-series metrics.
Outputs HDF5 arrays ready for LSTM Autoencoder and Temporal Transformer.
"""

import numpy as np
import pandas as pd

# Lazy import for h5py — not every environment has it installed
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    H5PY_AVAILABLE = False


class MetricPreprocessor:
    """Preprocesses metric time-series for downstream anomaly-detection models.

    Pipeline: fill_gaps → normalize → create_windows → (optional) save_hdf5.
    """

    def __init__(self, config: dict | None = None):
        """Initialise with an optional config dict.

        Expected keys (all optional, sane defaults applied):
            - window_size (int): timesteps per window, default 60
            - threshold_percentile (int): anomaly threshold, default 99
            - alpha (float): model_score vs rarity_prior weight, default 0.80
        """
        config = config or {}
        self.window_size: int = config.get("window_size", 60)
        self.threshold_percentile: int = config.get("threshold_percentile", 99)
        self.alpha: float = config.get("alpha", 0.80)

        # Populated by normalize(), consumed by denormalize()
        self.norm_means_: pd.DataFrame | None = None
        self.norm_stds_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # FR-12: Missing Data Handling
    # ------------------------------------------------------------------

    def fill_gaps(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Handle missing data in metric time series.

        Strategy (tiered):
            - Gaps <5 minutes  : forward-fill
            - Gaps 5–30 minutes: linear interpolation
            - Gaps >30 minutes : left as NaN (sentinel), excluded from windows

        Args:
            df: DataFrame with DatetimeIndex and metric columns.
            freq: Expected sampling frequency (default ``'1min'``).

        Returns:
            DataFrame reindexed to *freq* with gaps handled.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")

        # 1. Reindex to a regular frequency grid
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)

        # 2. Identify gap lengths (consecutive NaN runs) per column
        result = df.copy()

        for col in result.columns:
            series = result[col]
            is_null = series.isna()

            # Label each contiguous NaN run with a unique group id
            group_ids = (~is_null).cumsum()
            # Only care about NaN positions
            nan_groups = group_ids[is_null]

            if nan_groups.empty:
                continue

            gap_lengths = nan_groups.groupby(nan_groups).transform("count")

            freq_minutes = pd.tseries.frequencies.to_offset(freq).nanos / 60e9

            # Short gaps (<5 min): forward-fill
            short_mask = is_null & (gap_lengths * freq_minutes < 5)
            # Medium gaps (5-30 min): will interpolate
            medium_mask = (
                is_null
                & (gap_lengths * freq_minutes >= 5)
                & (gap_lengths * freq_minutes <= 30)
            )
            # Long gaps (>30 min): stay NaN

            # Apply forward-fill only to short-gap positions
            if short_mask.any():
                filled = series.ffill()
                result.loc[short_mask, col] = filled.loc[short_mask]

            # Apply linear interpolation only to medium-gap positions
            if medium_mask.any():
                interpolated = series.interpolate(method="linear")
                result.loc[medium_mask, col] = interpolated.loc[medium_mask]

        return result

    # ------------------------------------------------------------------
    # FR-13: Z-Score Normalization
    # ------------------------------------------------------------------

    def normalize(self, df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
        """Apply rolling z-score normalization per metric.

        Uses a rolling window of *window_days* days to compute mean and std,
        then normalises as ``(x - mean) / std``.  Normalization parameters
        are stored as ``self.norm_means_`` and ``self.norm_stds_`` so that
        :meth:`denormalize` can invert the transform.

        Args:
            df: DataFrame with metric columns (and DatetimeIndex).
            window_days: Rolling window size in days (default 7).

        Returns:
            Normalized DataFrame (same shape, same index).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex")

        freq = pd.infer_freq(df.index)
        if freq is None:
            # Fall back: estimate from median diff
            median_diff = df.index.to_series().diff().median()
        else:
            median_diff = pd.tseries.frequencies.to_offset(freq)

        # Convert window_days to number of periods
        window_td = pd.Timedelta(days=window_days)
        periods = max(int(window_td / median_diff), 1)

        rolling_mean = df.rolling(window=periods, min_periods=1).mean()
        rolling_std = df.rolling(window=periods, min_periods=1).std()

        # Avoid division by zero / NaN: replace 0 and NaN std with 1
        rolling_std = rolling_std.fillna(1.0).replace(0, 1.0)

        self.norm_means_ = rolling_mean
        self.norm_stds_ = rolling_std

        normalized = (df - rolling_mean) / rolling_std
        return normalized

    def denormalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse z-score normalization using stored parameters.

        Raises:
            RuntimeError: If :meth:`normalize` has not been called first.
        """
        if self.norm_means_ is None or self.norm_stds_ is None:
            raise RuntimeError(
                "No normalization parameters stored. Call normalize() first."
            )
        return df * self.norm_stds_ + self.norm_means_

    # ------------------------------------------------------------------
    # FR-14: Windowing
    # ------------------------------------------------------------------

    def create_windows(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """Create overlapping sliding windows for model input.

        Windows that contain **any** NaN sentinel values (from >30-min gaps)
        are automatically excluded.

        Args:
            df: Normalized metric DataFrame.
            window_size: Number of timesteps per window (default 60).
            overlap: Overlap fraction in [0, 1) (default 0.5 = 50 %).

        Returns:
            ``np.ndarray`` of shape ``(n_windows, window_size, n_metrics)``.
        """
        values = df.values  # shape (T, n_metrics)
        n_samples, n_metrics = values.shape
        stride = max(int(window_size * (1 - overlap)), 1)

        windows: list[np.ndarray] = []
        for start in range(0, n_samples - window_size + 1, stride):
            window = values[start : start + window_size]
            if np.isnan(window).any():
                continue  # skip windows tainted by NaN sentinels
            windows.append(window)

        if not windows:
            return np.empty((0, window_size, n_metrics), dtype=np.float64)

        return np.stack(windows, axis=0)

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def save_hdf5(
        self, windows: np.ndarray, path: str, metadata: dict | None = None
    ) -> None:
        """Save windowed arrays to HDF5 format.

        Args:
            windows: Array of shape ``(batch, window_size, n_metrics)``.
            path: Output file path.
            metadata: Optional dict of scalar/string attributes
                      (metric names, timestamps, etc.).

        Raises:
            ImportError: If *h5py* is not installed.
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 I/O. Install it with: pip install h5py"
            )

        with h5py.File(path, "w") as f:
            f.create_dataset("windows", data=windows, compression="gzip")
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value

    @staticmethod
    def load_hdf5(path: str) -> tuple[np.ndarray, dict]:
        """Load windowed arrays from HDF5.

        Args:
            path: Path to an HDF5 file produced by :meth:`save_hdf5`.

        Returns:
            ``(windows, metadata)`` where *windows* is an ndarray and
            *metadata* is a dict of file-level attributes.

        Raises:
            ImportError: If *h5py* is not installed.
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 I/O. Install it with: pip install h5py"
            )

        with h5py.File(path, "r") as f:
            windows = f["windows"][:]
            metadata = dict(f.attrs)

        return windows, metadata

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        df: pd.DataFrame,
        output_path: str | None = None,
        window_size: int = 60,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """Full preprocessing pipeline: fill_gaps → normalize → create_windows.

        Optionally persists the result as HDF5.

        Args:
            df: Raw metric DataFrame with DatetimeIndex.
            output_path: If given, save the windowed array to this HDF5 path.
            window_size: Timesteps per window (default 60).
            overlap: Overlap fraction (default 0.5).

        Returns:
            ``np.ndarray`` of shape ``(n_windows, window_size, n_metrics)``.
        """
        filled = self.fill_gaps(df)
        normalized = self.normalize(filled)
        windows = self.create_windows(
            normalized, window_size=window_size, overlap=overlap
        )

        if output_path is not None:
            metadata = {
                "n_windows": windows.shape[0],
                "window_size": window_size,
                "n_metrics": windows.shape[2] if windows.ndim == 3 else 0,
                "overlap": overlap,
            }
            self.save_hdf5(windows, output_path, metadata=metadata)

        return windows
