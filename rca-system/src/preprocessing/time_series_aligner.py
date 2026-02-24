"""
Time-Series Aligner

Handles alignment of multi-source metric DataFrames to a common,
evenly-spaced time grid before they are fed to the preprocessing
and anomaly-detection pipeline.

Typical usage:
    aligner = TimeSeriesAligner(freq='5min')
    aligned = aligner.align([prometheus_df, cloudwatch_df, app_metrics_df])
    merged   = aligner.merge_sources(aligned, strategy='outer')
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesAligner:
    """
    Aligns multiple metric DataFrames to a common DatetimeIndex with a
    fixed sampling frequency.

    All DataFrames are expected to have a DatetimeIndex (or a 'timestamp'
    column that will be converted to one).  After alignment, every DataFrame
    will share the same time grid so they can safely be concatenated or
    merged column-wise.

    Args:
        freq: Target sampling frequency, e.g. '5min', '1min', '1H'.
        interpolation_method: How to fill values created during up-sampling.
            'linear' (default) works well for smooth metrics.
            'ffill' is suitable for step-metric counters.
            'nearest' is a fast fallback.
        max_gap_fill: Maximum consecutive NaN values to interpolate before
            falling back to column median.  Expressed in number of target
            samples (e.g. 12 -> 60 min at 5-min freq).
        timezone: If provided, all timestamps are converted to this tz.
            Example: 'UTC', 'US/Eastern'.
    """

    def __init__(
        self,
        freq: str = '5min',
        interpolation_method: str = 'linear',
        max_gap_fill: int = 12,
        timezone: Optional[str] = 'UTC',
    ):
        self.freq = freq
        self.interpolation_method = interpolation_method
        self.max_gap_fill = max_gap_fill
        self.timezone = timezone

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align_single(self, df: pd.DataFrame, source_name: str = '') -> pd.DataFrame:
        """
        Align a single DataFrame to the target frequency.

        Steps:
        1.  Ensure DatetimeIndex
        2.  Convert timezone
        3.  Drop duplicate timestamps (keep last)
        4.  Sort by time
        5.  Resample to target frequency (aggregate duplicate rows)
        6.  Reindex to a regular time grid
        7.  Fill gaps via interpolation then median fallback

        Args:
            df: Input DataFrame.  Must have DatetimeIndex or 'timestamp' column.
            source_name: Label used in warning messages.

        Returns:
            DataFrame with a regular DatetimeIndex at `self.freq`.
        """
        df = self._ensure_datetime_index(df, source_name)
        df = self._harmonize_timezone(df, source_name)

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        # Resample to target freq (mean aggregation for downsampling)
        df = df.resample(self.freq).mean()

        # Fill gaps
        df = self._fill_gaps(df, source_name)

        return df

    def align_many(
        self, dataframes: List[pd.DataFrame], source_names: Optional[List[str]] = None
    ) -> List[pd.DataFrame]:
        """
        Align a list of DataFrames, each to the same common time grid.

        The common grid spans from the latest start to the earliest end
        across all DataFrames.  Frames not covering the full range are
        trimmed automatically.

        Args:
            dataframes: List of metric DataFrames (possibly from different sources).
            source_names: Optional labels for each DataFrame (used in warnings).

        Returns:
            List of aligned DataFrames, all sharing the same DatetimeIndex.
        """
        if not dataframes:
            return []

        names = source_names or [f'source_{i}' for i in range(len(dataframes))]

        # First pass: align each independently
        aligned = [
            self.align_single(df, name)
            for df, name in zip(dataframes, names)
        ]

        # Compute common time range
        common_start, common_end = self._common_time_range(aligned)
        if common_start is None:
            warnings.warn("No overlapping time range found across DataFrames.")
            return aligned

        # Second pass: restrict all frames to common range
        common_index = pd.date_range(start=common_start, end=common_end, freq=self.freq)
        result = []
        for df, name in zip(aligned, names):
            df_reindexed = df.reindex(common_index)
            df_reindexed = self._fill_gaps(df_reindexed, name)
            result.append(df_reindexed)

        return result

    def merge_sources(
        self,
        dataframes: List[pd.DataFrame],
        source_names: Optional[List[str]] = None,
        strategy: str = 'outer',
        prefix_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Merge multiple aligned DataFrames into a single wide DataFrame.

        Args:
            dataframes: Already-aligned DataFrames (use align_many() first).
            source_names: Optional prefixes for column renaming.
            strategy: Pandas join strategy: 'inner' or 'outer'.
            prefix_columns: If True, prefix each column with its source name
                            to avoid collisions.

        Returns:
            Wide DataFrame combining all metrics.
        """
        if not dataframes:
            return pd.DataFrame()

        names = source_names or [f'src{i}' for i in range(len(dataframes))]

        frames = []
        for df, name in zip(dataframes, names):
            if prefix_columns:
                df = df.add_prefix(f'{name}_')
            frames.append(df)

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.join(f, how=strategy)

        return merged

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological train / validation / test split (no shuffling).

        Args:
            df: Aligned metric DataFrame (full timeline).
            test_ratio: Fraction of data for test set.
            val_ratio: Fraction of data for validation set.

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        train = df.iloc[:n_train]
        val = df.iloc[n_train: n_train + n_val]
        test = df.iloc[n_train + n_val:]

        return train, val, test

    def get_alignment_report(self, df: pd.DataFrame, source_name: str = '') -> Dict:
        """
        Generate a diagnostic report about a DataFrame's time coverage.

        Returns dict with:
        - start_time, end_time: datetime boundaries
        - total_samples: number of rows
        - freq_detected: detected sampling frequency
        - missing_pct: % of NaN values
        - gaps: list of (start, end, duration_min) for gaps > max_gap_fill
        - is_regular: whether the index is perfectly evenly spaced
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'error': 'DataFrame does not have a DatetimeIndex'}

        freq_detected = pd.infer_freq(df.index)
        total_samples = len(df)
        missing_pct = df.isna().mean().mean() * 100

        # Find gaps
        diffs = pd.Series(df.index).diff().dt.total_seconds() / 60  # minutes
        target_gap_min = pd.tseries.frequencies.to_offset(self.freq).nanos / 1e9 / 60
        gap_mask = diffs > target_gap_min * (self.max_gap_fill + 1)
        gaps = []
        for i in diffs[gap_mask].index:
            gaps.append({
                'gap_start': str(df.index[i - 1]),
                'gap_end': str(df.index[i]),
                'duration_min': round(float(diffs.iloc[i]), 1),
            })

        is_regular = bool(
            freq_detected is not None and total_samples > 1
            and (diffs.dropna().std() < 1.0)  # < 1 min jitter
        )

        return {
            'source': source_name,
            'start_time': str(df.index[0]) if total_samples else None,
            'end_time': str(df.index[-1]) if total_samples else None,
            'total_samples': total_samples,
            'freq_detected': str(freq_detected),
            'missing_pct': round(missing_pct, 2),
            'gaps_above_threshold': gaps,
            'is_regular': is_regular,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_datetime_index(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Convert 'timestamp' column -> DatetimeIndex, or validate existing index."""
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.set_index('timestamp')
            df.index.name = 'timestamp'
        elif isinstance(df.index, pd.DatetimeIndex):
            pass  # Already good
        else:
            raise ValueError(
                f"[{source_name}] DataFrame must have a DatetimeIndex or a 'timestamp' column. "
                f"Got index type: {type(df.index).__name__}"
            )
        return df

    def _harmonize_timezone(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Ensure all timestamps are in the target timezone (default: UTC)."""
        if self.timezone is None:
            return df
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(self.timezone)
        return df

    def _fill_gaps(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Fill NaN values introduced by resampling."""
        df = df.copy()
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue

            if self.interpolation_method == 'ffill':
                df[col] = df[col].ffill(limit=self.max_gap_fill)
            else:
                df[col] = df[col].interpolate(
                    method=self.interpolation_method,
                    limit=self.max_gap_fill,
                    limit_direction='both',
                )

            # Any remaining NaN -> column median
            remaining = int(df[col].isna().sum())
            if remaining > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                warnings.warn(
                    f"[{source_name}] Column '{col}': {remaining}/{null_count} NaN values "
                    f"could not be interpolated -> filled with median ({median_val:.4g})."
                )
        return df

    def _common_time_range(
        self, aligned: List[pd.DataFrame]
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Find the overlapping time range across a list of aligned DataFrames."""
        starts = [df.index[0] for df in aligned if len(df) > 0]
        ends = [df.index[-1] for df in aligned if len(df) > 0]

        if not starts or not ends:
            return None, None

        common_start = max(starts)
        common_end = min(ends)

        if common_start >= common_end:
            return None, None

        return common_start, common_end


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    print("=== TimeSeriesAligner Smoke Test ===\n")
    gen = SyntheticMetricsGenerator(seed=42)

    # Simulate two separate sources with slightly different start times & a gap
    df_a = gen.generate_normal_behavior(duration_days=7)
    df_b = df_a[['cpu_utilization', 'memory_usage_percent']].copy()

    # Introduce irregular sampling / gap in df_b
    drop_mask = (df_b.index.hour == 3)  # drop 3 AM readings
    df_b = df_b[~drop_mask]

    aligner = TimeSeriesAligner(freq='5min', interpolation_method='linear')

    print("Aligning individual DataFrames...")
    aligned_a, aligned_b = aligner.align_many([df_a, df_b], source_names=['api', 'infra'])
    print(f"  Source A (api):   {len(aligned_a)} rows, {aligned_a.shape[1]} cols")
    print(f"  Source B (infra): {len(aligned_b)} rows, {aligned_b.shape[1]} cols")

    print("\nMerging sources...")
    merged = aligner.merge_sources(
        [aligned_a, aligned_b],
        source_names=['api', 'infra'],
        strategy='inner',
        prefix_columns=True,
    )
    print(f"  Merged shape: {merged.shape}")
    print(f"  Columns: {list(merged.columns)[:6]} ...")

    print("\nAlignment report for source A:")
    report = aligner.get_alignment_report(aligned_a, 'api')
    for k, v in report.items():
        if k != 'gaps_above_threshold':
            print(f"  {k}: {v}")
    print(f"  gaps_above_threshold: {len(report['gaps_above_threshold'])} detected")

    train, val, test = aligner.split_train_test(aligned_a)
    print(
        f"\nTrain/Val/Test split: {len(train)} / {len(val)} / {len(test)} rows\n"
    )
