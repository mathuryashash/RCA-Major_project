"""
Preprocessing Pipeline (Orchestrator)

Combines:
  1. TimeSeriesAligner  — align multi-source DataFrames to a common grid
  2. DataPreprocessor   — impute missing values, z-score normalise, build
                          sliding windows for the LSTM autoencoder

Typical usage
-------------
    from src.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(window_size=60, stride=5)

    # Fit on clean, normal-behaviour data
    pipeline.fit(normal_df)

    # Prepare windows for LSTM training
    train_windows, val_windows, test_windows = pipeline.prepare_train_data(normal_df)

    # At inference time, transform a new incoming batch
    windows, indices, timestamps = pipeline.transform_for_inference(new_df)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_normalizer import DataPreprocessor
from .time_series_aligner import TimeSeriesAligner


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for the RCA system.

    This is the single entry point for all data preparation:

    Stage 1 – Alignment
        • Resample each source to a common 5-min grid
        • Handle missing values / gaps introduced by resampling
        • Merge metric sources into one wide DataFrame

    Stage 2 – Normalisation (fit on normal data only)
        • Impute residual NaN values
        • Fit StandardScaler on normal operational data
        • Apply z-score normalisation to train / test / inference data

    Stage 3 – Windowing
        • Create overlapping sliding windows (shape: N × T × F)
          suitable for LSTM Autoencoder input

    Args:
        freq: Target resampling frequency (default '5min').
        window_size: Number of timesteps per LSTM window (default 60 = 5 h).
        stride: Step between consecutive windows during training (default 5).
        val_ratio: Fraction of normal data reserved for validation.
        test_ratio: Fraction of normal data reserved for test.
        interpolation_method: Gap-fill method passed to TimeSeriesAligner.
        timezone: Timezone normalisation (default 'UTC').
    """

    def __init__(
        self,
        freq: str = '5min',
        window_size: int = 60,
        stride: int = 5,
        val_ratio: float = 0.10,
        test_ratio: float = 0.15,
        interpolation_method: str = 'linear',
        timezone: str = 'UTC',
    ):
        self.freq = freq
        self.window_size = window_size
        self.stride = stride
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.aligner = TimeSeriesAligner(
            freq=freq,
            interpolation_method=interpolation_method,
            max_gap_fill=12,          # fill up to 12 × 5 min = 60 min gaps
            timezone=timezone,
        )
        self.preprocessor = DataPreprocessor(
            window_size=window_size,
            stride=stride,
        )

        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def align(
        self,
        dataframes: List[pd.DataFrame],
        source_names: Optional[List[str]] = None,
        merge_strategy: str = 'outer',
        prefix_columns: bool = False,
    ) -> pd.DataFrame:
        """
        Align and merge a list of metric DataFrames into one wide DataFrame.

        Convenience wrapper around TimeSeriesAligner.  If only a single
        DataFrame is provided it is still resampled to the target frequency.

        Args:
            dataframes: List of DataFrames (one per data source).
            source_names: Optional column-prefix names for each source.
            merge_strategy: 'inner' (intersection) or 'outer' (union).
            prefix_columns: Prefix each column with its source name.

        Returns:
            Single DataFrame with a regular DatetimeIndex.
        """
        if len(dataframes) == 1:
            return self.aligner.align_single(
                dataframes[0],
                source_name=(source_names[0] if source_names else 'default'),
            )

        aligned_list = self.aligner.align_many(dataframes, source_names=source_names)
        return self.aligner.merge_sources(
            aligned_list,
            source_names=source_names,
            strategy=merge_strategy,
            prefix_columns=prefix_columns,
        )

    def fit(self, normal_df: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Fit the normaliser (StandardScaler) on normal/healthy data.

        Must be called before any call to transform_* methods.
        The DataFrame should contain ONLY normal operational data
        (no injected failures) so the scaler learns true healthy ranges.

        Args:
            normal_df: Aligned metric DataFrame from normal operation.

        Returns:
            self (for method chaining).
        """
        self.preprocessor.fit(normal_df)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Training-time preparation
    # ------------------------------------------------------------------

    def prepare_train_data(
        self,
        normal_df: pd.DataFrame,
        stride_override: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split normal data -> train / val / test windows for LSTM training.

        This method fits the scaler if it hasn't been fitted yet.

        Args:
            normal_df: Aligned normal-behaviour DataFrame.
            stride_override: Override the instance stride for window creation.

        Returns:
            Tuple of (train_windows, val_windows, test_windows), each of
            shape (N, window_size, n_features) as float32.
        """
        self._assert_fitted_or_fit(normal_df)

        train_df, val_df, test_df = self.aligner.split_train_test(
            normal_df,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
        )

        s = stride_override or self.stride

        train_windows, _, _ = self.preprocessor.create_windows_from_df(train_df, stride=s)
        val_windows, _, _ = self.preprocessor.create_windows_from_df(val_df, stride=s)
        test_windows, _, _ = self.preprocessor.create_windows_from_df(test_df, stride=s)

        return train_windows, val_windows, test_windows

    # ------------------------------------------------------------------
    # Inference-time transformation
    # ------------------------------------------------------------------

    def transform_for_inference(
        self,
        df: pd.DataFrame,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Normalise a DataFrame and create windows for real-time inference.

        Uses stride=1 by default so every timestep gets its own window
        (dense coverage for latency-sensitive anomaly detection).

        Args:
            df: Incoming metric DataFrame (aligned to the same grid used
                during training).
            stride: Window step size (default 1 for dense inference coverage).

        Returns:
            windows:    (N, window_size, n_features) float32 array
            indices:    Start position of each window in df
            timestamps: End timestamp of each window (for result labelling)
        """
        self._assert_fitted()
        return self.preprocessor.create_windows_from_df(df, stride=stride)

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalise a DataFrame -> flat (T, F) array without windowing.

        Useful for statistical / causal-inference stages that operate on
        individual timesteps rather than LSTM windows.

        Returns:
            Normalised float32 array of shape (T, n_features).
        """
        self._assert_fitted()
        return self.preprocessor.transform(df)

    # ------------------------------------------------------------------
    # Full end-to-end convenience method
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        raw_dataframes: List[pd.DataFrame],
        source_names: Optional[List[str]] = None,
        fit: bool = True,
    ) -> Dict:
        """
        One-shot method: align -> validate -> fit -> create windows.

        Suitable for running from a script or notebook.

        Args:
            raw_dataframes: List of raw metric DataFrames.
            source_names: Optional source labels.
            fit: Whether to (re-)fit the scaler.  Set False to reuse an
                 already-fitted pipeline.

        Returns:
            dict with keys:
            - 'aligned_df':     merged aligned DataFrame
            - 'quality_report': data-quality dict
            - 'train_windows':  (N_train, T, F)
            - 'val_windows':    (N_val,   T, F)
            - 'test_windows':   (N_test,  T, F)
            - 'feature_names':  list of metric column names
        """
        # Stage 1: Align
        aligned_df = self.align(raw_dataframes, source_names=source_names)

        # Stage 2a: Validate quality
        quality_report = self.preprocessor.validate_data_quality(aligned_df)

        # Stage 2b: Fit normaliser
        if fit or not self.is_fitted:
            self.fit(aligned_df)

        # Stage 3: Windows
        train_windows, val_windows, test_windows = self.prepare_train_data(aligned_df)

        return {
            'aligned_df': aligned_df,
            'quality_report': quality_report,
            'train_windows': train_windows,
            'val_windows': val_windows,
            'test_windows': test_windows,
            'feature_names': self.preprocessor.get_feature_names(),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of metric features (available after fit)."""
        return self.preprocessor.n_features

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names (available after fit)."""
        return self.preprocessor.get_feature_names()

    def _assert_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "PreprocessingPipeline is not fitted. "
                "Call fit() with normal operational data first."
            )

    def _assert_fitted_or_fit(self, df: pd.DataFrame):
        """Fit automatically if not yet fitted (convenience for training scripts)."""
        if not self.is_fitted:
            self.fit(df)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    print("=== PreprocessingPipeline Smoke Test ===\n")

    gen = SyntheticMetricsGenerator(seed=42)
    normal_df = gen.generate_normal_behavior(duration_days=30)

    print(f"Raw normal data: {normal_df.shape[0]} samples × {normal_df.shape[1]} features")

    # --- Full pipeline in one call ---
    pipeline = PreprocessingPipeline(
        freq='5min',
        window_size=60,
        stride=10,          # stride=10 keeps training set manageable
        val_ratio=0.10,
        test_ratio=0.15,
    )

    result = pipeline.run_full_pipeline(
        raw_dataframes=[normal_df],
        source_names=['synthetic'],
        fit=True,
    )

    print(f"\nAligned DataFrame shape : {result['aligned_df'].shape}")
    print(f"Data quality score      : {result['quality_report']['quality_score']}")
    print(f"Quality issues          : {result['quality_report']['issues'] or 'None'}")
    print(f"\nTrain windows shape : {result['train_windows'].shape}")
    print(f"Val   windows shape : {result['val_windows'].shape}")
    print(f"Test  windows shape : {result['test_windows'].shape}")
    print(f"\nFeatures ({pipeline.n_features}): {pipeline.feature_names}")

    # --- Inference transform ---
    failure_df, _ = gen.inject_failure_scenario(
        normal_df,
        failure_type='cpu_spike',
        start_idx=500,
        duration_samples=100,
    )

    inf_windows, inf_indices, inf_ts = pipeline.transform_for_inference(failure_df, stride=1)
    print(f"\nInference windows shape : {inf_windows.shape}")
    print(f"First window ends at    : {inf_ts[0]}")
    print("\n✓ PreprocessingPipeline smoke test passed.")
