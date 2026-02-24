"""
Data Normalizer & Preprocessor

Handles all data preprocessing steps required before feeding
metrics into the LSTM autoencoder and causal inference engine:
- Missing value imputation
- Normalization (z-score per metric)
- Sliding window creation for LSTM input
- Data quality validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings


class DataPreprocessor:
    """
    Preprocesses raw metric DataFrames for the RCA pipeline.

    Typical usage:
        preprocessor = DataPreprocessor()
        preprocessor.fit(normal_df)          # Learn normalization params from normal data
        X_train = preprocessor.transform(normal_df)  # Normalized array
        X_test  = preprocessor.transform(failure_df) # Same scale

    The fit() call MUST use only normal (healthy) data so the scaler
    learns true normal ranges.
    """

    def __init__(self, window_size: int = 60, stride: int = 1):
        """
        Args:
            window_size: Number of timesteps per LSTM input window (default: 60 = 5 hours at 5-min intervals)
            stride: Step between consecutive windows (use >1 to reduce dataset size during training)
        """
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the normalizer on normal (healthy) data.

        Args:
            df: DataFrame from SyntheticMetricsGenerator.generate_normal_behavior()
                or equivalent. Index should be DateTimeIndex.

        Returns:
            self (for chaining)
        """
        df_clean = self._impute_missing(df)
        self.feature_names = list(df_clean.columns)
        self.scaler.fit(df_clean.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize data using fitted scaler.

        Returns:
            Normalized array of shape (num_samples, num_features)
        """
        self._assert_fitted()
        df_clean = self._impute_missing(df[self.feature_names])
        return self.scaler.transform(df_clean.values).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call (use only for normal data)."""
        return self.fit(df).transform(df)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        """Reverse normalization back to original scale."""
        self._assert_fitted()
        raw = self.scaler.inverse_transform(arr)
        return pd.DataFrame(raw, columns=self.feature_names)

    # ------------------------------------------------------------------
    # Sliding window creation for LSTM
    # ------------------------------------------------------------------

    def create_windows(
        self,
        normalized_data: np.ndarray,
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows suitable for LSTM autoencoder input.

        Args:
            normalized_data: Array of shape (T, F) — time steps × features
            stride: Override instance stride

        Returns:
            windows: Array of shape (N, window_size, F)
            indices: Start index of each window in the original array
        """
        s = stride if stride is not None else self.stride
        T, F = normalized_data.shape
        windows, indices = [], []

        for i in range(0, T - self.window_size + 1, s):
            windows.append(normalized_data[i: i + self.window_size])
            indices.append(i)

        if not windows:
            raise ValueError(
                f"Data length {T} < window_size {self.window_size}. "
                "Provide more data or reduce window_size."
            )

        return np.array(windows, dtype=np.float32), np.array(indices)

    def create_windows_from_df(
        self,
        df: pd.DataFrame,
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Convenience: normalize a DataFrame and create windows in one call.

        Returns:
            windows: (N, window_size, num_features)
            indices: start positions in original dataframe
            window_end_timestamps: timestamp at the END of each window
        """
        normalized = self.transform(df)
        windows, indices = self.create_windows(normalized, stride=stride)

        # The "representative" timestamp for each window is its end timestamp
        end_positions = indices + self.window_size - 1
        safe_positions = np.minimum(end_positions, len(df) - 1)
        window_end_timestamps = df.index[safe_positions]

        return windows, indices, window_end_timestamps

    # ------------------------------------------------------------------
    # Data quality & imputation
    # ------------------------------------------------------------------

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values:
        - Gaps < 5 timesteps: forward-fill
        - Gaps 5–60 timesteps: linear interpolation
        - Larger gaps: warn and fill with column median
        """
        df = df.copy()

        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count == 0:
                continue

            # Forward-fill for small gaps
            df[col] = df[col].fillna(method='ffill', limit=5)

            # Interpolate for medium gaps
            df[col] = df[col].interpolate(method='linear', limit=60)

            # Fill remaining with median
            remaining_nulls = df[col].isna().sum()
            if remaining_nulls > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                warnings.warn(
                    f"Column '{col}': {null_count} missing values found. "
                    f"{remaining_nulls} large gaps filled with median ({median_val:.2f}). "
                    "Analysis confidence may be reduced."
                )

        return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Check data quality and flag potential issues.

        Returns:
            report dict with keys:
            - 'quality_score': 0–100 (higher = better)
            - 'issues': list of issue descriptions
            - 'missing_pct': % of values that are NaN
            - 'frozen_metrics': metrics with zero variance in last hour
            - 'out_of_range': metrics with physically impossible values
        """
        issues = []
        samples_per_hour = 60 // max(1, 5)  # assuming 5-min intervals

        # 1. Missing values
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100

        if missing_pct > 5:
            issues.append(f"HIGH missing data: {missing_pct:.1f}% of values are NaN")
        elif missing_pct > 1:
            issues.append(f"Moderate missing data: {missing_pct:.1f}% NaN")

        # 2. Frozen metrics (sensor failure)
        frozen_metrics = []
        last_hour = df.tail(samples_per_hour * 2)  # last ~2 hours
        for col in df.columns:
            if last_hour[col].nunique() <= 1:
                frozen_metrics.append(col)
                issues.append(f"FROZEN metric '{col}': same value for ≥1 hour (sensor failure?)")

        # 3. Out-of-range values
        out_of_range = []
        range_checks = {
            'cpu_utilization': (0, 100),
            'memory_usage_percent': (0, 100),
            'error_rate_percent': (0, 100),
            'cache_hit_rate': (0, 100),
            'disk_io_wait_percent': (0, 100),
        }
        for col, (low, high) in range_checks.items():
            if col in df.columns:
                violations = ((df[col] < low - 5) | (df[col] > high + 5)).sum()
                if violations > 0:
                    out_of_range.append(col)
                    issues.append(
                        f"OUT OF RANGE: '{col}' has {violations} values outside [{low}, {high}]"
                    )

        # 4. Duplicate timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            dup_count = df.index.duplicated().sum()
            if dup_count > 0:
                issues.append(f"DUPLICATE timestamps: {dup_count} duplicates found")

        # 5. Quality score (simple heuristic)
        quality_score = max(0, 100 - missing_pct * 5 - len(issues) * 5)

        return {
            'quality_score': round(quality_score, 1),
            'issues': issues,
            'missing_pct': round(missing_pct, 2),
            'frozen_metrics': frozen_metrics,
            'out_of_range': out_of_range,
            'total_samples': len(df),
            'num_features': len(df.columns)
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _assert_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "DataPreprocessor is not fitted. Call fit() with normal data first."
            )

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names in the order used during fit."""
        self._assert_fitted()
        return self.feature_names

    @property
    def n_features(self) -> int:
        """Number of features (columns) learned from fit()."""
        self._assert_fitted()
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    gen = SyntheticMetricsGenerator()
    normal_df = gen.generate_normal_behavior(duration_days=10)

    preprocessor = DataPreprocessor(window_size=60, stride=5)
    preprocessor.fit(normal_df)

    print(f"Fitted on {len(normal_df)} samples, {preprocessor.n_features} features")

    # Quality check
    report = preprocessor.validate_data_quality(normal_df)
    print(f"Data quality score: {report['quality_score']}")
    print(f"Issues: {report['issues'] or 'None'}")

    # Create windows
    windows, indices, end_ts = preprocessor.create_windows_from_df(normal_df, stride=10)
    print(f"Windows shape: {windows.shape}  (N, window_size, features)")
    print(f"First window end timestamp: {end_ts[0]}")
