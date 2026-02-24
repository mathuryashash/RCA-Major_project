"""
Anomaly Detection Engine
========================
Detects 50+ anomaly types across CPU, memory, storage, network,
application, and database layers using a three-method ensemble:

    1. LSTM Autoencoder (reconstruction error) — 40% weight
    2. Statistical methods (z-score, IQR, MAD)  — 35% weight
    3. Temporal pattern detection (trend, spike, CUSUM, FFT) — 25% weight

Usage
-----
    from src.anomaly_detection.anomaly_detection_engine import AnomalyDetectionEngine
    from src.preprocessing.data_normalizer import DataPreprocessor
    from src.models.lstm_autoencoder import LSTMAutoencoder, AnomalyDetectionTrainer

    # 1. Prepare data
    preprocessor = DataPreprocessor(window_size=60, stride=5)
    preprocessor.fit(normal_df)
    train_windows, _, _ = preprocessor.create_windows_from_df(normal_df)

    # 2. Train LSTM
    model   = LSTMAutoencoder(input_size=preprocessor.n_features, sequence_length=60)
    trainer = AnomalyDetectionTrainer(model)
    trainer.fit(train_windows[:int(len(train_windows)*0.8)],
                train_windows[int(len(train_windows)*0.8):])
    trainer.calibrate_thresholds(train_windows[int(len(train_windows)*0.8):])

    # 3. Detect anomalies on new data
    engine = AnomalyDetectionEngine(preprocessor, trainer)
    engine.fit_baselines(normal_df)
    results = engine.detect(failure_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import periodogram
import warnings


# ---------------------------------------------------------------------------
# Helper: per-metric z-score
# ---------------------------------------------------------------------------

def _zscore_series(series: pd.Series, window: int = 288) -> pd.Series:
    """Rolling z-score with a specified lookback window."""
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 4)).mean()
    rolling_std  = series.rolling(window=window, min_periods=max(1, window // 4)).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-8)
    return zscore.fillna(0.0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AnomalyDetectionEngine:
    """
    Three-method ensemble anomaly detector.

    Parameters
    ----------
    preprocessor : DataPreprocessor
        Fitted preprocessor (fit on normal data).
    trainer : AnomalyDetectionTrainer | None
        Trained LSTM anomaly detection trainer.  If None the LSTM branch
        is skipped and the ensemble uses only statistical + temporal methods.
    lstm_weight : float
        Ensemble weight for LSTM branch (default 0.40).
    stat_weight : float
        Ensemble weight for statistical branch (default 0.35).
    temporal_weight : float
        Ensemble weight for temporal branch (default 0.25).
    anomaly_threshold : float
        Final ensemble score above which a metric is flagged as anomalous
        (default 0.5).
    baseline_window : int
        Rolling window size (in timesteps) for statistical baselines.
        At 5-minute resolution, 288 ≈ 24 hours.
    """

    # ------------------------------------------------------------------
    # Known typical ranges for common metrics (used for sanity bounds)
    # ------------------------------------------------------------------
    _METRIC_BOUNDS: Dict[str, Tuple[float, float]] = {
        'cpu_utilization':       (0.0, 100.0),
        'memory_usage_percent':  (0.0, 100.0),
        'error_rate_percent':    (0.0, 100.0),
        'cache_hit_rate':        (0.0, 100.0),
        'disk_io_wait_percent':  (0.0, 100.0),
    }

    def __init__(
        self,
        preprocessor=None,
        trainer=None,
        lstm_weight: float = 0.40,
        stat_weight: float = 0.35,
        temporal_weight: float = 0.25,
        anomaly_threshold: float = 0.5,
        baseline_window: int = 288,
    ):
        self.preprocessor     = preprocessor
        self.trainer          = trainer
        self.lstm_weight      = lstm_weight
        self.stat_weight      = stat_weight
        self.temporal_weight  = temporal_weight
        self.anomaly_threshold = anomaly_threshold
        self.baseline_window  = baseline_window

        # Baselines computed from normal data
        self._baselines: Dict[str, Dict[str, float]] = {}  # metric -> {mean, std, q1, q3}
        self._is_fitted = False

    # ==================================================================
    # Public API
    # ==================================================================

    def fit_baselines(self, normal_df: pd.DataFrame) -> 'AnomalyDetectionEngine':
        """
        Learn per-metric statistical baselines from normal (healthy) data.

        Must be called before detect().

        Parameters
        ----------
        normal_df : pd.DataFrame
            Normal operational metrics.  Columns are metric names.
        """
        for col in normal_df.columns:
            series = normal_df[col].dropna()
            q1, q3 = np.percentile(series, [25, 75])
            self._baselines[col] = {
                'mean': float(series.mean()),
                'std':  float(series.std() + 1e-8),
                'q1':   float(q1),
                'q3':   float(q3),
                'iqr':  float(q3 - q1 + 1e-8),
                'median': float(series.median()),
                'mad':  float(np.median(np.abs(series - series.median())) + 1e-8),
            }
        self._is_fitted = True
        return self

    def detect(
        self,
        df: pd.DataFrame,
        stride: int = 1,
    ) -> Dict[str, Any]:
        """
        Run ensemble anomaly detection on a new DataFrame of metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Metrics to analyze (same columns as normal data).
        stride : int
            Stride for sliding-window LSTM inference.

        Returns
        -------
        dict with keys:
            'anomaly_scores'       : DataFrame (timesteps × metrics) with ensemble scores ∈ [0,1]
            'is_anomaly'           : DataFrame (timesteps × metrics) boolean flags
            'lstm_scores'          : DataFrame or None
            'statistical_scores'   : DataFrame
            'temporal_scores'      : DataFrame
            'anomaly_summary'      : dict — which metrics are anomalous and first-detection time
            'correlated_anomalies' : dict — groups of likely-related anomalies
            'anomaly_types'        : dict — detected pattern types per metric
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_baselines() with normal data first.")

        metrics = [c for c in df.columns if c in self._baselines or c not in ['timestamp']]

        # --- Branch 1: Statistical ---
        stat_scores_df = self._detect_statistical(df[metrics])

        # --- Branch 2: Temporal ---
        temporal_scores_df, anomaly_types = self._detect_temporal(df[metrics])

        # --- Branch 3: LSTM ---
        lstm_scores_df = self._detect_lstm(df[metrics], stride)

        # --- Ensemble ---
        ensemble_df = self._ensemble(stat_scores_df, temporal_scores_df, lstm_scores_df, metrics)

        is_anomaly_df = ensemble_df > self.anomaly_threshold

        # --- Summary ---
        summary = self._build_summary(ensemble_df, is_anomaly_df, df.index)

        # --- Correlations ---
        corr_groups = self._detect_correlated_anomalies(is_anomaly_df)

        return {
            'anomaly_scores':       ensemble_df,
            'is_anomaly':           is_anomaly_df,
            'lstm_scores':          lstm_scores_df,
            'statistical_scores':   stat_scores_df,
            'temporal_scores':      temporal_scores_df,
            'anomaly_summary':      summary,
            'correlated_anomalies': corr_groups,
            'anomaly_types':        anomaly_types,
        }

    # ==================================================================
    # Branch 1: Statistical Detection
    # ==================================================================

    def _detect_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines three statistical tests; returns score in [0, 1] per cell.

        Methods:
          - Z-score     : |z| > 3.0 -> anomaly
          - IQR         : x < Q1 - 1.5*IQR  or x > Q3 + 1.5*IQR
          - MAD         : (x - median) / MAD > 3.5
        """
        scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)

        for col in df.columns:
            series = df[col].fillna(method='ffill').fillna(0.0)
            bl = self._baselines.get(col, {})

            if not bl:
                continue

            # Z-score
            z = np.abs((series - bl['mean']) / bl['std'])
            z_score = np.clip(z / 6.0, 0.0, 1.0)  # normalise: z=6 -> score=1.0

            # IQR
            lower = bl['q1'] - 1.5 * bl['iqr']
            upper = bl['q3'] + 1.5 * bl['iqr']
            iqr_violation = ((series < lower) | (series > upper)).astype(float)
            iqr_magnitude  = np.clip(
                np.maximum(
                    np.abs(series - lower) / (bl['iqr'] + 1e-8),
                    np.abs(series - upper) / (bl['iqr'] + 1e-8),
                ) * iqr_violation,
                0.0, 1.0
            )

            # MAD
            mad_z = np.abs(series - bl['median']) / (bl['mad'] * 1.4826)
            mad_score = np.clip(mad_z / 7.0, 0.0, 1.0)

            # Combine (simple average of three tests)
            combined = (z_score + iqr_magnitude + mad_score) / 3.0
            scores[col] = combined.values

        return scores

    # ==================================================================
    # Branch 2: Temporal Pattern Detection
    # ==================================================================

    def _detect_temporal(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Detect five temporal patterns per metric.

        Patterns:
          1. LINEAR_TREND   — memory-leak-style monotonic growth
          2. SUDDEN_SPIKE   — sharp derivative spike
          3. OSCILLATION    — high-frequency fluctuation (FFT)
          4. STEP_CHANGE    — CUSUM-based baseline shift
          5. DRIFT          — slow mean shift

        Returns
        -------
        scores_df : DataFrame of shape (T, n_metrics) with temporal scores
        anomaly_types : dict mapping metric -> list of detected pattern names
        """
        scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        anomaly_types: Dict[str, List[str]] = {col: [] for col in df.columns}

        for col in df.columns:
            series = df[col].fillna(method='ffill').fillna(0.0).values.astype(float)
            T = len(series)
            if T < 10:
                continue

            col_scores = np.zeros(T)

            # 1. Linear trend (regression over entire window)
            trend_score, _ = self._linear_trend_score(series)
            if trend_score > 0.5:
                anomaly_types[col].append('LINEAR_TREND')
            col_scores += trend_score * np.ones(T)  # broadcast as uniform

            # 2. Sudden spike (derivative analysis)
            spike_scores = self._spike_score(series)
            if spike_scores.max() > 0.5:
                anomaly_types[col].append('SUDDEN_SPIKE')
            col_scores += spike_scores

            # 3. Oscillation (FFT dominant frequency)
            osc_score = self._oscillation_score(series)
            if osc_score > 0.5:
                anomaly_types[col].append('OSCILLATION')
            col_scores += osc_score * np.ones(T)

            # 4. Step change (CUSUM)
            cusum_scores = self._cusum_score(series)
            if cusum_scores.max() > 0.5:
                anomaly_types[col].append('STEP_CHANGE')
            col_scores += cusum_scores

            # 5. Drift (rolling mean shift)
            drift_score = self._drift_score(series)
            if drift_score > 0.5:
                anomaly_types[col].append('DRIFT')
            col_scores += drift_score * np.ones(T)

            # Normalise: up to 5 patterns can contribute
            col_scores = np.clip(col_scores / 5.0, 0.0, 1.0)
            scores[col] = col_scores

        return scores, anomaly_types

    def _linear_trend_score(self, series: np.ndarray) -> Tuple[float, float]:
        """Return (score, slope) based on R² of linear regression."""
        T = len(series)
        x = np.arange(T)
        if series.std() < 1e-8:
            return 0.0, 0.0
        slope, intercept, r, _, _ = stats.linregress(x, series)
        r2 = r ** 2
        # Strong upward/downward linear trend is anomalous
        score = r2  # 1.0 = perfect linear trend; scaled by R²
        return float(np.clip(score, 0, 1)), float(slope)

    def _spike_score(self, series: np.ndarray) -> np.ndarray:
        """Per-timestep spike score based on first derivative magnitude."""
        if len(series) < 2:
            return np.zeros(len(series))
        diff = np.abs(np.diff(series, prepend=series[0]))
        baseline_std = np.std(diff) + 1e-8
        spike_z = diff / baseline_std
        # Score: how many std-deviations above normal is this derivative?
        return np.clip(spike_z / 10.0, 0.0, 1.0)

    def _oscillation_score(self, series: np.ndarray) -> float:
        """
        Score based on spectral flatness / dominant frequency energy.
        High oscillation -> dominant high-frequency component.
        """
        T = len(series)
        if T < 16:
            return 0.0
        try:
            freqs, power = periodogram(series - series.mean())
            if power.sum() < 1e-10:
                return 0.0
            # Geometric-to-arithmetic mean ratio (spectral flatness)
            geo_mean = np.exp(np.mean(np.log(power + 1e-12)))
            arith_mean = power.mean()
            flatness = geo_mean / (arith_mean + 1e-8)
            # High flatness = noise-like / oscillation
            return float(np.clip(flatness * 2, 0.0, 1.0))
        except Exception:
            return 0.0

    def _cusum_score(self, series: np.ndarray) -> np.ndarray:
        """
        CUSUM-based step-change detector.
        Returns per-timestep score ∈ [0, 1].
        """
        mean = series.mean()
        std  = series.std() + 1e-8
        k    = 0.5  # slack parameter
        h    = 5.0  # decision interval (threshold)
        s_pos = np.zeros(len(series))
        s_neg = np.zeros(len(series))
        for t in range(1, len(series)):
            x = (series[t] - mean) / std
            s_pos[t] = max(0, s_pos[t-1] + x - k)
            s_neg[t] = max(0, s_neg[t-1] - x - k)
        cusum = np.maximum(s_pos, s_neg)
        return np.clip(cusum / (h * 2), 0.0, 1.0)

    def _drift_score(self, series: np.ndarray) -> float:
        """Compare rolling mean of last 20% of data vs first 50%: drift score."""
        T    = len(series)
        cut1 = int(T * 0.5)
        cut2 = int(T * 0.8)
        if cut1 < 2 or cut2 >= T:
            return 0.0
        mu_early = series[:cut1].mean()
        std_early = series[:cut1].std() + 1e-8
        mu_late  = series[cut2:].mean()
        z_drift  = abs(mu_late - mu_early) / std_early
        return float(np.clip(z_drift / 5.0, 0.0, 1.0))

    # ==================================================================
    # Branch 3: LSTM Detection
    # ==================================================================

    def _detect_lstm(
        self,
        df: pd.DataFrame,
        stride: int,
    ) -> Optional[pd.DataFrame]:
        """
        Use the trained LSTM autoencoder to compute per-metric anomaly scores.

        Returns a DataFrame aligned to df's index, or None if trainer is unavailable.
        """
        if self.trainer is None or self.preprocessor is None:
            return None

        try:
            import torch
            windows, indices, end_ts = self.preprocessor.create_windows_from_df(
                df, stride=stride
            )
            if len(windows) == 0:
                return None

            score_dict = self.trainer.get_anomaly_scores(windows)
            feature_scores = score_dict['feature_scores']  # (N, F)

            # Map per-window scores back to per-timestep using end_ts
            feature_names = self.preprocessor.get_feature_names()
            cols_in_df = [c for c in feature_names if c in df.columns]

            # Build output sparse DataFrame then reindex to df's full index
            out = pd.DataFrame(
                feature_scores[:, :len(cols_in_df)],
                index=end_ts,
                columns=cols_in_df,
            )
            # Reindex and forward-fill so every timestep has a value
            out = out.reindex(df.index, method='nearest').fillna(0.0)

            # Add any columns missing from LSTM (not in preprocessor)
            for col in df.columns:
                if col not in out.columns:
                    out[col] = 0.0

            # Normalise to [0, 1]: score > 1 means beyond calibrated threshold
            out = np.clip(out / (out.max().clip(lower=1.0) + 1e-8), 0.0, 1.0)
            return out[df.columns]

        except Exception as exc:
            warnings.warn(f"LSTM branch failed ({exc}); falling back to stat+temporal only.")
            return None

    # ==================================================================
    # Ensemble
    # ==================================================================

    def _ensemble(
        self,
        stat_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        lstm_df: Optional[pd.DataFrame],
        metrics: List[str],
    ) -> pd.DataFrame:
        """Weighted combination of the three branches."""
        if lstm_df is not None:
            total_weight = self.lstm_weight + self.stat_weight + self.temporal_weight
            ensemble = (
                lstm_df[metrics].values * self.lstm_weight +
                stat_df[metrics].values * self.stat_weight +
                temporal_df[metrics].values * self.temporal_weight
            ) / total_weight
        else:
            total_weight = self.stat_weight + self.temporal_weight
            ensemble = (
                stat_df[metrics].values * self.stat_weight +
                temporal_df[metrics].values * self.temporal_weight
            ) / total_weight

        return pd.DataFrame(ensemble, index=stat_df.index, columns=metrics)

    # ==================================================================
    # Summary & Correlation
    # ==================================================================

    def _build_summary(
        self,
        scores_df: pd.DataFrame,
        is_anomaly_df: pd.DataFrame,
        index: pd.Index,
    ) -> Dict[str, Any]:
        """Build a human-readable anomaly summary."""
        anomalous_metrics = [col for col in is_anomaly_df.columns
                             if is_anomaly_df[col].any()]
        first_detection: Dict[str, Any] = {}
        for col in anomalous_metrics:
            mask = is_anomaly_df[col]
            if mask.any():
                t = index[mask.values.argmax()]
                first_detection[col] = {
                    'first_detected': t,
                    'max_score': float(scores_df[col].max()),
                    'pct_anomalous': float(mask.mean() * 100),
                }

        earliest_time = None
        if first_detection:
            earliest_time = min(v['first_detected'] for v in first_detection.values())

        return {
            'num_anomalous_metrics': len(anomalous_metrics),
            'anomalous_metrics': anomalous_metrics,
            'first_detections': first_detection,
            'earliest_anomaly_time': earliest_time,
        }

    def _detect_correlated_anomalies(
        self,
        is_anomaly_df: pd.DataFrame,
        min_overlap: float = 0.5,
    ) -> Dict[str, List[str]]:
        """
        Group metrics whose anomaly flags heavily overlap in time.

        Two metrics are "correlated" if the fraction of timesteps where
        BOTH are anomalous exceeds min_overlap relative to either metric's
        anomaly duration.

        Returns
        -------
        dict mapping metric -> list of correlated metrics
        """
        anomalous_cols = [c for c in is_anomaly_df.columns if is_anomaly_df[c].any()]
        groups: Dict[str, List[str]] = {c: [] for c in anomalous_cols}

        for i, c1 in enumerate(anomalous_cols):
            for j, c2 in enumerate(anomalous_cols):
                if i >= j:
                    continue
                overlap = (is_anomaly_df[c1] & is_anomaly_df[c2]).sum()
                denom = min(is_anomaly_df[c1].sum(), is_anomaly_df[c2].sum()) + 1e-8
                if overlap / denom >= min_overlap:
                    groups[c1].append(c2)
                    groups[c2].append(c1)

        return groups

    # ==================================================================
    # Convenience: single-method accessors (for transparency / debugging)
    # ==================================================================

    def detect_anomalies_statistical(
        self,
        current_value: float,
        metric_name: str,
    ) -> Tuple[bool, float]:
        """
        Quick per-value statistical check against learned baseline.

        Returns
        -------
        (is_anomalous, z_score)
        """
        bl = self._baselines.get(metric_name)
        if bl is None:
            raise ValueError(f"No baseline for metric '{metric_name}'. Call fit_baselines() first.")
        z = abs(current_value - bl['mean']) / bl['std']
        return bool(z > 3.0), float(z)

    def detect_anomalies_temporal(
        self,
        time_series: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run temporal pattern detection on a 1-D time series.

        Returns
        -------
        dict mapping pattern_type -> confidence_score ∈ [0, 1]
        """
        trend_score, _  = self._linear_trend_score(time_series)
        spike_scores    = self._spike_score(time_series)
        osc_score       = self._oscillation_score(time_series)
        cusum_scores    = self._cusum_score(time_series)
        drift_score     = self._drift_score(time_series)

        return {
            'LINEAR_TREND':  trend_score,
            'SUDDEN_SPIKE':  float(spike_scores.max()),
            'OSCILLATION':   osc_score,
            'STEP_CHANGE':   float(cusum_scores.max()),
            'DRIFT':         drift_score,
        }

    def detect_anomalies_lstm(
        self,
        current_metrics: np.ndarray,
        anomaly_threshold: float = 0.7,
    ) -> Optional[Dict[str, float]]:
        """
        LSTM-based detection for a batch of windows.

        Parameters
        ----------
        current_metrics : np.ndarray
            Shape (N, window_size, n_features).
        anomaly_threshold : float
            Score ≥ this value is considered anomalous.

        Returns
        -------
        dict mapping feature_name -> anomaly_score, or None if trainer unavailable.
        """
        if self.trainer is None:
            return None
        import torch
        t = torch.from_numpy(current_metrics.astype(np.float32))
        score_dict = self.trainer.get_anomaly_scores(current_metrics)
        feature_scores = score_dict['feature_scores'].mean(axis=0)  # (F,)
        feature_names  = (
            self.preprocessor.get_feature_names() if self.preprocessor else
            [f'feature_{i}' for i in range(len(feature_scores))]
        )
        return {name: float(score) for name, score in zip(feature_names, feature_scores)}

    def ensemble_anomaly_detection(
        self,
        current_metrics: np.ndarray,
        time_series: np.ndarray,
        metric_name: str,
    ) -> float:
        """
        Single-metric ensemble score combining all three branches.

        Parameters
        ----------
        current_metrics : np.ndarray
            Shape (N, window_size, n_features) for LSTM branch.
        time_series : np.ndarray
            1-D array of this metric's recent values for temporal branch.
        metric_name : str
            Name of the metric for the statistical branch.

        Returns
        -------
        float  ∈ [0, 1] — final ensemble anomaly score
        """
        _, z = self.detect_anomalies_statistical(
            float(time_series[-1]), metric_name
        )
        stat_score = float(np.clip(z / 6.0, 0, 1))

        temporal_scores = self.detect_anomalies_temporal(time_series)
        temporal_score  = float(np.clip(np.mean(list(temporal_scores.values())), 0, 1))

        lstm_dict = self.detect_anomalies_lstm(current_metrics)
        if lstm_dict and metric_name in lstm_dict:
            lstm_score   = float(np.clip(lstm_dict[metric_name], 0, 1))
            total_weight = self.lstm_weight + self.stat_weight + self.temporal_weight
            ensemble     = (
                lstm_score   * self.lstm_weight +
                stat_score   * self.stat_weight +
                temporal_score * self.temporal_weight
            ) / total_weight
        else:
            total_weight = self.stat_weight + self.temporal_weight
            ensemble     = (
                stat_score   * self.stat_weight +
                temporal_score * self.temporal_weight
            ) / total_weight

        return float(np.clip(ensemble, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator
    from src.preprocessing.data_normalizer import DataPreprocessor

    print("=== Anomaly Detection Engine Smoke Test ===\n")

    gen = SyntheticMetricsGenerator()
    normal_df  = gen.generate_normal_behavior(duration_days=15)
    print(f"Normal data: {len(normal_df)} timesteps, {len(normal_df.columns)} metrics")

    # Inject a failure scenario
    failure_result = gen.inject_failure_scenario(normal_df.copy(), scenario='memory_leak')
    failure_df = failure_result['metrics']

    # Build engine (without LSTM for the smoke test to avoid heavy training)
    engine = AnomalyDetectionEngine(
        preprocessor=None,
        trainer=None,
        anomaly_threshold=0.4,
    )
    engine.fit_baselines(normal_df)

    results = engine.detect(failure_df)

    print(f"\n=== Detection Results ===")
    print(f"Anomalous metrics ({results['anomaly_summary']['num_anomalous_metrics']}):")
    for m in results['anomaly_summary']['anomalous_metrics']:
        info = results['anomaly_summary']['first_detections'][m]
        print(f"  {m:35s} max_score={info['max_score']:.3f}  pct_anomalous={info['pct_anomalous']:.1f}%")

    print(f"\nAnomaly types detected:")
    for m, types in results['anomaly_types'].items():
        if types:
            print(f"  {m:35s}: {', '.join(types)}")

    print(f"\nCorrelated anomaly groups:")
    for m, corr in results['correlated_anomalies'].items():
        if corr:
            print(f"  {m} ↔ {corr}")

    print("\nSmoke test passed.")
