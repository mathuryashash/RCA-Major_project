"""
Unified Anomaly Scorer (FR-19)

Combines scores from multiple anomaly detectors (TF-IDF, LogBERT, LSTM AE,
Temporal Transformer) into a single unified anomaly score per signal.

Formula: score = alpha * model_score + (1 - alpha) * rarity_prior
Default: alpha = 0.80, anomaly threshold = 0.5
"""

import numpy as np
from typing import Dict, List, Optional, Union


class UnifiedAnomalyScorer:
    """Combines multiple anomaly detector scores into a unified score."""

    LOG_LEVEL_WEIGHTS: Dict[str, float] = {
        "DEBUG": 0.1,
        "INFO": 0.2,
        "WARN": 0.5,
        "WARNING": 0.5,
        "ERROR": 0.8,
        "CRITICAL": 1.0,
        "FATAL": 1.0,
    }

    def __init__(
        self,
        alpha: float = 0.80,
        anomaly_threshold: float = 0.5,
        log_model_weights: Optional[Dict[str, float]] = None,
        metric_model_weights: Optional[Dict[str, float]] = None,
    ):
        self.alpha = alpha
        self.anomaly_threshold = anomaly_threshold
        self.log_model_weights = log_model_weights or {"tfidf": 0.3, "logbert": 0.7}
        self.metric_model_weights = metric_model_weights or {
            "lstm_ae": 0.6,
            "transformer": 0.4,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(values: Union[np.ndarray, List, None]) -> Optional[np.ndarray]:
        """Convert input to float64 ndarray, or return None."""
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return arr

    @staticmethod
    def _clip_scores(scores: np.ndarray) -> np.ndarray:
        return np.clip(scores, 0.0, 1.0)

    def _weighted_combination(
        self,
        scores_dict: Dict[str, Optional[np.ndarray]],
        weight_map: Dict[str, float],
    ) -> np.ndarray:
        """Compute a weighted average over available score arrays.

        If some score arrays are None they are skipped and the weights of the
        remaining arrays are re-normalised so that they still sum to 1.
        """
        available: Dict[str, np.ndarray] = {}
        for key, arr in scores_dict.items():
            if arr is not None and len(arr) > 0:
                available[key] = arr

        if not available:
            # Nothing available – return zeros (length inferred later by caller)
            return np.array([])

        total_weight = sum(weight_map.get(k, 0.0) for k in available)
        if total_weight == 0.0:
            total_weight = 1.0  # avoid division by zero

        # Determine output length from first available array
        n = len(next(iter(available.values())))
        combined = np.zeros(n, dtype=np.float64)
        for key, arr in available.items():
            w = weight_map.get(key, 0.0) / total_weight
            combined += w * arr

        return self._clip_scores(combined)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_logs(
        self,
        tfidf_scores: Union[np.ndarray, List, None] = None,
        logbert_scores: Union[np.ndarray, List, None] = None,
        log_levels: Union[List[str], np.ndarray, None] = None,
        log_frequencies: Union[np.ndarray, List, None] = None,
    ) -> Dict[str, np.ndarray]:
        """Score log signals.

        Returns
        -------
        dict with keys:
            unified_scores  – np.ndarray of floats in [0, 1]
            is_anomalous    – np.ndarray of bool
            model_scores    – np.ndarray (weighted model combination)
            rarity_priors   – np.ndarray (log-level based prior)
        """
        tfidf_arr = self._to_array(tfidf_scores)
        logbert_arr = self._to_array(logbert_scores)

        # Model score (weighted combination of available detectors)
        model_scores = self._weighted_combination(
            {"tfidf": tfidf_arr, "logbert": logbert_arr},
            self.log_model_weights,
        )

        # Determine length from model_scores or log_levels
        if model_scores.size > 0:
            n = len(model_scores)
        elif log_levels is not None:
            n = len(log_levels)
        else:
            return {
                "unified_scores": np.array([]),
                "is_anomalous": np.array([], dtype=bool),
                "model_scores": np.array([]),
                "rarity_priors": np.array([]),
            }

        if model_scores.size == 0:
            model_scores = np.zeros(n, dtype=np.float64)

        # Rarity prior from log levels
        if log_levels is not None and len(log_levels) > 0:
            rarity_priors = np.array(
                [self.LOG_LEVEL_WEIGHTS.get(str(lv).upper(), 0.2) for lv in log_levels],
                dtype=np.float64,
            )
        else:
            # Fallback: uniform moderate prior
            rarity_priors = np.full(n, 0.2, dtype=np.float64)

        # Unified score = alpha * model_score + (1 - alpha) * rarity_prior
        unified = self.alpha * model_scores + (1.0 - self.alpha) * rarity_priors
        unified = self._clip_scores(unified)

        return {
            "unified_scores": unified,
            "is_anomalous": unified > self.anomaly_threshold,
            "model_scores": model_scores,
            "rarity_priors": rarity_priors,
        }

    def score_metrics(
        self,
        lstm_ae_scores: Union[np.ndarray, List, None] = None,
        transformer_scores: Union[np.ndarray, List, None] = None,
        metric_stats: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, np.ndarray]:
        """Score metric signals.

        Parameters
        ----------
        lstm_ae_scores : per-feature anomaly scores from LSTM Autoencoder
        transformer_scores : per-feature anomaly scores from Temporal Transformer
        metric_stats : optional dict of {metric_name: (mean, std)} for z-score
                       based rarity priors

        Returns
        -------
        dict with keys: unified_scores, is_anomalous, model_scores, rarity_priors
        """
        lstm_arr = self._to_array(lstm_ae_scores)
        trans_arr = self._to_array(transformer_scores)

        model_scores = self._weighted_combination(
            {"lstm_ae": lstm_arr, "transformer": trans_arr},
            self.metric_model_weights,
        )

        if model_scores.size == 0:
            return {
                "unified_scores": np.array([]),
                "is_anomalous": np.array([], dtype=bool),
                "model_scores": np.array([]),
                "rarity_priors": np.array([]),
            }

        n = len(model_scores)

        # Rarity prior: use inverse-frequency / z-score hint if available
        if metric_stats is not None and len(metric_stats) > 0:
            # Build rarity prior from z-scores (higher z → higher rarity)
            rarity_priors = np.zeros(n, dtype=np.float64)
            for i, (_, (mean, std)) in enumerate(metric_stats.items()):
                if i >= n:
                    break
                if std > 0:
                    # Normalise z-score to [0, 1] via sigmoid-like mapping
                    z = abs(model_scores[i] - 0.5) * 2.0  # rough re-centre
                    rarity_priors[i] = min(1.0, z)
                else:
                    rarity_priors[i] = 0.5
        else:
            # Default moderate prior when no stats available
            rarity_priors = np.full(n, 0.3, dtype=np.float64)

        unified = self.alpha * model_scores + (1.0 - self.alpha) * rarity_priors
        unified = self._clip_scores(unified)

        return {
            "unified_scores": unified,
            "is_anomalous": unified > self.anomaly_threshold,
            "model_scores": model_scores,
            "rarity_priors": rarity_priors,
        }

    def score_unified(
        self,
        log_result: Optional[Dict[str, np.ndarray]] = None,
        metric_result: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Combine log and metric results into a single unified output.

        Returns all signals concatenated and ranked by descending score.
        """
        all_scores: List[float] = []
        all_anomalous: List[bool] = []
        all_sources: List[str] = []

        if log_result is not None and log_result["unified_scores"].size > 0:
            all_scores.extend(log_result["unified_scores"].tolist())
            all_anomalous.extend(log_result["is_anomalous"].tolist())
            all_sources.extend(["log"] * len(log_result["unified_scores"]))

        if metric_result is not None and metric_result["unified_scores"].size > 0:
            all_scores.extend(metric_result["unified_scores"].tolist())
            all_anomalous.extend(metric_result["is_anomalous"].tolist())
            all_sources.extend(["metric"] * len(metric_result["unified_scores"]))

        if not all_scores:
            return {
                "unified_scores": np.array([]),
                "is_anomalous": np.array([], dtype=bool),
                "sources": np.array([], dtype=str),
                "rank_indices": np.array([], dtype=int),
            }

        scores_arr = np.array(all_scores, dtype=np.float64)
        anomalous_arr = np.array(all_anomalous, dtype=bool)
        sources_arr = np.array(all_sources, dtype=str)

        # Rank by descending score
        rank_indices = np.argsort(-scores_arr)

        return {
            "unified_scores": scores_arr[rank_indices],
            "is_anomalous": anomalous_arr[rank_indices],
            "sources": sources_arr[rank_indices],
            "rank_indices": rank_indices,
        }
