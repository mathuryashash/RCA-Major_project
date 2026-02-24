"""
RCA Pipeline — End-to-End Orchestrator

Ties together all 4 stages:
  Stage 1: Data ingestion & preprocessing
  Stage 2: Anomaly detection (LSTM Autoencoder)
  Stage 3: Causal inference (Granger causality)
  Stage 4: Root cause ranking & report generation

Usage:
    pipeline = RCAPipeline()
    pipeline.train(normal_df)
    results = pipeline.analyze(failure_df)
    pipeline.print_report(results)
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
import json
import time

from src.preprocessing.data_normalizer import DataPreprocessor
from src.models.lstm_autoencoder import LSTMAutoencoder, AnomalyDetectionTrainer
from src.causal_inference.granger_causality import CausalInferenceEngine
from src.root_cause_ranking.scorer import RootCauseRanker, RootCauseScore

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RCAPipelineConfig:
    """Configuration for the RCA pipeline."""

    # Preprocessing
    WINDOW_SIZE: int = 60          # LSTM window: 60 timesteps = 5h at 5-min intervals
    TRAIN_STRIDE: int = 5          # Step between training windows (reduces dataset size)
    INFERENCE_STRIDE: int = 1      # Step between inference windows (finer resolution)

    # LSTM Autoencoder
    HIDDEN_SIZE: int = 64
    LATENT_SIZE: int = 32
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    LR: float = 1e-3
    EPOCHS: int = 50
    BATCH_SIZE: int = 32
    PATIENCE: int = 10
    THRESHOLD_PERCENTILE: float = 99.0  # 1% false positive rate
    ANOMALY_SCORE_THRESHOLD: float = 1.0  # Normalized score above which = anomaly

    # Causal Inference
    MAX_LAG: int = 10
    SIGNIFICANCE_LEVEL: float = 0.05
    MIN_ANOMALY_SCORE: float = 0.5   # Only include metrics above this score

    # Root Cause Ranking
    TOP_N: int = 5


class RCAPipeline:
    """
    End-to-end RCA pipeline orchestrator.

    The pipeline has two phases:
    1. train(normal_df) — learns normal behavior from healthy data
    2. analyze(failure_df) — detects anomalies, infers causes, ranks root causes

    Both phases can be run on synthetic data (for development/evaluation)
    or real production metrics.
    """

    def __init__(self, config: Optional[RCAPipelineConfig] = None):
        self.config = config or RCAPipelineConfig()
        self.preprocessor = DataPreprocessor(
            window_size=self.config.WINDOW_SIZE,
            stride=self.config.TRAIN_STRIDE
        )
        self.model: Optional[LSTMAutoencoder] = None
        self.trainer: Optional[AnomalyDetectionTrainer] = None
        self.causal_engine = CausalInferenceEngine(
            max_lag=self.config.MAX_LAG,
            significance_level=self.config.SIGNIFICANCE_LEVEL,
            min_anomaly_score=self.config.MIN_ANOMALY_SCORE
        )
        self.ranker = RootCauseRanker()
        self.is_trained = False

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------

    def train(
        self,
        normal_df: pd.DataFrame,
        val_split: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the LSTM autoencoder on normal operational data.

        Args:
            normal_df: DataFrame of healthy metrics from SyntheticMetricsGenerator
                       or real monitoring data
            val_split: Fraction of data to use for validation (and threshold calibration)
            save_path: If provided, save trained model here

        Returns:
            Training history dict
        """
        print("=" * 60)
        print("PHASE 1: TRAINING ON NORMAL DATA")
        print("=" * 60)

        # 1. Validate data quality
        quality_report = self.preprocessor.validate_data_quality(normal_df)
        print(f"Data quality score: {quality_report['quality_score']}/100")
        if quality_report['issues']:
            for issue in quality_report['issues']:
                print(f"  ⚠️  {issue}")

        # 2. Fit preprocessor and create windows
        print(f"\nPreprocessing: fitting scaler on {len(normal_df)} samples...")
        self.preprocessor.fit(normal_df)
        windows, _, _ = self.preprocessor.create_windows_from_df(
            normal_df, stride=self.config.TRAIN_STRIDE
        )
        print(f"Created {len(windows)} training windows (size={self.config.WINDOW_SIZE})")

        # 3. Train/val split
        split_idx = int(len(windows) * (1 - val_split))
        train_w = windows[:split_idx]
        val_w = windows[split_idx:]
        print(f"Train windows: {len(train_w)}  |  Val windows: {len(val_w)}")

        # 4. Build and train model
        n_features = self.preprocessor.n_features
        self.model = LSTMAutoencoder(
            input_size=n_features,
            sequence_length=self.config.WINDOW_SIZE,
            hidden_size=self.config.HIDDEN_SIZE,
            latent_size=self.config.LATENT_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        )
        self.trainer = AnomalyDetectionTrainer(self.model, lr=self.config.LR)

        print(f"\nTraining LSTM Autoencoder ({n_features} features, "
              f"hidden={self.config.HIDDEN_SIZE}, latent={self.config.LATENT_SIZE})...")
        history = self.trainer.fit(
            train_w, val_w,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            patience=self.config.PATIENCE,
            save_path=save_path
        )

        # 5. Calibrate thresholds
        print("\nCalibrating anomaly detection thresholds...")
        self.trainer.calibrate_thresholds(val_w, percentile=self.config.THRESHOLD_PERCENTILE)

        self.is_trained = True
        print("\n✅ Training complete.\n")
        return history

    # ------------------------------------------------------------------
    # Phase 2: Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        failure_df: pd.DataFrame,
        event_correlations: Optional[List[Dict]] = None,
        known_dependencies: Optional[List[Tuple[str, str, float]]] = None
    ) -> Dict:
        """
        Run the full RCA pipeline on failure/suspect data.

        Args:
            failure_df: Metric DataFrame covering the failure window
            event_correlations: Deployment/config events for correlation
            known_dependencies: Topology-based edges to inject into causal graph

        Returns:
            results dict with keys:
            'anomaly_scores', 'anomaly_first_detected', 'causal_graph',
            'ranked_causes', 'report', 'timing'
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before calling analyze(). "
                               "Call train(normal_df) first.")

        print("=" * 60)
        print("PHASE 2: ANALYZING FAILURE DATA")
        print("=" * 60)

        results = {}
        timing = {}

        # --- Stage 2: Anomaly Detection ---
        t0 = time.time()
        print("\n[Stage 2] Running anomaly detection...")
        windows, indices, timestamps = self.preprocessor.create_windows_from_df(
            failure_df, stride=self.config.INFERENCE_STRIDE
        )
        detection = self.trainer.detect_anomalies(
            windows, score_threshold=self.config.ANOMALY_SCORE_THRESHOLD
        )
        timing['anomaly_detection_sec'] = round(time.time() - t0, 2)

        # Map feature indices back to metric names
        feature_names = self.preprocessor.get_feature_names()
        anomalous_feature_mask = detection['anomalous_features']  # (N, F)

        # Aggregate: max score per metric across all windows
        anomaly_scores: Dict[str, float] = {}
        for f_idx, feat_name in enumerate(feature_names):
            max_score = float(detection['scores']['feature_scores'][:, f_idx].max())
            anomaly_scores[feat_name] = round(max_score, 4)

        # Find first detection time per metric
        anomaly_first_detected: Dict[str, pd.Timestamp] = {}
        for f_idx, feat_name in enumerate(feature_names):
            anomaly_windows = np.where(anomalous_feature_mask[:, f_idx])[0]
            if len(anomaly_windows) > 0:
                first_win = anomaly_windows[0]
                if first_win < len(timestamps):
                    anomaly_first_detected[feat_name] = timestamps[first_win]

        anomalous_count = sum(1 for s in anomaly_scores.values()
                              if s >= self.config.ANOMALY_SCORE_THRESHOLD)
        print(f"  Detected {anomalous_count} anomalous metrics out of {len(feature_names)}")
        for m, s in sorted(anomaly_scores.items(), key=lambda x: -x[1]):
            flag = " ⚠️" if s >= self.config.ANOMALY_SCORE_THRESHOLD else ""
            print(f"    {m:35s}: {s:.3f}{flag}")

        results['anomaly_scores'] = anomaly_scores
        results['anomaly_first_detected'] = {
            k: str(v) for k, v in anomaly_first_detected.items()
        }

        # --- Stage 3: Causal Inference ---
        t0 = time.time()
        print("\n[Stage 3] Running causal inference...")
        causal_graph = self.causal_engine.run(
            metrics_df=failure_df,
            anomaly_scores=anomaly_scores,
            anomaly_first_detected=anomaly_first_detected,
            known_dependencies=known_dependencies
        )
        timing['causal_inference_sec'] = round(time.time() - t0, 2)

        causal_summary = self.causal_engine.format_results(causal_graph)
        results['causal_graph_summary'] = causal_summary

        print(f"  Causal graph: {causal_summary['total_nodes']} nodes, "
              f"{causal_summary['total_edges']} edges")
        print(f"  Potential root causes (graph sources): "
              f"{causal_summary['potential_root_causes']}")

        # --- Stage 4: Root Cause Ranking ---
        t0 = time.time()
        print("\n[Stage 4] Ranking root causes...")
        ranked_causes = self.ranker.rank(
            causal_graph=causal_graph,
            anomaly_scores=anomaly_scores,
            anomaly_first_detected=anomaly_first_detected,
            event_correlations=event_correlations
        )
        timing['ranking_sec'] = round(time.time() - t0, 2)

        results['ranked_causes'] = ranked_causes
        results['timing'] = timing
        results['causal_graph'] = causal_graph

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict, top_n: int = 3):
        """Print a formatted RCA report to console."""
        print()
        print(self.ranker.generate_summary_report(results['ranked_causes'], top_n=top_n))

        print("\n=== DETAILED EVIDENCE ===")
        for rc in results['ranked_causes'][:top_n]:
            print()
            print(rc.justification)

        print("\n=== TIMING ===")
        for stage, sec in results['timing'].items():
            print(f"  {stage}: {sec}s")

    def generate_json_report(self, results: Dict) -> Dict:
        """Serialize results to a JSON-compatible dict."""
        ranked = results.get('ranked_causes', [])
        return {
            'anomaly_scores': results.get('anomaly_scores', {}),
            'anomaly_first_detected': results.get('anomaly_first_detected', {}),
            'causal_graph': results.get('causal_graph_summary', {}),
            'root_causes': [
                {
                    'rank': rc.rank,
                    'metric': rc.metric,
                    'confidence': rc.confidence,
                    'confidence_pct': rc.confidence_pct,
                    'final_score': rc.final_score,
                    'factor_scores': rc.factor_scores,
                    'downstream_effects': rc.downstream_effects,
                    'justification': rc.justification
                }
                for rc in ranked
            ],
            'timing': results.get('timing', {})
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    gen = SyntheticMetricsGenerator(seed=42)

    # Generate normal data for training
    print("Generating synthetic normal data (30 days)...")
    normal_df = gen.generate_normal_behavior(duration_days=30)

    # Generate failure data
    print("Injecting database slow query failure...")
    failure_start = (28 * 24 * 60) // 5  # Day 28
    failure_df, meta = gen.inject_failure_scenario(
        normal_df, 'database_slow_query', failure_start, duration_samples=150, severity=1.0
    )

    # Use just the training portion (normal) for training
    train_df = normal_df.iloc[:failure_start]
    # Use the failure window for analysis
    analysis_df = failure_df.iloc[max(0, failure_start - 50):]

    # Create a minimal config for fast demo
    config = RCAPipelineConfig()
    config.EPOCHS = 5          # Very low for demo
    config.PATIENCE = 3
    config.WINDOW_SIZE = 30    # Smaller window for faster training
    config.TRAIN_STRIDE = 10
    config.MAX_LAG = 5

    pipeline = RCAPipeline(config=config)

    # Train
    pipeline.train(train_df)

    # Analyze
    results = pipeline.analyze(analysis_df)

    # Report
    pipeline.print_report(results)

    # JSON output
    json_report = pipeline.generate_json_report(results)
    print("\n=== JSON REPORT (truncated) ===")
    print(json.dumps({k: v for k, v in json_report.items() if k != 'justification'},
                     indent=2, default=str)[:1500])

    print(f"\n✅ Ground truth root cause: {meta['root_cause']}")
    print(f"✅ Ground truth causal chain: {meta['causal_chain']}")
