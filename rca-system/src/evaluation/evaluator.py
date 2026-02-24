"""
RCA Evaluation Framework
========================
Evaluates pipeline performance against labeled synthetic scenarios.

Metrics computed (per PRD Section 4.1):
  - Top-1 accuracy    : Correct root cause is ranked #1
  - Top-3 accuracy    : Correct root cause appears in top-3
  - Mean Reciprocal Rank (MRR)
  - Precision / Recall / F1 on anomaly detection
  - Per-failure-type breakdown

Usage
-----
    from src.evaluation.evaluator import RCAEvaluator
    from src.rca_pipeline import RCAPipeline, RCAPipelineConfig
    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator

    gen = SyntheticMetricsGenerator(seed=42)
    scenarios = gen.generate_evaluation_dataset(num_scenarios=12, normal_days=15)

    config = RCAPipelineConfig()
    config.EPOCHS = 10

    evaluator = RCAEvaluator(config=config)
    report = evaluator.evaluate(scenarios)
    evaluator.print_report(report)
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Evaluation result for a single scenario."""

    scenario_id: int
    failure_type: str
    ground_truth_root_cause: str          # canonical name used in metadata
    expected_anomalous_metrics: List[str]

    # Pipeline outputs
    ranked_causes: List[str] = field(default_factory=list)   # metric names, rank 1 first
    detected_anomalous_metrics: List[str] = field(default_factory=list)

    # Derived accuracy
    top1_correct: bool = False
    top3_correct: bool = False
    reciprocal_rank: float = 0.0          # 0 if not found in top-5

    # Anomaly detection quality
    anomaly_precision: float = 0.0
    anomaly_recall: float = 0.0
    anomaly_f1: float = 0.0

    # Pipeline timing
    train_secs: float = 0.0
    analyze_secs: float = 0.0

    # Error info (if pipeline failed)
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all scenarios."""

    num_scenarios: int = 0
    num_successful: int = 0
    num_failed: int = 0

    # Core accuracy metrics
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    mean_reciprocal_rank: float = 0.0

    # Anomaly detection metrics
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0

    # Per-failure-type breakdown
    per_type_top1: Dict[str, float] = field(default_factory=dict)
    per_type_top3: Dict[str, float] = field(default_factory=dict)
    per_type_count: Dict[str, int] = field(default_factory=dict)

    # Timing
    mean_train_secs: float = 0.0
    mean_analyze_secs: float = 0.0

    # Individual scenario results (for detailed inspection)
    scenario_results: List[ScenarioResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ground truth mapping
# ---------------------------------------------------------------------------

# Maps failure_type -> list of acceptable root-cause metric names.
# If the pipeline's #1 ranked cause is in this list, it's considered correct
# at the metric level (since the synthetic generator describes root causes as
# natural-language strings, not metric names).
_GROUND_TRUTH_METRICS: Dict[str, List[str]] = {
    'database_slow_query': [
        'api_latency_p50_ms', 'api_latency_p95_ms', 'api_latency_p99_ms',
    ],
    'memory_leak': [
        'memory_usage_percent',
    ],
    'network_partition': [
        'api_latency_p95_ms', 'api_latency_p99_ms', 'error_rate_percent',
    ],
    'thread_pool_exhaustion': [
        'api_latency_p50_ms', 'db_connections_active',
    ],
    'cpu_spike': [
        'cpu_utilization',
    ],
    'disk_io_spike': [
        'disk_io_wait_percent',
    ],
}


def _is_correct_root_cause(failure_type: str, predicted_metric: str) -> bool:
    """Return True if predicted_metric is an acceptable root cause for failure_type."""
    accepted = _GROUND_TRUTH_METRICS.get(failure_type, [])
    return predicted_metric in accepted


def _reciprocal_rank(failure_type: str, ranked_metrics: List[str], top_k: int = 5) -> float:
    """Compute 1/rank for first correct root cause in the ranked list (up to top_k)."""
    for i, m in enumerate(ranked_metrics[:top_k], start=1):
        if _is_correct_root_cause(failure_type, m):
            return 1.0 / i
    return 0.0


def _anomaly_precision_recall_f1(
    predicted: List[str],
    expected: List[str]
) -> Tuple[float, float, float]:
    """Compute P/R/F1 for anomalous metric detection."""
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = sum(1 for m in predicted if m in expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RCAEvaluator:
    """
    Evaluates RCAPipeline performance across a set of labeled scenarios.

    Parameters
    ----------
    config : RCAPipelineConfig | None
        Pipeline configuration.  If None uses defaults.
    anomaly_score_threshold : float
        Score above which a metric is considered detected-as-anomalous for
        the P/R/F1 calculation.
    verbose : bool
        If True, print progress while evaluating.
    """

    def __init__(
        self,
        config=None,
        anomaly_score_threshold: float = 1.0,
        verbose: bool = True,
    ):
        self.config = config
        self.anomaly_score_threshold = anomaly_score_threshold
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scenarios: List[Dict],
        share_training: bool = False,
    ) -> EvaluationReport:
        """
        Run the RCA pipeline on every scenario and compute accuracy metrics.

        Parameters
        ----------
        scenarios : list of dicts
            As returned by SyntheticMetricsGenerator.generate_evaluation_dataset().
        share_training : bool
            If True, train a single model on the first scenario's normal data
            and reuse it for all scenarios (faster but less accurate).
            If False (default), train fresh per scenario.

        Returns
        -------
        EvaluationReport with aggregated metrics and per-scenario details.
        """
        from src.rca_pipeline import RCAPipeline, RCAPipelineConfig  # local import

        cfg = self.config or RCAPipelineConfig()
        report = EvaluationReport(num_scenarios=len(scenarios))
        scenario_results: List[ScenarioResult] = []

        shared_pipeline: Optional[RCAPipeline] = None

        for idx, scenario in enumerate(scenarios):
            scenario_id = scenario['scenario_id']
            failure_type = scenario['failure_type']
            normal_data: pd.DataFrame = scenario['normal_data']
            failure_data: pd.DataFrame = scenario['failure_data']
            meta: Dict = scenario['metadata']
            normal_period_end_idx: int = scenario.get('normal_period_end_idx', len(normal_data) // 2)

            if self.verbose:
                print(f"\n[{idx+1}/{len(scenarios)}] Scenario {scenario_id}: {failure_type}")

            result = ScenarioResult(
                scenario_id=scenario_id,
                failure_type=failure_type,
                ground_truth_root_cause=meta.get('root_cause', ''),
                expected_anomalous_metrics=meta.get('expected_anomalous_metrics', []),
            )

            try:
                pipeline = RCAPipeline(config=cfg)

                # ---- Training ----
                train_df = normal_data.iloc[:normal_period_end_idx]
                t0 = time.time()
                pipeline.train(train_df)
                result.train_secs = round(time.time() - t0, 2)

                # ---- Analysis on failure window ----
                analysis_df = failure_data.iloc[max(0, normal_period_end_idx - 50):]
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rca_results = pipeline.analyze(analysis_df)
                result.analyze_secs = round(time.time() - t0, 2)

                # ---- Extract ranked causes ----
                ranked_causes_objs = rca_results.get('ranked_causes', [])
                ranked_metrics = [rc.metric for rc in ranked_causes_objs]
                result.ranked_causes = ranked_metrics

                # ---- Root cause accuracy ----
                if ranked_metrics:
                    result.top1_correct = _is_correct_root_cause(failure_type, ranked_metrics[0])
                    result.top3_correct = any(
                        _is_correct_root_cause(failure_type, m) for m in ranked_metrics[:3]
                    )
                result.reciprocal_rank = _reciprocal_rank(failure_type, ranked_metrics)

                # ---- Anomaly detection accuracy ----
                anomaly_scores: Dict[str, float] = rca_results.get('anomaly_scores', {})
                detected = [
                    m for m, s in anomaly_scores.items()
                    if s >= self.anomaly_score_threshold
                ]
                result.detected_anomalous_metrics = detected
                p, r, f1 = _anomaly_precision_recall_f1(
                    detected, meta.get('expected_anomalous_metrics', [])
                )
                result.anomaly_precision = p
                result.anomaly_recall = r
                result.anomaly_f1 = f1

                if self.verbose:
                    tick = "✅" if result.top1_correct else ("⚠️" if result.top3_correct else "❌")
                    print(f"  {tick} Top-1={result.top1_correct}  Top-3={result.top3_correct}  "
                          f"MRR={result.reciprocal_rank:.2f}  "
                          f"F1={f1:.2f}  "
                          f"time={result.analyze_secs:.1f}s")
                    if ranked_metrics:
                        print(f"     Rank #1: {ranked_metrics[0]}  (GT: {failure_type})")

                report.num_successful += 1

            except Exception as exc:
                result.error = str(exc)
                report.num_failed += 1
                if self.verbose:
                    print(f"  ❌ PIPELINE ERROR: {exc}")

            scenario_results.append(result)

        # ---- Aggregate ----
        report = self._aggregate(report, scenario_results)
        return report

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        report: EvaluationReport,
        results: List[ScenarioResult],
    ) -> EvaluationReport:
        successful = [r for r in results if r.error is None]
        report.scenario_results = results

        if not successful:
            return report

        report.top1_accuracy = np.mean([r.top1_correct for r in successful])
        report.top3_accuracy = np.mean([r.top3_correct for r in successful])
        report.mean_reciprocal_rank = np.mean([r.reciprocal_rank for r in successful])
        report.mean_precision = np.mean([r.anomaly_precision for r in successful])
        report.mean_recall = np.mean([r.anomaly_recall for r in successful])
        report.mean_f1 = np.mean([r.anomaly_f1 for r in successful])
        report.mean_train_secs = np.mean([r.train_secs for r in successful])
        report.mean_analyze_secs = np.mean([r.analyze_secs for r in successful])

        # Per-type breakdown
        failure_types = set(r.failure_type for r in successful)
        for ft in failure_types:
            ft_results = [r for r in successful if r.failure_type == ft]
            report.per_type_count[ft] = len(ft_results)
            report.per_type_top1[ft] = np.mean([r.top1_correct for r in ft_results])
            report.per_type_top3[ft] = np.mean([r.top3_correct for r in ft_results])

        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: EvaluationReport):
        """Print a formatted evaluation report to stdout."""
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║             RCA SYSTEM — EVALUATION REPORT               ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Scenarios:   {report.num_scenarios} total  "
              f"({report.num_successful} succeeded, {report.num_failed} failed)")
        print()

        # PRD targets
        top1_target = 0.65
        top3_target = 0.85
        f1_target = 0.77

        def _bar(val: float) -> str:
            filled = int(val * 20)
            return "[" + "█" * filled + "░" * (20 - filled) + "]"

        def _status(val: float, target: float) -> str:
            return "✅ PASS" if val >= target else "❌ FAIL"

        print("  ── Root Cause Ranking ──────────────────────────────────")
        print(f"  Top-1 Accuracy    : {report.top1_accuracy*100:5.1f}%  "
              f"{_bar(report.top1_accuracy)}  "
              f"target={top1_target*100:.0f}%  {_status(report.top1_accuracy, top1_target)}")
        print(f"  Top-3 Accuracy    : {report.top3_accuracy*100:5.1f}%  "
              f"{_bar(report.top3_accuracy)}  "
              f"target={top3_target*100:.0f}%  {_status(report.top3_accuracy, top3_target)}")
        print(f"  Mean Recip. Rank  : {report.mean_reciprocal_rank:.3f}")
        print()

        print("  ── Anomaly Detection ───────────────────────────────────")
        print(f"  Precision         : {report.mean_precision*100:5.1f}%")
        print(f"  Recall            : {report.mean_recall*100:5.1f}%")
        print(f"  F1 Score          : {report.mean_f1:.3f}  "
              f"target={f1_target:.2f}  {_status(report.mean_f1, f1_target)}")
        print()

        print("  ── Timing ──────────────────────────────────────────────")
        print(f"  Avg. Train time   : {report.mean_train_secs:.1f}s")
        print(f"  Avg. Analyze time : {report.mean_analyze_secs:.1f}s")
        print()

        if report.per_type_top1:
            print("  ── Per-Failure-Type Breakdown ──────────────────────────")
            header = f"  {'Failure Type':<30s} {'Count':>5} {'Top-1':>6} {'Top-3':>6}"
            print(header)
            print("  " + "-" * 52)
            for ft in sorted(report.per_type_top1):
                cnt = report.per_type_count[ft]
                t1 = report.per_type_top1[ft]
                t3 = report.per_type_top3[ft]
                tick = "✅" if t1 >= top1_target else ("⚠️ " if t3 >= top3_target else "❌")
                print(f"  {ft:<30s} {cnt:>5}  {t1*100:>5.1f}%  {t3*100:>5.1f}%  {tick}")

        print()
        print("  ── Scenario Details ────────────────────────────────────")
        for r in report.scenario_results:
            status = "ERR" if r.error else ("T1✅" if r.top1_correct else ("T3⚠️" if r.top3_correct else "❌"))
            top1_pred = r.ranked_causes[0] if r.ranked_causes else "—"
            print(f"  [{r.scenario_id}] {r.failure_type:<30s} {status}  pred={top1_pred}")
        print()

    def save_report_json(self, report: EvaluationReport, filepath: str):
        """Save report to JSON file."""
        import json

        def _ser(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, bool):
                return obj
            raise TypeError(f"Cannot serialize {type(obj)}")

        data = {
            'summary': {
                'num_scenarios': report.num_scenarios,
                'num_successful': report.num_successful,
                'num_failed': report.num_failed,
                'top1_accuracy': float(report.top1_accuracy),
                'top3_accuracy': float(report.top3_accuracy),
                'mean_reciprocal_rank': float(report.mean_reciprocal_rank),
                'mean_precision': float(report.mean_precision),
                'mean_recall': float(report.mean_recall),
                'mean_f1': float(report.mean_f1),
                'mean_train_secs': float(report.mean_train_secs),
                'mean_analyze_secs': float(report.mean_analyze_secs),
            },
            'per_type': {
                ft: {
                    'count': report.per_type_count[ft],
                    'top1': float(report.per_type_top1[ft]),
                    'top3': float(report.per_type_top3[ft]),
                }
                for ft in report.per_type_top1
            },
            'scenarios': [
                {
                    'id': r.scenario_id,
                    'failure_type': r.failure_type,
                    'top1_correct': r.top1_correct,
                    'top3_correct': r.top3_correct,
                    'reciprocal_rank': float(r.reciprocal_rank),
                    'anomaly_f1': float(r.anomaly_f1),
                    'ranked_causes': r.ranked_causes[:5],
                    'train_secs': r.train_secs,
                    'analyze_secs': r.analyze_secs,
                    'error': r.error,
                }
                for r in report.scenario_results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=_ser)
        print(f"Report saved to {filepath}")


# ---------------------------------------------------------------------------
# Standalone evaluation script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator
    from src.rca_pipeline import RCAPipelineConfig

    print("=== RCA Evaluation Framework ===\n")

    gen = SyntheticMetricsGenerator(seed=42)

    # Generate a small evaluation set (6 scenarios = one of each failure type)
    print("Generating evaluation dataset (6 scenarios, 15 days each)...")
    scenarios = gen.generate_evaluation_dataset(
        num_scenarios=6,
        normal_days=15,
        failure_duration_samples=100
    )

    # Fast config for demo
    config = RCAPipelineConfig()
    config.EPOCHS = 5
    config.PATIENCE = 3
    config.WINDOW_SIZE = 30
    config.TRAIN_STRIDE = 10
    config.MAX_LAG = 5

    evaluator = RCAEvaluator(config=config, verbose=True)
    report = evaluator.evaluate(scenarios)
    evaluator.print_report(report)
    evaluator.save_report_json(report, 'evaluation_report.json')
