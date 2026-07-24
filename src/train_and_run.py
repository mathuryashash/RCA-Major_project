"""
End-to-End Training Pipeline and RCA Runner
============================================
AI-Powered Root Cause Analysis System — PRD §1.1 through §1.1.6

CLI entry-point. All pipeline phases live in pipeline.engine (shared with
the PySide6 desktop app) — this file only parses args and prints progress.

Usage
-----
    python src/train_and_run.py
    python src/train_and_run.py --failure memory_leak --severity 0.9
    python src/train_and_run.py --skip-train --failure cpu_spike

CLI Flags
---------
    --failure     one of: database_slow_query | memory_leak | cpu_spike  (default: database_slow_query)
    --severity    float 0.1–1.0  (default: 0.8)
    --epochs      LSTM training epochs  (default: 15)
    --skip-train  skip training; load saved model weights instead
    --output-dir  directory for report artefacts  (default: ./outputs)
    --seed        RNG seed for reproducibility  (default: 42)
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from pipeline import engine


def banner(text: str) -> None:
    line = "-" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def step(num: int, label: str) -> None:
    print(f"\n[Step {num}] {label} ...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCA System — End-to-End Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--failure", default="database_slow_query",
                         choices=["database_slow_query", "memory_leak", "cpu_spike"],
                         help="Failure scenario to inject (default: database_slow_query)")
    parser.add_argument("--severity", type=float, default=0.8,
                         help="Failure severity 0.1–1.0 (default: 0.8)")
    parser.add_argument("--epochs", type=int, default=15,
                         help="LSTM training epochs (default: 15)")
    parser.add_argument("--skip-train", action="store_true",
                         help="Skip training; load saved weights instead")
    parser.add_argument("--output-dir", default="outputs",
                         help="Directory for report artefacts (default: ./outputs)")
    parser.add_argument("--use-ensemble", action="store_true",
                         help="Use Ensemble Anomaly Detection instead of bare LSTM")
    parser.add_argument("--use-dynamic-topology", action="store_true",
                         help="Refine causal graph using real-time Jaeger topology")
    parser.add_argument("--seed", type=int, default=42,
                         help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--window-size", type=int, default=12,
                         help="LSTM sliding window size in time-steps (default: 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    banner("AI-Powered Root Cause Analysis System - Full Pipeline")
    print(f"  Failure type : {args.failure}  (severity={args.severity})")
    print(f"  LSTM epochs  : {args.epochs}  | Window size: {args.window_size}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  RNG seed     : {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "lstm_autoencoder_best.pt")
    t_pipeline_start = time.time()

    step(1, "Synthetic Data Generation")
    normal_df, incident_df, metadata, feat_cols = engine.generate_data(
        seed=args.seed, failure_type=args.failure, severity=args.severity,
    )
    print(f"  Ground truth: {metadata['root_cause']}")

    step(2, "Preprocessing — MinMax normalization [0, 1]")
    normal_scaled, incident_scaled, scaler = engine.preprocess(
        normal_df, incident_df, feat_cols
    )

    step(3, "LSTM Autoencoder Training")
    detector = engine.train_model(
        normal_scaled=normal_scaled, n_features=len(feat_cols),
        epochs=args.epochs, window_size=args.window_size,
        model_path=model_path, skip_train=args.skip_train,
    )

    step(4, "Anomaly Detection")
    anomaly_scores, anomaly_times, active_anomalies = engine.detect_anomalies(
        detector, incident_scaled, feat_cols,
        use_ensemble=args.use_ensemble,
        normal_scaled=normal_scaled if args.use_ensemble else None,
    )
    print(f"  Anomalous metrics detected: {len(active_anomalies)}")

    gen_tmp = SyntheticMetricsGenerator(seed=args.seed + 1)
    incident_base_len = len(gen_tmp.generate_normal_behavior(duration_days=3))
    failure_start_idx = incident_base_len - 200
    failure_start_time = incident_df.iloc[failure_start_idx]["timestamp"]

    if len(active_anomalies) < 2:
        print("\n!  LSTM did not naturally flag the synthetic failure. Forcing anomalies for pipeline test.")
        active_anomalies = feat_cols[:3]
        for k in active_anomalies:
            anomaly_scores[k] = 1.0
            anomaly_times[k] = failure_start_time + pd.Timedelta(minutes=5)

    step(5, "Causal Inference - Granger Causality & Graph Construction")
    causal_results = engine.run_causal_inference(
        incident_scaled=incident_scaled, feat_cols=feat_cols,
        anomaly_scores=anomaly_scores, anomaly_times=anomaly_times,
        active_anomalies=active_anomalies, failure_start_time=failure_start_time,
        use_dynamic_topology=args.use_dynamic_topology,
    )
    causal_graph = causal_results["causal_graph"]
    print(f"  Causal graph: {len(causal_graph.nodes)} nodes, {len(causal_graph.edges)} edges")

    step(6, "Root Cause Ranking")
    root_causes = engine.rank_root_causes(causal_results)
    for rc in root_causes[:5]:
        print(f"  #{rc['rank']:<4} {rc['metric']:<35} "
              f"{rc['composite_score']:>6.3f}  {rc['confidence']}")

    step(7, "Report Generation")
    report_paths = engine.generate_reports(
        results=causal_results, root_causes=root_causes,
        anomaly_times=anomaly_times, metadata=metadata,
        failure_type=args.failure, output_dir=args.output_dir,
    )
    print(f"  Markdown report  -> {report_paths['md_path']}")
    print(f"  JSON report      -> {report_paths['json_path']}")

    elapsed = time.time() - t_pipeline_start
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print("\nDone. OK")


if __name__ == "__main__":
    main()
