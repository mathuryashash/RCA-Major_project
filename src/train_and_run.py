"""Train and run RCA against the local telemetry collector database."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import engine
from telemetry import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RCA from collected local telemetry")
    parser.add_argument("--db", default=str(config.db_path()), help="Path to telemetry.db")
    parser.add_argument("--model", default=str(config.app_dir() / "telemetry_model.pt"), help="Model artifact path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--train", action="store_true", help="Train/retrain before RCA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train:
        baseline, features, _detector, _scaler = engine.train_from_real_telemetry(
            args.db, args.model, args.epochs, args.window_size,
        )
        print(f"Trained on {len(baseline):,} clean samples across {len(features)} features.")
    result = engine.run_real_rca(args.db, args.model, args.lookback_hours, args.max_lag)
    if not result["active_anomalies"]:
        print("No anomalies detected in the selected observed window.")
        return
    print(f"Detected {len(result['active_anomalies'])} anomalous metrics.")
    for candidate in result["root_causes"][:5]:
        print(f"#{candidate['rank']} {candidate['metric']} {candidate['composite_score']:.3f} {candidate['confidence']}")


if __name__ == "__main__":
    main()
