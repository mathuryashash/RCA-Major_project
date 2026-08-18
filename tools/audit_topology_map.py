"""Which causal directions does the subsystem map actually reject?

The yield survey found that the hand-written map in `dynamic_graph.py`
discards 43% of every pair the statistics accept -- more than multiple-testing
correction and the effect-size floor put together. That map has never been
checked against anything. It was written from intuition about how a laptop
behaves, and it is now the largest filter in the causal pipeline.

This collects the pairs it rejects across every analysable incident and groups
them by subsystem transition, so the question stops being "is the map right?"
in the abstract and becomes "does this system permit disk to affect CPU, and
should it?".

Read only. No injection.

    python tools/audit_topology_map.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal_inference.dynamic_graph import DEPENDENCIES, DynamicGraphGenerator  # noqa: E402
from pipeline import engine  # noqa: E402
from telemetry import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", default="outputs/topology_audit.json")
    args = parser.parse_args()

    database = config.db_path()
    model = config.app_dir() / "telemetry_model.pt"
    if not engine.model_status(model).exists:
        print("no usable model")
        return 2

    topology = DynamicGraphGenerator()
    floor = args.max_lag * 3 + 2

    incidents = [
        incident
        for incident in engine.detect_incidents(database, model, lookback_hours=24 * args.days)
        if incident.duration_minutes * 2 >= floor
    ]
    print(f"auditing {len(incidents)} analysable incidents ...\n")

    kept: Counter = Counter()
    pruned: Counter = Counter()
    cycle_broken: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    unknown_metric: Counter = Counter()

    for n, incident in enumerate(incidents, start=1):
        try:
            payload = engine.run_real_rca(
                database, model, max_lag=args.max_lag,
                start=incident.start, end=incident.end, trigger="topology-audit",
            )
        except Exception:                           # noqa: BLE001
            continue
        results = payload.get("causal_results")
        if not results:
            continue

        accepted = results.get("granger_results") or {}
        survivors = set(results["causal_graph"].edges)

        for (cause, effect), info in accepted.items():
            source = topology.subsystem_for_metric(cause)
            target = topology.subsystem_for_metric(effect)
            transition = f"{source} -> {target}"
            if source is None or target is None:
                unknown_metric[cause if source is None else effect] += 1
            # Two filters run between acceptance and the final graph, and
            # conflating them misattributes the map's influence: cycles are
            # broken inside CausalGraphBuilder.build() *before*
            # refine_causal_graph() ever sees the edge. Ask the map directly
            # rather than inferring its verdict from the survivors.
            permitted = topology.is_path_possible(cause, effect)
            if (cause, effect) in survivors:
                kept[transition] += 1
            elif not permitted:
                pruned[transition] += 1
                examples[transition].append({
                    "cause": cause, "effect": effect,
                    "p_value": float(info["p_value"]), "lag": int(info["optimal_lag"]),
                    "strength": float(info["strength"]),
                    "at": str(incident.start),
                })
            else:
                cycle_broken[transition] += 1
        if n % 20 == 0:
            print(f"  {n}/{len(incidents)} ...")

    total_pruned = sum(pruned.values())
    total_kept = sum(kept.values())
    total_cycle = sum(cycle_broken.values())
    accepted = total_kept + total_pruned + total_cycle
    print("\n" + "=" * 74)
    print(f"pairs accepted by the statistics : {accepted}")
    print(f"  survived to the final graph    : {total_kept}"
          f"  ({100 * total_kept / max(accepted, 1):.0f}%)")
    print(f"  rejected by the subsystem map  : {total_pruned}"
          f"  ({100 * total_pruned / max(accepted, 1):.0f}%)")
    print(f"  removed by cycle-breaking      : {total_cycle}"
          f"  ({100 * total_cycle / max(accepted, 1):.0f}%)\n")

    print(f"{'subsystem transition':28}{'rejected':>10}{'kept':>7}   strongest rejected pair")
    for transition, count in pruned.most_common():
        best = max(examples[transition], key=lambda e: e["strength"])
        print(f"{transition:28}{count:>10}{kept.get(transition, 0):>7}   "
              f"{best['cause']} -> {best['effect']} "
              f"(strength {best['strength']:.3f}, lag {best['lag']})")

    if kept:
        print(f"\n{'transitions the map allows and that survived':50}")
        for transition, count in kept.most_common():
            print(f"  {transition:28}{count:>6}")

    if unknown_metric:
        print("\nmetrics with no declared subsystem (rejected automatically):")
        for metric, count in unknown_metric.most_common():
            print(f"  {metric:28}{count:>6}")

    if cycle_broken:
        print("\nremoved by cycle-breaking, not by the map:")
        for transition, count in cycle_broken.most_common():
            print(f"  {transition:28}{count:>6}")

    print("\nthe map, for reference:")
    for source, target in DEPENDENCIES:
        print(f"  {source} -> {target}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kept": dict(kept), "pruned": dict(pruned),
        "cycle_broken": dict(cycle_broken),
        "unknown_metric": dict(unknown_metric),
        "examples": {k: v[:20] for k, v in examples.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
