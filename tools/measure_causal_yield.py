"""How often does the causal layer actually produce an answer?

Every causal claim in this project rests on two injected faults. That is an
anecdote, not a rate. This runs the real pipeline over every incident the
detector can find in the collected history and reports the distribution:
how often an edge survives, how often nothing was ever tested because the
window was too short, and how often the statistics found something the
subsystem map then refused.

The topology prune in particular has been treated as significant on the
strength of a single observation. Ninety-odd incidents will say whether it is
a systematic constraint or a one-off.

No injection, no privileges, no effect on the machine -- it only reads.

    python tools/measure_causal_yield.py
    python tools/measure_causal_yield.py --max-lag 5 --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline import engine  # noqa: E402
from telemetry import config  # noqa: E402


def analysable(incident, max_lag: int) -> bool:
    """Windows below the Granger floor cannot produce an edge at any setting.

    Counting them as "no causal chain" is the conflation the reporting change
    exists to prevent, so they are separated here too rather than averaged in.
    """
    samples = incident.duration_minutes * 60 / 30.0
    return samples >= max_lag * 3 + 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--days", type=int, default=30, help="history to search")
    parser.add_argument("--limit", type=int, default=0, help="0 = every incident")
    parser.add_argument("--out", default="outputs/causal_yield.json")
    args = parser.parse_args()

    database = config.db_path()
    model = config.app_dir() / "telemetry_model.pt"
    status = engine.model_status(model)
    if not status.exists:
        print(f"no usable model: {status.reason}")
        return 2

    print("finding incidents in collected history ...")
    incidents = engine.detect_incidents(database, model, lookback_hours=24 * args.days)
    short = [i for i in incidents if not analysable(i, args.max_lag)]
    usable = [i for i in incidents if analysable(i, args.max_lag)]
    if args.limit:
        usable = usable[: args.limit]

    print(f"  {len(incidents)} incidents, {len(short)} below the Granger floor, "
          f"{len(usable)} analysable\n")

    rows = []
    support = Counter()
    failures = Counter()
    started = time.time()

    for n, incident in enumerate(usable, start=1):
        label = f"[{n}/{len(usable)}] {incident.start:%m-%d %H:%M} {incident.duration_minutes:6.1f}m"
        try:
            began = time.time()
            payload = engine.run_real_rca(
                database, model, max_lag=args.max_lag,
                start=incident.start, end=incident.end, trigger="yield-survey",
            )
        except Exception as exc:                    # noqa: BLE001 - survey must continue
            failures[type(exc).__name__] += 1
            print(f"{label}  unanalysable: {exc}")
            continue

        evidence = payload["evidence"]
        verdict = evidence.get("causal_support") or "no anomaly detected"
        support[verdict] += 1
        top = payload["root_causes"][0]["metric"] if payload["root_causes"] else None
        rows.append({
            "start": str(incident.start),
            "minutes": round(incident.duration_minutes, 1),
            "trigger": incident.trigger,
            "samples": evidence.get("samples_analysed"),
            "flagged": evidence.get("anomalous_metrics", 0),
            "pairs_accepted": evidence.get("causal_pairs_tested", 0),
            "pruned_by_topology": evidence.get("pairs_pruned_by_topology", 0),
            "edges": evidence.get("surviving_causal_edges", 0),
            "support": verdict,
            "top_metric": top,
            "seconds": round(time.time() - began, 1),
        })
        print(f"{label}  flagged={rows[-1]['flagged']:>2}  "
              f"pairs={rows[-1]['pairs_accepted']:>2}  edges={rows[-1]['edges']:>2}  "
              f"{verdict}")

    print("\n" + "=" * 72)
    print(f"analysed {len(rows)} incidents in {(time.time()-started)/60:.1f} min")
    print(f"{len(short)} more were below the Granger floor and could not be tested\n")

    total = max(len(rows), 1)
    for verdict, count in support.most_common():
        print(f"  {count:>4} ({100*count/total:5.1f}%)  {verdict}")
    if failures:
        print("\n  unanalysable:")
        for name, count in failures.most_common():
            print(f"  {count:>4}            {name}")

    explained = [r for r in rows if r["edges"] > 0]
    if explained:
        edges = sorted(r["edges"] for r in explained)
        print(f"\n  surviving edges where any survived: "
              f"min {edges[0]}, median {edges[len(edges)//2]}, max {edges[-1]}")
        print("  most common leading metric where explained: "
              f"{Counter(r['top_metric'] for r in explained).most_common(3)}")

    pruned = [r for r in rows if r["pruned_by_topology"] > 0]
    print(f"\n  incidents where the topology map removed an accepted pair: "
          f"{len(pruned)} of {total} ({100*len(pruned)/total:.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "max_lag": args.max_lag,
        "incidents_found": len(incidents),
        "below_granger_floor": len(short),
        "analysed": len(rows),
        "support": dict(support),
        "failures": dict(failures),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
