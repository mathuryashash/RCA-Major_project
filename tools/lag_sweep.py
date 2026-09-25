"""L-sweep: how does the Granger max_lag setting trade off tested population
against explanation rate?

The paper claims (Section VI): "Re-running the survey at L in {3,4,5} over an
identical population shows the cliff disappears at L<=4, but recovers little:
explanations rise only from 30 to 34 while the tested population nearly
doubles, and median surviving edges falls from 2 to 1."

This script re-derives those numbers from the actual collected history on
this machine by calling tools/measure_causal_yield.py's own functions at each
L, over the *same* incident population (locked by a shared --days window), and
prints/saves a single comparison table plus the raw JSON per L.

Read only, no injection, no privileges.

    python tools/lag_sweep.py
    python tools/lag_sweep.py --lags 2,3,4,5,6 --days 30
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
    samples = incident.duration_minutes * 60 / 30.0
    return samples >= max_lag * 3 + 2


def run_one_lag(database, model, max_lag: int, lookback_hours: int) -> dict:
    incidents = engine.detect_incidents(database, model, lookback_hours=lookback_hours)
    short = [i for i in incidents if not analysable(i, max_lag)]
    usable = [i for i in incidents if analysable(i, max_lag)]

    rows = []
    support = Counter()
    failures = Counter()

    for n, incident in enumerate(usable, start=1):
        try:
            payload = engine.run_real_rca(
                database, model, max_lag=max_lag,
                start=incident.start, end=incident.end,
                trigger=f"lag-sweep-L{max_lag}",
            )
        except Exception as exc:  # noqa: BLE001 - survey must continue
            failures[type(exc).__name__] += 1
            continue

        evidence = payload["evidence"]
        verdict = evidence.get("causal_support") or "no anomaly detected"
        support[verdict] += 1
        rows.append({
            "start": str(incident.start),
            "minutes": round(incident.duration_minutes, 1),
            "edges": evidence.get("surviving_causal_edges", 0),
            "support": verdict,
        })

    explained = [r for r in rows if r["edges"] > 0]
    edges_sorted = sorted(r["edges"] for r in explained) if explained else []
    median_edges = edges_sorted[len(edges_sorted) // 2] if edges_sorted else 0

    return {
        "max_lag": max_lag,
        "incidents_found": len(incidents),
        "below_floor": len(short),
        "tested": len(rows),
        "explained": len(explained),
        "explained_pct_of_all": round(100 * len(explained) / max(len(incidents), 1), 1),
        "explained_pct_of_tested": round(100 * len(explained) / max(len(rows), 1), 1),
        "median_surviving_edges": median_edges,
        "support_breakdown": dict(support),
        "failures": dict(failures),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", default="3,4,5", help="comma-separated max_lag values")
    parser.add_argument("--days", type=int, default=30, help="history to search")
    parser.add_argument("--out", default="outputs/lag_sweep.json")
    args = parser.parse_args()

    database = config.db_path()
    model = config.app_dir() / "telemetry_model.pt"
    status = engine.model_status(model)
    if not status.exists:
        print(f"no usable model: {status.reason}")
        return 2

    lags = [int(x) for x in args.lags.split(",")]
    lookback_hours = 24 * args.days

    results = {}
    started = time.time()
    for lag in lags:
        print(f"\n=== L = {lag} ===")
        t0 = time.time()
        results[lag] = run_one_lag(database, model, lag, lookback_hours)
        r = results[lag]
        print(f"  {r['incidents_found']} incidents, {r['below_floor']} below floor, "
              f"{r['tested']} tested, {r['explained']} explained "
              f"({r['explained_pct_of_all']}% of all, {r['explained_pct_of_tested']}% of tested), "
              f"median edges {r['median_surviving_edges']}  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 78)
    print(f"{'L':>3}  {'incidents':>10}  {'tested':>7}  {'explained':>10}  "
          f"{'%all':>6}  {'%tested':>8}  {'median edges':>13}")
    for lag in lags:
        r = results[lag]
        print(f"{lag:>3}  {r['incidents_found']:>10}  {r['tested']:>7}  {r['explained']:>10}  "
              f"{r['explained_pct_of_all']:>6}  {r['explained_pct_of_tested']:>8}  "
              f"{r['median_surviving_edges']:>13}")
    print(f"\ntotal sweep time: {(time.time()-started)/60:.1f} min")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
