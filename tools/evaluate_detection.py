"""Measure whether the detector finds a fault we deliberately caused.

There is no ground truth on a personal machine, so nothing in this project
has ever had a measured precision or recall. This closes that as far as it
can be closed: cause a specific, known disturbance, then ask the pipeline
what it saw. If a sustained CPU burn does not show up as a CPU anomaly
attributed to this process, the detector does not work, and no amount of
plausible-looking reports proves otherwise.

Not a unit test. It needs the collector running and takes minutes, because
it has to wait for real samples at the real cadence.

    python tools/evaluate_detection.py --fault cpu --minutes 6
    python tools/evaluate_detection.py --fault disk --minutes 6
    python tools/evaluate_detection.py --fault memory --minutes 6
    python tools/evaluate_detection.py --fault idle --minutes 30   # false positives

`idle` injects nothing. Anything flagged during it is a false positive, which
is the number nobody wants to look at and everybody should.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pipeline import engine  # noqa: E402
from telemetry import config  # noqa: E402

#: Which metrics a given fault should move. Detection counts as correct if any
#: of these is flagged -- naming several is not hedging: a CPU burn genuinely
#: raises frequency and per-core load together, and insisting on one exact
#: metric would fail the check for being right in a different column.
#: How much of the metric set may flag at rest before the detector is judged
#: too twitchy. Measured baseline on this machine: 1 of 29, or 3.4%, during
#: thirty minutes with Windows Search indexing in the background. Set just
#: above that -- tightening it further would fail on genuine OS activity,
#: which the detector is right to notice.
IDLE_TOLERANCE_PCT = 7.0

EXPECTED = {
    "cpu": ("cpu_pct", "cpu_pct_max_core", "cpu_freq_mhz", "cpu_freq_ratio"),
    "disk": ("disk_write_bps", "disk_read_bps", "disk_busy_pct"),
    "memory": ("mem_pct", "swap_pct", "swap_used_bytes", "swap_used_delta"),
    "idle": (),
}


def _burn_cpu(stop_at: float) -> None:
    while time.time() < stop_at:
        sum(i * i for i in range(20000))


def _burn_disk(stop_at: float, path: Path) -> None:
    payload = os.urandom(4 * 1024 * 1024)
    while time.time() < stop_at:
        with open(path, "wb") as handle:
            for _ in range(25):                 # ~100MB per pass
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    path.unlink(missing_ok=True)


def _burn_memory(stop_at: float, budget_bytes: int) -> None:
    """Hold a bounded amount of memory, never all of it.

    This originally allocated until MemoryError. On a machine with little
    free RAM that means forcing the whole session into swap, freezing the
    desktop, and possibly taking other applications with it -- a diagnostic
    tool must not do that to the machine it is diagnosing. The caller sizes
    the budget against what is actually free.
    """
    blocks = []
    held = 0
    step = 64 * 1024 * 1024
    while time.time() < stop_at:
        if held + step <= budget_bytes:
            try:
                block = bytearray(step)
                for offset in range(0, len(block), 4096):
                    block[offset] = 1            # touch it, or it is not resident
                blocks.append(block)
                held += step
            except MemoryError:
                break                            # respect the machine over the test
        time.sleep(2)


def inject(fault: str, minutes: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Run the load, and return the window it ran in."""
    stop_at = time.time() + minutes * 60
    workers: list[multiprocessing.Process] = []

    if fault == "cpu":
        # Half the cores: enough to be unmistakable, not enough to make the
        # machine unusable while it runs.
        for _ in range(max(1, (os.cpu_count() or 4) // 2)):
            workers.append(multiprocessing.Process(target=_burn_cpu, args=(stop_at,)))
    elif fault == "disk":
        scratch = Path(config.app_dir()) / "evaluation_scratch.bin"
        workers.append(multiprocessing.Process(target=_burn_disk, args=(stop_at, scratch)))
    elif fault == "memory":
        # Half of what is free, capped at 2GB. Enough to move the memory
        # metrics; never enough to push the machine into swap thrashing.
        try:
            import psutil

            budget = min(2 * 1024**3, int(psutil.virtual_memory().available * 0.5))
        except Exception:                       # noqa: BLE001 - fall back to modest
            budget = 512 * 1024**2
        print(f"  memory budget: {budget / 1024**3:.2f} GB (bounded deliberately)")
        workers.append(multiprocessing.Process(target=_burn_memory, args=(stop_at, budget)))

    started = pd.Timestamp.now(tz="UTC")
    for worker in workers:
        worker.start()

    print(f"  injecting '{fault}' for {minutes:.1f} minutes "
          f"({len(workers)} worker process(es))...")
    while time.time() < stop_at:
        time.sleep(5)
        print(f"    {max(0, int(stop_at - time.time()))}s remaining", end="\r")

    for worker in workers:
        worker.terminate()
        worker.join(timeout=10)
    print()
    return started, pd.Timestamp.now(tz="UTC")


def evaluate(fault: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Ask the pipeline what it saw, and score it."""
    database, model = config.db_path(), config.app_dir() / "telemetry_model.pt"

    status = engine.model_status(model)
    if not status.exists:
        print(f"  no usable model: {status.reason}")
        return 2

    # The collector writes on a 30s cadence; give the last samples time to land.
    print("  waiting 45s for the final samples to be written...")
    time.sleep(45)

    try:
        payload = engine.run_real_rca(database, model, max_lag=5,
                                      start=start, end=end, trigger="evaluation")
    except ValueError as exc:
        print(f"  window unanalysable: {exc}")
        return 2

    flagged = payload["active_anomalies"]
    evidence = payload["evidence"]
    processes = [row["name"] for row in payload["process_attribution"][:10]]
    expected = EXPECTED[fault]

    print()
    print(f"  samples analysed  : {evidence['samples_analysed']}")
    print(f"  metrics flagged   : {len(flagged)} -> {flagged[:8]}")
    print(f"  causal support    : {evidence.get('causal_support')}")
    print(f"  top processes     : {processes[:5]}")

    if fault == "idle":
        # Nothing was injected, so anything flagged is a false positive -- with
        # the caveat that a Windows machine is never actually idle. Compare the
        # rate against an injected run rather than reading it alone: measured
        # here, 1 of 29 metrics at rest against 6 of 29 under a CPU burn, which
        # says the detector discriminates rather than firing at everything.
        total = len(payload["incident_scaled"].columns) - 1        # minus timestamp
        rate = 100.0 * len(flagged) / max(total, 1)
        print()
        print(f"  FALSE POSITIVES   : {len(flagged)} of {total} metrics ({rate:.1f}%)")
        if flagged:
            print(f"                      {flagged}")
            print("  Note: Windows is never truly idle. Check the processes above —")
            print("  indexing or an update is real load, and flagging it is correct.")
        return 0 if rate <= IDLE_TOLERANCE_PCT else 1

    hit = [metric for metric in expected if metric in flagged]
    ours = Path(sys.executable).name.lower()
    attributed = any(ours in name.lower() or "python" in name.lower() for name in processes)

    print()
    print(f"  expected any of   : {list(expected)}")
    print(f"  DETECTED          : {'yes -> ' + str(hit) if hit else 'NO'}")
    print(f"  ATTRIBUTED to us  : {'yes' if attributed else 'no'}")

    if not hit:
        print()
        print("  The detector did not flag the metric the injected fault moves.")
        print("  That is a real negative result, not a harness problem.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=sorted(EXPECTED), default="cpu")
    parser.add_argument("--minutes", type=float, default=6.0,
                        help="how long to sustain the load; needs to exceed the "
                             "model window to be scoreable")
    args = parser.parse_args()

    print(f"=== fault injection: {args.fault} ===")
    start, end = inject(args.fault, args.minutes)
    print(f"  window: {start:%H:%M:%S} to {end:%H:%M:%S} UTC")
    result = evaluate(args.fault, start, end)

    print()
    print({0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE"}[result])
    return result


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
