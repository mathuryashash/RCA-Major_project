# Real-Telemetry RCA — Design

**Date:** 2026-07-27
**Status:** Approved, ready for planning
**Supersedes:** synthetic data generation throughout the pipeline

## Goal

Replace all synthetic data generation with real laptop telemetry, and make the
root cause output actionable on a laptop rather than on a web service.

Two halves:

1. **Collection.** A headless collector samples `psutil` and the Windows Event
   Log into a local SQLite store. Nothing is fabricated anywhere.
2. **Root cause.** The existing LSTM autoencoder and Granger causal pipeline are
   pointed at that store, constrained by a laptop subsystem topology, and
   extended with process attribution so the answer names a process, not just a
   metric.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Usage model | Background collector, on-demand analysis | Always-on mode stays possible later as a second consumer of the store |
| Collector location | Separate process, Task Scheduler at logon | Collecting only while the GUI is open produces a sparse, biased baseline |
| Cadence, system metrics | 30 s | Baseline in ~3 days instead of ~30; short incidents survive |
| Cadence, per-process | 5 min, bursting to 30 s under load | Measured: per-process enumeration costs ~900 ms (see Measurements) |
| Failure families collected | performance, crashes, exhaustion, power | Collection is irreversible; detection is not |
| Failure families with full RCA | performance + crashes | Exhaustion needs a drift detector that does not exist yet |
| Baseline selection | Event-filtered | The Event Log already says when the machine was unhealthy |
| Regime conditioning | Columns stored, not used | Cannot backfill; enabling it later is a modelling spec |
| Compute device | CPU, threads capped at 4 | Measured: 3.9x faster than the 20-thread default; GPU cannot help a 0.52 MB model |
| Streamlit dashboard | Deleted | Superseded by the desktop app; blocks deleting the generator |

## Measurements

All measured on the development machine (20 logical cores, 375 running
processes, torch 2.12.0+cpu), not estimated.

### Collector cost

| Operation | Cost |
|---|---|
| System-wide metrics (cpu, mem, swap, disk, net, freq, battery) | 8.1 ms |
| `process_iter(['name'])` | 1.5 ms |
| `process_iter(['name','cpu_percent'])` | 909 ms |
| `process_iter(['name','memory_info'])` | 886 ms |

Any per-process attribute beyond `name` forces an `OpenProcess` per PID; the
specific attribute is irrelevant, and asking for a second one is nearly free.
Restricting the query to a top-N shortlist does **not** help, because the
shortlist can only be computed after paying the enumeration cost.

Resulting CPU budget:

| Cadence | Cost |
|---|---|
| System only @30 s | 0.027% of one core |
| + per-process @30 s | 3.06% of one core — rejected |
| + per-process @60 s | 1.54% of one core |
| + per-process @300 s | **0.33% of one core — chosen** |

Collector RSS: 27 MB.

### Training cost

Realistic shape: 8,640 samples (3 days @30 s), window 60, stride 5, 25
features, 1,716 windows, batch 32, 20 epochs.

| Threads | Epoch | 20 epochs |
|---|---|---|
| 1 | 1.53 s | 0.51 min |
| 2 | 1.20 s | **0.40 min** |
| 4 | 1.20 s | 0.40 min |
| 8 | 1.76 s | 0.59 min |
| 12 | 2.31 s | 0.77 min |
| 20 (torch default here) | 4.66 s | 1.55 min |

Torch's default of one thread per core is **3.9x slower** than capping at 2–4:
the ops are too small to amortize thread dispatch, and the LSTM's 60 sequential
timesteps limit available parallelism.

Model size: 129k parameters, 0.52 MB fp32 (measured at both 10 and 25 features).

### Why not GPU

- Training already completes in ~24 s at the corrected thread count.
- The model is 0.52 MB. A 60-step LSTM at hidden=64, batch=32 is kernel-launch
  bound, not compute bound; a GPU idles between launches.
- Bundle cost: torch is 628 MB of the current 1,508 MB distribution, and that is
  the CPU-only wheel. The CUDA wheel adds roughly 2 GB, a 2.3x increase to save
  under a minute on an operation that runs every few weeks.
- CUDA bundling reintroduces PyInstaller pain; a custom `hook-torch.py` was
  already required for torch 2.12 on Windows.

`device='cpu'` stays hardcoded. The app's diagnostics panel states "CUDA
detected but unused (model too small to benefit)" when a GPU is present, so the
behaviour reads as deliberate rather than as an oversight.

## Architecture

```
┌─────────────────┐         ┌──────────────┐        ┌────────────────────┐
│ collector.py    │ writes  │ telemetry.db │ reads  │ RCA-Desktop.exe    │
│ headless, 30s   ├────────>│   (SQLite)   │<───────┤ train / analyze /  │
│ Task Scheduler  │         │     WAL      │        │ view               │
└─────────────────┘         └──────────────┘        └────────────────────┘
```

The collector never calls the detector. An always-on mode is added later as a
second reader of the store, not as a change to how data is produced.

SQLite rather than Parquet or CSV: the collector appends one row every 30 s.
Parquet cannot append without rewriting; CSV cannot be read safely while being
written. SQLite is stdlib, supports a concurrent reader and writer under WAL,
and survives the collector being killed mid-write.

Location: `%LOCALAPPDATA%\RCA\telemetry.db`.

### Schema

```sql
samples      (ts INTEGER PRIMARY KEY, ...~25 REAL columns...)
proc_samples (ts INTEGER, pid INTEGER, name TEXT, cpu_pct REAL,
              rss INTEGER, io_read INTEGER, io_write INTEGER)
events       (ts INTEGER, provider TEXT, event_id INTEGER,
              level TEXT, message TEXT)
meta         (key TEXT PRIMARY KEY, value TEXT)
```

Uptime gaps are **derived**, not stored: a gap is any consecutive pair of
`samples.ts` more than 90 s apart. Sleep, hibernate, shutdown, and collector
crashes all present identically, which is the desired behaviour — no training
window may span one.

| Table | Rate | Year 1 | Steady state |
|---|---|---|---|
| `samples` | 2,880 rows/day | ~263 MB | grows, never purged |
| `proc_samples` | 4,320 rows/day | 95 MB | ~8 MB (30-day purge) |
| `events` | hundreds/day | ~10 MB | grows slowly |

**The 30-day purge does not affect the model.** The model trains on `samples`,
which is never purged; it has never seen `proc_samples`. The purge only means
incidents older than 30 days show mechanism without process attribution.

A trained model does not decay — it is a frozen `.pt` snapshot. The real risk is
the opposite of forgetting: it goes stale when usage patterns change and keeps
scoring confidently against an outdated notion of normal. See Retraining.

## Collection

### Metric channels (`samples`, every 30 s)

| Group | Columns |
|---|---|
| CPU | `cpu_pct`, `cpu_pct_max_core`, `cpu_freq_mhz`, `cpu_freq_ratio` |
| Memory | `mem_pct`, `mem_available_mb`, `swap_pct`, `swap_in_rate`, `swap_out_rate` |
| Disk | `disk_read_bps`, `disk_write_bps`, `disk_busy_pct`, `disk_free_pct` |
| Network | `net_sent_bps`, `net_recv_bps` |
| Load | `process_count`, `thread_count` |
| Power | `battery_pct`, `battery_drain_rate`, `power_plugged` |
| Context (stored, excluded from the model) | `on_battery`, `user_idle_sec`, `foreground_app` |

Temperature is deliberately absent: `psutil.sensors_temperatures()` returns
nothing on most Windows laptops. `cpu_freq_ratio` (current ÷ max) is the
reliable throttle proxy and needs no extra dependency.

### Per-process (`proc_samples`, every 5 min)

Top 15 processes by CPU and by RSS, deduplicated. Bursts to every 30 s while
`cpu_pct > 80`, `mem_pct > 85`, or `disk_busy_pct > 80` — free when the machine
is healthy, dense exactly when attribution will be read.

### Events (`events`, polled every 5 min)

Read via `win32evtlog` from the System and Application logs using a stored
bookmark, so restarts neither re-read nor miss entries. System and Application
are readable without elevation; the Security log requires it and is not used.

| Purpose | Provider / ID |
|---|---|
| Unexpected shutdown | Kernel-Power 41 |
| Application crash | Application Error 1000 |
| Application hang | Application Hang 1002 |
| Disk fault | `disk` 7, 51, 153 |
| Hardware error | WHEA-Logger |
| Resource exhaustion | Resource-Exhaustion-Detector 2004 |
| Change events (deployment analogue) | WindowsUpdateClient, MsiInstaller |

## Baseline and retraining

Training data is all retained history **except**:

- any 30-minute window overlapping a bad event (crash, unexpected shutdown,
  disk error, resource exhaustion, thermal throttle), and
- any window spanning an uptime gap >90 s.

The 99th-percentile reconstruction-error threshold absorbs residual
contamination.

Retraining:

- First train when ≥3 days of clean baseline exist — 8,640 samples, which at
  window 60 / stride 5 yields ~1,716 training windows. Until then Stage 1 shows
  "collecting — N days remaining".
- On demand from Stage 1.
- **Staleness alarm:** if the rolling 7-day median reconstruction error drifts
  more than 2x from its value at training time, warn that the model no longer
  matches current usage and offer a retrain. This is drift detection on the
  error signal itself — no new model, and it uses the existing
  `concept_drift_handler.py` seam.
- Retraining always uses the full retained history, so the model broadens over
  time rather than narrowing.

## Detection and root cause

### Incidents are discovered, not supplied

`failure_start_time` is currently passed in from the synthetic injection point.
It becomes `incident.start`, derived from one of two triggers that produce the
same record:

```
detector-triggered              event-triggered
────────────────                ───────────────
LSTM over history               crash / BSOD / disk error
  ↓                               ↓
anomalous windows               event timestamp
  ↓                               ↓
merge runs <5 min apart         window = [t-30min, t+5min]
drop runs <3 windows (90 s)
  ↓                               ↓
        └──────── Incident ────────┘
   {start, end, peak_severity, trigger, metrics[]}
```

The 3-window minimum suppresses single-sample noise; the 5-minute merge stops
one episode fragmenting into several reports.

The event-triggered path is what makes crash diagnosis work at all: a BSOD
produces no gradual metric anomaly — the machine stops — so a detector-only
design would miss it. The event defines the window and RCA asks what was
abnormal in the preceding 30 minutes.

### Real events replace the fabricated one

`engine.run_causal_inference` currently invents a "deployment" 20 minutes before
every incident, and that fiction feeds the ranker's `event_correlation` term.
Real events fill the same slot with the same `events_df` shape;
`EventCorrelator.correlate()` is unchanged. Windows Update, driver installs, and
MSI installs are the genuine laptop analogue of a code deployment.

### Topology prior

Every metric maps to a subsystem, and only physically plausible directed edges
survive Granger:

Using only collected column names — note that `cpu_temp` and `fan_rpm` are
deliberately not collected, so the throttle path is expressed through
`cpu_freq_ratio`:

```
process_count ──> cpu_pct ──> cpu_freq_ratio        (sustained load throttles)
thread_count  ──> cpu_pct

mem_pct ──> swap_out_rate ──> disk_busy_pct ──> cpu_pct
disk_free_pct ──> swap_out_rate                     (no room to page out)

net_recv_bps ──> disk_write_bps                     (downloads hit disk)

power_plugged ──> cpu_freq_ratio                    (power-saving throttle)
battery_pct   ──> cpu_freq_ratio
```

Edges absent from the table are pruned before ranking, so `net_recv_bps →
mem_pct` cannot be inferred however well it fits statistically. This reuses the existing
`DynamicGraphGenerator.refine_causal_graph()` seam; the Jaeger service lookup is
replaced by a static subsystem adjacency table. Same interface, no new
component.

Without this, the 40%-weighted `causal_outflow` term crowns whichever metric has
the most outgoing edges — on a laptop always CPU or temperature, because
everything is thermally coupled. Physically true, diagnostically useless.

### Process attribution

The metric graph explains mechanism; attribution answers who.

1. Take the top-ranked metric from the causal ranker.
2. Pull `proc_samples` for the incident window and the 30 minutes preceding it.
3. Compute each process's delta on the metric's governing resource — Δrss for
   memory, Δcpu_pct for CPU, Δio_bytes for disk.
4. Rank by share of the total delta.
5. If the top process explains <30% of the delta, report **"diffuse — no single
   process responsible"**.

Rule 5 matters: memory pressure from forty browser tabs is genuinely diffuse,
and a system that always names a culprit will confidently name the wrong one.

### Output

```
Incident  2026-07-27 14:32 → 14:38  (6m)   confidence High

Mechanism    mem_pct ──> swap_out_rate ──> disk_busy_pct ──> cpu_pct
             (Granger, p<0.05, lags 1-3 samples)

Attribution  chrome.exe   +5.9 GB RSS   78% of delta
             Code.exe     +1.1 GB RSS   15% of delta

Correlated   Resource-Exhaustion-Detector 2004 @ 14:33
```

## User interface changes

Stage 1 — the `baseline_days` slider drove synthetic generation and is removed.
It becomes a read-only status line: days collected, clean windows available,
model age, staleness state.

Stage 2 — the failure-scenario dropdown was a synthetic-scenario picker. It
becomes a list of detected incidents plus a custom range selector:

```
Detected incidents (last 7 days)
  2026-07-27 14:32   6m    High     disk_busy_pct     [detector]
  2026-07-26 09:15   2m    Medium   mem_pct           [detector]
  2026-07-25 23:41   —     Critical Kernel-Power 41   [event]
  ─────────────────────────────────────────────────────
  or analyze custom range:  [ from ] [ to ]
```

## Synthetic removal inventory

| File | What | Action |
|---|---|---|
| `data_ingestion/synthetic_generator.py` | the generator | delete |
| `pipeline/engine.py:19,27-60` | `generate_data()` | replace with `load_baseline()` / `load_window()` |
| `pipeline/engine.py:195-199` | fabricated deployment event | real `events` rows |
| `pipeline/engine.py:287-290` | `ground_truth` block | delete |
| `desktop/workers.py:26,73` | both workers call `generate_data` | read from store |
| `desktop/views/stage2_view.py:14` | scenario dropdown | incident list |
| `train_and_run.py:34,91,116` | CLI generates data | CLI reads store |
| `reporting/dashboard.py` | Streamlit app, 5 call sites | delete |
| `reporting/anomaly_simulator.py` | synthetic demo tool | delete |
| `anomaly_detection/anomaly_scorer.py:119-122` | `__main__` demo block | delete |
| `causal_inference/causal_engine.py:553-563` | `__main__` demo block | delete |
| `tests/test_pipeline_engine.py` | 3 tests call `generate_data` | rewrite against fixture |
| `models/lstm_autoencoder.py:178` | stale print string | edit |

Dead once real telemetry lands, all server-infrastructure sources with no laptop
meaning: `jaeger_connector.py`, `deployment_listener.py`,
`cloudwatch_connector.py`, `prometheus_connector.py`.

Reports lose the `ground_truth` block — real incidents have no oracle. It is
replaced by an honest confidence statement: composite score, share of delta
explained by attribution, number of Granger edges surviving pruning, and whether
the incident was detector- or event-triggered.

## Dependencies and packaging

Added to `requirements.txt`:

```
psutil>=5.9.0
pywin32>=306; sys_platform == "win32"
```

`psutil` was not previously declared at all. Removed with the Streamlit
dashboard: `streamlit`.

**`packaging/excludes.txt` must be regenerated.** That list is derived from the
current dependency set — the app's real import closure plus its declared
transitive dependencies — so adding `pywin32`/`psutil` and dropping `streamlit`
invalidates it. A stale entry surfaces as an ImportError at app startup, not as
a build failure.

## Failure handling

| Condition | Behaviour |
|---|---|
| <3 days baseline | Stage 1 disabled, "collecting — N days remaining" |
| Laptop slept mid-window | gap >90 s splits the series; no window spans it |
| `psutil.NoSuchProcess` mid-tick | skip that process, keep the tick |
| DB locked | WAL plus retry; collector drops the tick rather than blocking |
| Disk full | collector stops writing, logs, keeps running |
| Event Log access denied | System/Application need no elevation; Security unused |
| Incident older than 30 days | mechanism shown, attribution "process detail purged" |
| Model missing or stale | Stage 2 disabled with a retrain prompt |

## Testing

Generating fake telemetry to test with is precisely what this design removes, so
tests use a **recorded fixture** instead: the real collector is run for ~2 hours
on a development machine and the resulting SQLite file (a few hundred KB) is
committed as `tests/fixtures/telemetry_sample.db`. Real data, deterministic, no
cost at test time.

Tests, all covering logic that can silently be wrong:

- gap detection splits windows correctly across a real sleep gap
- baseline filter excludes windows overlapping a seeded bad event
- incident segmentation merges runs <5 min apart and drops runs <3 windows
- topology prior prunes a known-implausible edge
- attribution shares sum to ~1.0, and the <30% case reports "diffuse"

## Explicitly out of scope

- Drift detector for resource exhaustion. Data is collected; the detector gets
  its own spec once real history exists to tune thresholds against, rather than
  being guessed blind.
- Regime-conditioned modelling. Context columns are collected so this stays
  possible; enabling it is a modelling change, not a collection change.
- Always-on live monitoring. The architecture keeps it available as a second
  consumer of the store.
- Battery and power causal analysis. Collected and threshold-flagged only.
- Cross-machine or fleet analysis. Single machine, single store.
