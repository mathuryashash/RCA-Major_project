# End-to-End Workflow

What happens from the moment someone runs the executable to the moment they
read a report — with the real timings measured on a live installation.

---

## The whole path at a glance

```
 install ──► consent ──► collect ──► (≈21 h) ──► train ──► detect ──► analyse ──► report
                            ▲                                                      │
                            └──────────── continues in the background ─────────────┘
```

---

## Stage 0 — Install and consent

**User action:** extract the ZIP, run `RCA-Desktop.exe`.

On first launch the app shows a disclosure dialog naming what is recorded, at
what cadence, what is never recorded, where it is stored, how long it is
kept, and how to erase it. Storing Event Log message text is a separate,
unticked choice.

**Nothing is collected until the user agrees.** Declining is honoured, and
asked again next launch rather than being silently permanent.

```
desktop/main.py::_ensure_collector_running
  └─ desktop/consent.py::ensure_consent
       ├─ store.connect + init_schema      ← a first run has no database yet
       ├─ collector.consent_granted?  ──► yes: proceed
       └─ ConsentDialog.exec()        ──► granted → schedule.start_now()
```

Running `RCA-Collector.exe install` additionally writes a per-user Startup
entry, a Start menu shortcut and an Add/Remove Programs entry.

---

## Stage 1 — Collection

**Runs continuously, unattended.**

Three loops in one process (`telemetry/collector.py::run_forever`):

| Every | Action | Written to |
|---|---|---|
| 30 s | psutil + NVML readings, 29 metrics | `samples` |
| 300 s (30 s under load) | Top-15 processes by CPU | `proc_samples` |
| 300 s | Event Log poll from a watermark, allowlist filtered | `events` |

A singleton mutex prevents duplicates. The loop checks `stop.flag` once per
cycle, which is how `delete-all-data` asks it to exit.

**Gap handling is the important part.** Each sample records `elapsed_ms`
since the previous one. When that exceeds 1.5× the cadence — sleep, reboot,
crash — the discontinuity is recorded rather than smoothed over, and every
consumer downstream works in contiguous segments.

### Measured reality

```
span                13.2 days
samples             10,555        (continuous collection would give ~38,000)
coverage            27.8%
segments            46
longest segment     14.4 hours
median segment      17 samples (8.5 minutes)
```

The collector runs about a quarter of the time. This is the single most
consequential number in the system.

---

## Stage 2 — Readiness

**Where a new user waits, and where the app must be honest.**

Training needs **250 clean windows**. "Clean" excludes any segment containing
an anomalous event, and windows never span a gap. With stride 5 and window 12
that is ~1,262 uninterrupted samples ≈ **21 hours** of collection.

```
analysis.baseline_status
  ├─ load_samples + load_events
  ├─ clean_baseline           ← drop segments around known faults
  ├─ contiguous_windows       ← split at every gap
  └─ windows_in(each segment) ← accumulate across all of them
```

Accumulating across segments matters. An earlier version used only the single
longest run, which on a laptop that sleeps may never reach the threshold at
all.

Stage 1 shows samples collected, longest run, current run, and time
remaining. The Train button stays disabled until the threshold is genuinely
met — the gate and the trainer share one readiness function so the UI cannot
enable a button that then fails.

---

## Stage 3 — Training

**User action:** click Train. **Measured: 8.5 s** at defaults.

```
engine.train_from_real_telemetry(progress=…)
   5%  load telemetry            load_real_telemetry
  15%  build windows per segment contiguous_windows → create_windows → concat
  25%  fit                       AnomalyDetector.train (per-epoch callback)
  85%  ─ epochs complete
  90%  reference error           median_recon_error
  95%  save artifact             model + scaler + feature list + reference
```

The scaler and feature list are saved **inside** the artifact. A model applied
under different scaling than it was trained with produces confident nonsense.

| Epochs | Window | Time |
|---|---|---|
| 5 | 12 | 8.5 s |
| 20 | 12 | 17.5 s |
| 30 | 12 | 24.6 s |
| 30 | 60 | ~64 s |

**The first epoch takes ~4.6 s against ~0.5 s for the rest** — PyTorch loads
Dynamo lazily through the optimiser's constructor on first use. Stage 1
quotes an estimate before you commit, calibrated against your machine's last
real run.

---

## Stage 4 — Incident discovery

**User action:** click Find Incidents. **Measured: 0.6 s.**

Two independent triggers produce the same record type:

- **Detector** — contiguous runs of flagged rows (≥ 3 consecutive)
- **Event** — an allowlisted Windows fault, windowed 30 min before to 5 min after

Both are filtered so that **every incident offered can actually be analysed**.
Two ways one cannot be:

- Events are kept 365 days while samples exist only while the collector ran,
  so an event can name a window holding *no telemetry at all*.
- A detector run can be 3 samples — shorter than the model window it must be
  scored through. These are widened into surrounding context that is known to
  exist, rather than discarded.

Incidents within 5 minutes are merged, then filtered — that order matters,
because two adjacent short windows can jointly cover enough contiguous
samples when neither does alone.

---

## Stage 5 — Inference

**User action:** select an incident, click Run. **Measured: 0.2 s – 12.5 s.**

```
engine.run_real_rca(progress=…)
  10%  load model artifact
  25%  load window                window_between
  40%  validate                   contiguous segment ≥ window_size, features present
  55%  score                      detect_anomalies → per-metric error + first-seen times
       └─ drift measured here (the model is already running over this window)
  70%  causal inference           stationarity → Granger → FDR → effect size → DAG
  85%  rank + attribute processes
```

The worker then generates the report at 95% and completes at 100%.

**Cost grows with the square of the window** — Granger tests every ordered
pair and the anomalous-metric count itself rises with width. Stage 2 quotes an
estimate and warns when the range is below the Granger floor.

---

## Stage 6 — The report

Four views plus Markdown and JSON export.

**Root Causes** — ranked candidates with composite score, confidence,
outflow, downstream effects.
**Causal Graph** — nodes per anomalous metric, arrows for surviving edges
labelled with lag.
**Anomaly Timeline** — five most anomalous metrics scaled 0–1, red dashed
lines at first threshold crossing.
**Report** — the full Markdown.

### What the report will refuse to say

This is the part that took the longest to get right.

- **If no edge survived**, it does not name a "primary root cause". It names
  the *leading correlated metric* and states that this is not a causal claim.
- **If the top two are within a hundredth**, it says the ranking is arbitrary.
- **If the window was too short to test causality at all**, it says so and
  distinguishes that from "tested, nothing survived" — the difference between
  a negative result and no result.
- **If the model has drifted**, it says so, and notes that analysing an older
  incident often reports this.

---

## Stage 7 — Uninstall and erase

```
uninstall         stops collection, removes startup entry, Start menu
                  shortcut and Add/Remove Programs entry.
                  KEEPS your data.

delete-all-data   stops collection, then erases the whole data directory:
                  database, model, reports, logs — plus rendered charts left
                  in the temp directory.
```

Deletion is **gated on the database unlink succeeding**, which is proof the
collector has exited. An earlier version cleared the directory immediately —
destroying `stop.flag`, which lives inside it, so the collector never saw the
request, kept the database locked, and every retry destroyed the model and
reports while the telemetry survived. Exactly inverted.

Exported reports are not touched: the user chose where those went.

---

## The loop this is meant to support

```
  incident happens  ──►  collector already recorded it
          │
          ▼
  open the app, Find Incidents, pick it
          │
          ▼
  read what was abnormal, what it correlated with, which processes were busy
          │
          ▼
  either an explanation, or an explicit statement that the data cannot support one
```

The last line is the design goal. A tool that always produces an answer is
easy to build and worthless.
