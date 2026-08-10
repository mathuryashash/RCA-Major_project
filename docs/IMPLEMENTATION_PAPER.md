# LocalRCA: On-Device Root Cause Analysis for Windows Endpoints

**An implementation paper**

Version 1.1.0 · 2026-08-11

---

## Abstract

LocalRCA is a Windows desktop application that continuously records system
telemetry on a single machine, learns that machine's normal behaviour with an
LSTM autoencoder, and — after an incident — attempts to explain what went
wrong using constrained Granger causality over the anomalous metrics. It runs
entirely on the endpoint: no telemetry leaves the machine, and the system
makes no network connections.

This paper documents what was built, what was measured, and where the
measurements contradict the design's assumptions. It is written from
instrumented runs on a live installation over 13.2 days, not from intended
behaviour. Several of the most useful results are negative, and they are
reported as such: the causal layer produced no supported causal chain on the
majority of real incidents examined, and the dominant limiting factor turned
out to be collection coverage rather than model quality.

---

## 1. Problem and motivation

Post-hoc diagnosis of a personal or workstation-class machine is poorly
served by existing tooling. Cloud observability stacks assume a fleet, a
network path, and a willingness to export telemetry. Windows' own Event Log
records that a fault occurred but rarely why. The gap addressed here is a
single machine, offline, where the user wants an explanation for a slowdown or
crash that has already happened.

Three constraints followed from that framing and shaped the entire design:

1. **No egress.** Telemetry describing which applications a person runs is
   sensitive. The system was built so that nothing is transmitted.
2. **No ground truth.** There are no labelled incidents on a personal machine.
   The system cannot be trained to recognise known faults, and cannot be
   evaluated against a labelled test set.
3. **Cold start is unavoidable.** "Normal" must be learned from the machine
   itself, so the tool is useless until it has observed enough normal
   behaviour.

Constraint 2 is the most consequential and is revisited in §7.

---

## 2. System architecture

Two processes, deliberately separated:

```
┌─────────────────────┐        ┌──────────────────────┐
│  RCA-Collector.exe  │        │   RCA-Desktop.exe    │
│  (console, at logon)│        │   (windowed GUI)     │
│                     │        │                      │
│  30s  system sample │──────► │  Stage 1: training   │
│  300s process sample│  SQLite│  Stage 2: inference  │
│  Event Log ingest   │◄───────│  reports & figures   │
└─────────────────────┘        └──────────────────────┘
         %LOCALAPPDATA%\RCA\telemetry.db
```

The split exists because collection must survive the GUI being closed. A
single-process design was tried first and abandoned: the desktop application
resolved its own executable path when asked to start collection, and in a
frozen build that path *is* the GUI, so it launched a second window, which
launched a third. The collector is now located by explicit binary name, and
refuses to launch if the expected sibling is absent rather than falling back
to the running executable.

### 2.1 Collection

| Stream | Cadence | Content |
|---|---|---|
| System metrics | 30 s | 29 features: CPU (aggregate, max core, frequency), memory, swap, disk (read/write/busy/free), network (sent/recv), battery, GPU via NVML |
| Process samples | 300 s, 30 s under load | Top-15 by CPU: name, CPU %, RSS, I/O bytes |
| Windows events | 300 s poll | Allowlisted providers only: Kernel-Power 41, Application Error 1000, Application Hang 1002, disk 7/51/153, WHEA, Resource-Exhaustion 2004, Update, MsiInstaller |

Only an allowlist of event families is retained; the collector advances its
read watermark across all events but discards non-allowlisted records before
storage. Event message text is stored only under an explicit opt-in and is
redacted for user paths, UNC paths, URLs, email addresses and the username
first. Window titles, keystrokes and file contents are never captured.

Retention is 30 days for process samples and 365 days for events. This
asymmetry matters and is a source of a defect discussed in §6.2.

### 2.2 Detection

An LSTM autoencoder is trained only on *clean* baseline segments — periods
with no anomalous event and no collection gap. Reconstruction error per metric
is thresholded to flag anomalies.

The training data are built as **windows within contiguous segments**, never
spanning a gap. This was not the original design. Training initially used the
single longest uninterrupted run, which discarded most of the collected
history: on a laptop that sleeps, a single run long enough to satisfy the
window requirement may never occur. Accumulating windows across all clean
segments raised usable training data substantially on the same database.

### 2.3 Causal inference

For metrics flagged anomalous in an incident window:

1. Each series is differenced until an ADF test no longer rejects
   non-stationarity (maximum two rounds).
2. Pairwise Granger causality is tested at lags 1..*max_lag*.
3. p-values are corrected for multiple testing (FDR).
4. An effect-size floor is applied, using the F-statistic as a bounded proxy:
   `f / (f + n)`.
5. Surviving edges form a directed graph; cycles are broken; PageRank supplies
   a topology-centrality term.

Root causes are ranked by a composite of causal outflow, causal inflow,
temporal priority, anomaly severity and event correlation.

**The multiple-testing correction and effect-size floor are the honest core of
this design, and they are also why the system usually reports nothing.** That
result is examined in §5.

---

## 3. Measured characteristics

All figures below are from a live installation, measured 2026-08-11.

### 3.1 Dataset

| Property | Value |
|---|---|
| Observation span | 13.2 days |
| System samples | 10,555 |
| Process samples | 243,089 |
| Event records | 4,194 |
| Features per sample | 29 |
| Database size | 28.9 MB |
| Clean samples available for training | 8,721 |

### 3.2 Collection coverage — the dominant finding

| Property | Value |
|---|---|
| **Coverage against continuous sampling** | **27.8%** |
| Contiguous segments | 46 |
| Longest unbroken segment | 14.4 hours |
| **Median segment length** | **17 samples (8.5 minutes)** |

At 30-second cadence, 13.2 days of continuous collection would yield ~38,000
samples; 10,555 were recorded. **The collector runs roughly a quarter of the
time, in fragments whose median length is under nine minutes.**

Part of this is measurement artefact — the collector was repeatedly terminated
during development builds. But the architecture contributes: the collector is
registered in the per-user Startup folder and nothing supervises it, so a
crash or a sleep ends a segment permanently until the next logon.

This single number explains more downstream behaviour than any other, and
§5 traces its consequences.

### 3.3 Runtime cost

Training, on 1,701 windows from 8,516 clean samples:

| Epochs | Window | Time |
|---|---|---|
| 5 | 12 | 8.5 s |
| 20 | 12 | 17.5 s |
| 30 | 12 | 24.6 s |
| 5 | 60 | 13.2 s |
| 30 | 60 | ~64 s |

Per-epoch cost is linear in window length (an LSTM traverses every timestep)
and linear in window count. Fitted as
`(0.40 + 0.031 × window_size) × (n_windows / 1701)` seconds per epoch, with
held-out points landing within 0.05 s.

**The first epoch costs ~4.6 s against ~0.5 s for subsequent ones.** PyTorch
imports Dynamo lazily through the optimiser's constructor on first use. This
is invisible in aggregate timings and was only found by instrumenting
per-epoch boundaries.

Inference, measured across window sizes:

| Samples | Anomalous metrics | Time |
|---|---|---|
| 104 | 5 | 0.8 s |
| 464 | 6 | 1.4 s |
| 1,334 | 10 | 12.5 s |

Cost grows with the *square* of the sample count, because Granger tests every
ordered pair and the anomalous-metric count itself rises with window width.
Modelled as `0.7 + 6.7×10⁻⁶ × samples²`.

**Training compute is not a limiting factor.** Even maximal settings complete
in about a minute. The binding constraint is the ~21 hours of clean
collection needed before training unlocks at all — which, at 27.8% coverage,
is several days of wall-clock time.

---

## 4. Distribution

PyInstaller, two executables, 1,531 MB, unsigned. Notable packaging results:

- **`optree` was excluded but its metadata was not.** PyTorch treats optree as
  optional and decides availability via `importlib.metadata.version`. The
  frozen build shipped `optree-0.18.0.dist-info` *without* the package, so
  PyTorch read a version, concluded it was present, imported it, and failed
  inside Adam's constructor. Training worked from source and failed only when
  packaged. Excluding a lazily-imported dependency while shipping its metadata
  is a general hazard of static-closure-based exclusion lists.
- **Excludes that break the thing they trim.** `torch.export` and
  `torch._inductor` were excluded as tracing-only. Both are imported at module
  scope — by `torch/__init__.py` and `torch._dynamo.guards` respectively — so
  the exclusions would have broken `import torch` outright had they taken
  effect.
- **A windowed build starts with no valid stdout/stderr.** The packaged
  application terminated with `0xC0000409` in `Qt6Core.dll` roughly forty
  seconds after launch, with no diagnostic anywhere. It survives when file
  descriptors 1 and 2 are pointed at the null device before Qt is imported.
  The mechanism remains unestablished: it is *not* PySide6 escalating a slot
  exception to `qFatal`, which was measured and disproved — PySide6 survives
  such exceptions with or without valid descriptors. The fix is validated by
  outcome, not by mechanism, and this is stated in the code.

---

## 5. Evaluation, and what it does not show

### 5.1 Detection

The autoencoder's reference reconstruction error is recorded at training time
and compared against current error at each analysis. A drift ratio above 2.0
marks the model stale.

Observed: a model trained on data through 30 July scored 57× its reference
error against current data. Retraining on 4,608 fresh clean samples brought
drift to **1.54×** against recent windows.

A caveat found while measuring this: **staleness is computed against whichever
window is being analysed**, so analysing a historical incident with a freshly
trained model reports the model as stale. That is a property of the window,
not the model, and the current report conflates them.

### 5.2 Causal inference — a negative result

Across 26 incidents from the live database, the causal layer produced **zero
surviving edges on the majority**. Widening a single incident window shows why:

| Samples in window | Edges | Leading candidate |
|---|---|---|
| 16 | 0 | `disk_free_pct` |
| 56 | 0 | `battery_drain_rate` |
| 96 | 0 | `cpu_freq_mhz` |
| 256 | 0 | `cpu_freq_mhz` |
| 496 | **3** | `disk_read_bps` (0.762) |

Two distinct failures are visible here.

**First, a window can be too small to test at all.** Granger requires
`max_lag × 3` aligned observations and differencing consumes up to two more.
An 8-minute incident window holds 16 samples against a floor of 17, so *no
pair is ever compared*. The system reported this identically to "tested,
nothing survived" — the difference between a negative result and no result.
This is now distinguished explicitly in the report.

**Second, and more seriously: with no edges the ranking is meaningless by
construction.** Every metric receives identical graph influence and zero
outflow, so the composite score collapses to temporal priority and severity
alone. In the run above the top two candidates scored 0.4620 and 0.4616 — a
gap of 0.0004 — and the leading candidate *changed at every window width*. The
system was presenting an arbitrary ordering with a confidence percentage
attached, while its own evidence section stated that no causal claim was
supported.

The report now refuses to name a "primary root cause" when no edge survives,
labels the leading metric as correlated rather than causal, and declares the
ranking arbitrary when candidates fall within a hundredth of each other.

### 5.3 The evaluation that has not been done

There is no ground truth, and consequently **no measurement of detection
precision or recall anywhere in this work**. Every claim about correctness is
a plausibility judgement. The practical remedy — not yet implemented — is
fault injection: run a known CPU, disk or memory stressor, confirm it is
detected and attributed. Until that exists, "is the root cause correct" has no
answer, only an absence of contradiction.

---

## 6. Defects worth recording

Two classes proved instructive.

### 6.1 Failures invisible from source

The RCA worker stored its analysis window as `self.start`, overwriting
`QThread.start` — the method that launches the thread — with a `pandas`
Timestamp. Clicking Run raised `TypeError: 'Timestamp' object is not
callable` inside the handler, so no worker was created and the progress bar
sat at 0% indefinitely. Training was unaffected because its worker takes no
such argument, which made the outage look specific to RCA.

Every timing measurement taken from source was correct and irrelevant: they
called the pipeline directly and bypassed the worker entirely. The bug was
found only after unhandled exceptions were routed to a log file — in a
windowed build there is no stdout, so the traceback had been going nowhere.

### 6.2 Fixes that were worse than the defect

Two are worth naming because both passed their tests.

**An erase command that destroyed its own stop signal.** `delete-all-data`
was extended to remove the whole data directory. The collector's `stop.flag`
lives *inside* that directory, so the erase retracted the stop request
microseconds after making it. The collector polls once per 30-second cycle,
never saw the flag, kept the database locked, and every 250 ms retry destroyed
the trained model and reports again — while the database, the actual privacy
concern, survived. Deletion is now gated on the database unlink succeeding,
which is proof the collector has exited.

**An incident filter that discarded every model detection.** Incidents whose
window could not be analysed were filtered out. A detector-flagged run can be
as short as three samples, and the filter judged it against a twelve-sample
model window, so *all* detector-triggered incidents were removed — leaving
only Windows Event Log faults. The incident list shrank from 26 to 9 and every
remaining entry succeeded, which resembled success. What had actually happened
is that 21 of the model's own findings were silently discarded. Short
incidents are now widened into surrounding context that was always present.

Both fixes were made confidently, passed their tests, and were caught by
adversarial review rather than by testing.

---

## 7. Limitations

1. **Collection coverage of 27.8%** with a median segment under nine minutes.
   Everything downstream inherits this. Training data are fragmented, many
   incident windows are unanalysable, and drift measurements swing.
2. **No ground truth and no fault-injection harness**, so detection quality is
   unquantified.
3. **Causal inference rarely produces an edge** at the window sizes real
   incidents present. The statistical gates are correct; the data are thin.
4. **`model_stale` conflates** a stale model with an old analysis window.
5. **Unsigned distribution, 1.5 GB**, with no update mechanism.
6. **Single-machine scope.** Nothing correlates across machines, by design.

---

## 8. Conclusion

The engineering is sound in the parts that were measured: collection is
cheap, training completes in under a minute, the packaging failures are
understood (with one honest exception), and the privacy claim — no network
code anywhere in the source — holds.

The analytical claims are weaker than the interface originally suggested, and
the substantive contribution of the later work was making the system say so.
A tool that reports "no causal chain was supported" when the data cannot
support one is more useful than a confident ranking that changes with the
window. The most valuable results here were negative, and the most valuable
changes were the ones that stopped the system overstating what it knew.

The clearest path forward is not a better model. It is collector supervision
to raise coverage above 90%, and a fault-injection harness so that
"correct" becomes a measurement rather than an impression.

---

## Appendix A · Reproducing the measurements

```powershell
python -m pytest tests/ -q          # 88 tests
.\packaging\build.ps1               # both executables, ~19 minutes
```

Coverage, segment statistics and readiness are computed by
`telemetry.analysis.baseline_status` and `contiguous_windows`. Runtime figures
come from `pipeline.engine.estimate_training_seconds` and
`estimate_rca_seconds`, whose constants were fitted to the measurements in
§3.3 and which recalibrate against each real training run on the host machine.

## Appendix B · Threats to validity

- All measurements come from **one machine** (Windows 11, NVIDIA GPU present,
  CPU-only training) over 13.2 days. Nothing here establishes generality.
- **Coverage is depressed by the development process itself** — the collector
  was repeatedly terminated by rebuilds. The architectural weakness is real,
  but 27.8% should not be read as the steady-state figure for an undisturbed
  installation.
- Timing constants were fitted on a machine **running a PyInstaller build
  concurrently**, so they lean pessimistic.
- The causal results in §5.2 come from a **single incident** widened
  progressively. The instability of the leading candidate is demonstrated, not
  its distribution.
