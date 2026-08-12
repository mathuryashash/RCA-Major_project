# LocalRCA: On-Device Root Cause Analysis for Windows Endpoints

**An implementation paper**

Version 1.2.0 · revised 2026-08-12

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
instrumented runs on a live installation, not from intended behaviour.

The revision of 2026-08-12 adds the first **fault-injection evaluation**, which
the original draft listed as the single most important missing piece. Five
controlled runs are reported. A 30-minute CPU burn was detected, correctly
attributed, and — for the first time in this project — **correctly explained**:
the ranking named `cpu_pct` first with six surviving causal edges pointing away
from CPU. A 30-minute disk burn was detected and correctly ranked but produced
**no causal chain at all**, for a reason that turned out to be a subsystem
topology constraint rather than a statistical one. A 30-minute memory hold was
detected and then **attributed to four processes that had nothing to do with
it**, exposing a defect that made memory-bound causes unnameable by
construction. A 30-minute idle run flagged 1 metric of 29.

Several of the most useful results remain negative, and they are reported as
such. The causal layer is now known to have been *starved* rather than broken —
a distinction established by running the same fault at two window widths — and
collection coverage remains the dominant limiting factor.

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

Constraint 2 is the most consequential. §8 describes the only workable
response found: manufacture ground truth by causing a fault deliberately.

---

## 2. System architecture

### 2.1 Process topology

Two processes, deliberately separated:

```
┌──────────────────────────┐            ┌───────────────────────────┐
│   RCA-Collector.exe      │            │      RCA-Desktop.exe      │
│   console, no window     │            │      PySide6 / Qt 6       │
│                          │            │                           │
│  ┌────────────────────┐  │            │  ┌─────────────────────┐  │
│  │ 30 s system sample │  │            │  │ Stage 1  training   │  │
│  │ 300 s process scan │  │  SQLite    │  │ Stage 2  inference  │  │
│  │ 300 s event poll   │──┼── WAL ────►│  │ reports & figures   │  │
│  └────────────────────┘  │◄───────────┤  └─────────────────────┘  │
│   singleton lock         │  read-only │   QThread workers         │
└──────────────────────────┘            └───────────────────────────┘
            ▲                                        │
            │ restarts                               │ consent, install,
   ┌────────┴─────────┐                              │ uninstall
   │  supervise.ps1   │                              ▼
   │  logon entry     │              %LOCALAPPDATA%\RCA\telemetry.db
   └──────────────────┘                             telemetry_model.pt
                                                    desktop.log
```

The split exists because collection must survive the GUI being closed. A
single-process design was tried first and abandoned: the desktop application
resolved its own executable path when asked to start collection, and in a
frozen build that path *is* the GUI, so it launched a second window, which
launched a third. The collector is now located by explicit binary name
(`RCA-Collector.exe`) and refuses to launch if the expected sibling is absent,
rather than falling back to the running executable.

### 2.2 Supervision

The collector is registered in the per-user **Startup folder**, not Task
Scheduler. `schtasks /SC ONLOGON` and the Task Scheduler COM API both fail with
access denied for a standard user — measured, not assumed — because creating a
logon-triggered task is an administrative operation. Demanding UAC elevation
for a tool that only reads the local machine is a poor trade.

The logon entry starts a generated PowerShell supervisor rather than the
collector directly, because a collector that died mid-session previously stayed
dead until the next logon:

```
for attempt in 0 .. 11:
    if exists(stop.flag):      break
    run collector, wait for exit
    if exists(stop.flag):      break
    if not exists(collector):  break        # uninstalled
    sleep( min(300, 15 × (attempt+1)) )     # seconds
```

The restart budget of 12 with that backoff spans a little over an hour: long
enough to ride out a transient fault, short enough that a genuinely broken
collector stops relaunching and leaves its reason in the log rather than
looping forever. Verified by killing the collector and observing the
supervisor replace it (PID 76356 → 47292) within 45 s.

### 2.3 Storage

One SQLite database in WAL mode at `%LOCALAPPDATA%\RCA\telemetry.db`, with a
`SCHEMA_VERSION` meta row and columns added defensively on open. WAL matters
here: the collector writes every 30 s while the GUI reads, and the default
rollback journal would block one against the other.

| Table | Contents | Retention |
|---|---|---|
| `metrics` | 29 numeric features per 30 s sample | unbounded |
| `processes` | top-15 by CPU per scan | 30 days |
| `events` | allowlisted Windows Event Log records | 365 days |
| `meta` | schema version, event watermarks, model reference error | — |

The retention asymmetry is deliberate — process detail is the most sensitive
data collected — and it is the direct cause of a defect in §10.

### 2.4 Threading

Every long operation runs in a `QThread` worker that emits Qt signals for
progress and completion. Training and RCA both take tens of seconds, and doing
either on the GUI thread freezes the window in a way Windows reports as "not
responding". The worker boundary is also where the project's worst bug lived
(§10.1).

---

## 3. Data collection

### 3.1 Streams

| Stream | Cadence | Content |
|---|---|---|
| System metrics | 30 s | 29 features (below) |
| Process samples | 300 s, 30 s under load | Top-15 by CPU: name, CPU %, RSS, I/O bytes |
| Windows events | 300 s poll | Allowlisted providers only |

The process cadence adapts: it tightens to 30 s while the machine is under
load, because a 5-minute scan is useless for attributing a 90-second spike.

### 3.2 The 29 features

| Subsystem | Features |
|---|---|
| CPU | `cpu_pct`, `cpu_pct_max_core`, `cpu_freq_mhz`, `cpu_freq_ratio` |
| Memory | `mem_pct`, `mem_available_mb`, `swap_pct`, `swap_used_bytes`, `swap_used_delta` |
| Disk | `disk_read_bps`, `disk_write_bps`, `disk_busy_pct`, `disk_free_pct` |
| Network | `net_sent_bps`, `net_recv_bps` |
| Power | `battery_pct`, `battery_drain_rate`, `power_plugged` |
| Process | `process_count` |
| GPU (NVML) | utilisation, memory, temperature where a supported device exists |

Counter-derived features (`*_bps`, `*_delta`, `battery_drain_rate`) are rates
computed against the previous sample, and are emitted as zero rather than a
spike across a collection gap — a gap of eight hours would otherwise produce a
meaningless "bytes per second" of enormous magnitude at the resume point.

### 3.3 Event allowlist

Only these families are retained: Kernel-Power 41 (unexpected shutdown),
Application Error 1000, Application Hang 1002, disk 7/51/153, WHEA hardware
errors, Resource-Exhaustion 2004, Windows Update, and MsiInstaller. The
collector advances its read watermark across *all* events and discards
non-allowlisted records before storage, so the watermark stays correct without
retaining what was skipped.

### 3.4 Privacy posture

Event message text is stored only under an explicit opt-in, and is redacted
first for user paths, UNC paths, URLs, email addresses and the username.
Window titles, keystrokes, clipboard and file contents are never captured.
There is no network code anywhere in the source; the privacy claim is
structural rather than policy-based.

A consent dialog on first launch states what is recorded before any collection
begins, and declining collects nothing.

---

## 4. Method

This section states the actual computation. Symbols: $n$ metrics ($n = 29$),
$W$ window length in samples, $T$ timesteps in a window ($T = W$), $L$ maximum
Granger lag.

### 4.1 Scaling

A `MinMaxScaler` is fitted **on the clean baseline only**:

$$x'_{j} = \frac{x_j - \min_j}{\max_j - \min_j}$$

The incident window is transformed with the *baseline's* parameters and then
clipped:

$$x'_{\text{incident}} = \operatorname{clip}\big(\text{scaler}(x), 0, 1\big)$$

Fitting on the incident would normalise away the very deviation being looked
for. Clipping is what allows an incident value outside the baseline range to
saturate at 1.0 instead of producing a scaled value of 4.7 that the model has
never seen.

### 4.2 Segmentation and windowing

Training windows are built **within contiguous segments only**, never spanning
a collection gap. A segment is a maximal run of samples whose consecutive
spacing is within tolerance of the 30 s cadence.

For each segment $S$ with $|S| \ge W$, windows are taken at stride 5 for
training and stride 1 for detection:

$$\mathcal{W}(S) = \{\, S[i : i+W] \;:\; i = 0, 5, 10, \dots \,\}$$

This was not the original design. Training initially used the single longest
uninterrupted run, which discarded most of the collected history — on a laptop
that sleeps, one run long enough to satisfy the window requirement may never
occur. Accumulating windows across all clean segments raised usable training
data substantially on the same database.

A segment is *clean* if it contains no anomalous event and no gap.

### 4.3 The LSTM autoencoder

```
x ∈ ℝ^{B×T×29}
   │
   ├─ encoder   LSTM(29 → 64, layers=2, dropout=0.2)     ─► h_T ∈ ℝ^{B×64}
   ├─ bottleneck Linear(64 → 32)                          ─► z   ∈ ℝ^{B×32}
   ├─ expand    Linear(32 → 64), repeat T times           ─► ℝ^{B×T×64}
   ├─ decoder   LSTM(64 → 64, layers=2, dropout=0.2)      ─► ℝ^{B×T×64}
   └─ output    Linear(64 → 29)                           ─► x̂ ∈ ℝ^{B×T×29}
```

Only the final hidden state of the encoder reaches the bottleneck, so the
whole window is compressed to 32 numbers before reconstruction. That
compression is the entire mechanism: a window resembling training data
survives the bottleneck; an unfamiliar one does not.

Training objective, over all elements of the batch:

$$\mathcal{L} = \frac{1}{B\,T\,n}\sum_{b,t,j} \big(x_{btj} - \hat{x}_{btj}\big)^2$$

Optimiser Adam, $\text{lr} = 10^{-3}$, batch size 32, validation split 0.2
taken as the **tail** of the window sequence. The checkpoint with the lowest
validation loss is kept and reloaded before threshold calibration, so
calibration never runs against an over-fitted final epoch.

Torch is capped at 4 threads. Measured, the default of one thread per core was
**3.9× slower** at 20 threads than at 4: the per-op work is small and the
LSTM's sequential timesteps limit real parallelism, so dispatch dominates.

### 4.4 Per-metric anomaly score

Reconstruction error is reduced over time but **not** over features, which is
what makes the output attributable to a metric rather than to a window:

$$s_j(\text{window}) = \frac{1}{T}\sum_{t=1}^{T}\big(x_{tj} - \hat{x}_{tj}\big)^2$$

### 4.5 Threshold calibration

Per-metric thresholds are the 99th percentile of validation-window error:

$$\tau_j = P_{99}\big(\{\, s_j(w) : w \in \text{validation} \,\}\big)$$

A window is anomalous in metric $j$ when $s_j > \tau_j$. The reported score is
normalised so that 1.0 is exactly the threshold:

$$\tilde{s}_j = \frac{s_j}{\tau_j + 10^{-8}}$$

and a metric's incident-level score is $\max$ over the incident's windows.
Its first-seen time is the timestamp of the first window that crossed.

Per-metric rather than global thresholds matter because the metrics have
wildly different reconstructability: `power_plugged` is nearly constant and
reconstructs almost perfectly, while `disk_read_bps` is bursty by nature. One
global threshold would flag disk permanently and power never.

### 4.6 Drift and staleness

The median reconstruction error at training time is stored in the model
artifact as $e_{\text{ref}}$. At each analysis:

$$\rho = \frac{\operatorname{median}_j e_j(\text{window})}{e_{\text{ref}}},
\qquad \text{stale} \iff \rho > 2.0$$

Observed: a model trained on data through 30 July scored **57×** its reference
error against later data; retraining on 4,608 fresh clean samples brought
$\rho$ to **1.54×**.

**Known defect:** $\rho$ is computed against whichever window is being
analysed, so analysing a *historical* incident with a freshly trained model
reports the model as stale. That is a property of the window, not the model,
and the current report conflates them.

### 4.7 Incident detection

Two triggers produce candidate incidents:

- **Detector** — a run of $\ge 3$ consecutive windows with any metric above
  threshold.
- **Event** — an allowlisted Windows event, windowed with surrounding context.

Overlapping candidates are **merged first and filtered second**. The order is
not cosmetic: filtering first discarded every detector incident, because a
3-sample run was being judged against a 12-sample model window (§10.2). Short
incidents are now widened into the surrounding context, which was always
present in the database.

When analysing, the **largest** contiguous segment inside the window is used,
not the last. For an event-triggered incident the interesting data precedes the
event, so taking the trailing fragment after a sleep gap would analyse the
wrong side of the crash.

### 4.8 Stationarity

Granger causality assumes stationary series. Each series is differenced until
an Augmented Dickey–Fuller test rejects a unit root, to a maximum of two
rounds:

```
for round in 1..2:
    if adfuller(series).pvalue < 0.05:  break     # already stationary
    series ← series.diff()
```

Two rounds is a deliberate ceiling. A series still non-stationary after two
differences is usually a counter or a step change, and differencing it further
destroys the signal faster than it removes the trend.

### 4.9 Granger causality

For every ordered pair $(c, e)$ of *anomalous* metrics — restricting to
anomalous metrics is what keeps the pair count tractable — the two stationary
series are inner-joined on timestamp, and the test is skipped unless

$$|\text{aligned}| \ \ge\ 3L$$

For lags $\ell = 1 \dots L$, the SSR $F$-test compares the restricted and
unrestricted models:

$$F(\ell) = \frac{(\mathrm{SSR}_r - \mathrm{SSR}_u)/\ell}{\mathrm{SSR}_u/(N - 2\ell - 1)}$$

where the restricted model regresses $e_t$ on its own lags alone and the
unrestricted model adds $\ell$ lags of $c$. The lag with the smallest $p$-value
is selected:

$$\ell^* = \arg\min_{\ell} \; p(\ell)$$

**Effective sample floor.** Differencing consumes up to two observations, so
the practical minimum for any pair to be compared at all is

$$N_{\min} = 3L + 2$$

At the default $L = 5$ that is **17 samples**, or 8.5 minutes of collection.
This single inequality explains most of the project's negative results.

### 4.10 Multiple-testing correction

With $m$ anomalous metrics there are $m(m-1)$ ordered pairs, and a laptop
exposes many correlated resource metrics, so uncorrected $p$-values would
manufacture causality. Benjamini–Hochberg controls the false discovery rate:
sort the $p$-values ascending, find the largest rank $k$ satisfying

$$p_{(k)} \ \le\ \frac{\alpha\,k}{m_{\text{tests}}}, \qquad \alpha = 0.05$$

and accept ranks $1 \dots k$.

### 4.11 Effect-size floor

A $p$-value alone is insufficient: with a long local history, negligible
improvements become "significant". The $F$-statistic is converted into a
bounded, transparent effect proxy:

$$\text{effect} = \frac{F}{F + N}$$

Pairs below $\text{effect} < 0.10$ are dropped regardless of significance. The
quantity is monotone in $F$, lies in $(0,1)$, and shrinks as the sample grows —
which is exactly the correction wanted, since large $N$ is what made the
$p$-value small in the first place.

**The FDR correction and this floor are the honest core of the design, and
also the reason the system frequently reports nothing.**

### 4.12 Graph construction and cycle breaking

Accepted pairs become directed edges carrying `strength`, `p_value` and `lag`.
Nodes carry `anomaly_score`. Cycles are then removed to obtain a DAG,
preferring the edge that contradicts observation:

1. If $t_{\text{first}}(u) > t_{\text{first}}(v)$, the edge $u \to v$ claims a
   cause that appeared *after* its effect — remove it.
2. Otherwise remove the weakest edge in the cycle by `strength`.

### 4.13 Topology constraint

A statistical edge is kept only if the operating system permits propagation in
that direction. The subsystem map is explicit and inspectable rather than
learned:

```
power  → cpu, memory, disk
cpu    → memory, disk, network
memory → disk, network
disk   → network
process→ cpu, memory, disk, network
```

An edge $a \to b$ survives if `subsystem(a) == subsystem(b)` or a directed path
exists between their subsystems. Metrics with no declared subsystem are
rejected rather than trusted.

This map is **asymmetric by construction**, and §8.3 reports the first measured
case where that asymmetry discarded the only surviving edge in a run.

### 4.14 Ranking

PageRank is computed on the **reversed** graph, weighted by edge strength, so
a high score means "influential source" rather than "popular sink":

$$\text{PR} = \operatorname{pagerank}\big(G^{\mathsf{T}}, \text{weight}=\text{strength}\big)$$

Five normalised components are combined:

| Component | Definition | Weight |
|---|---|---|
| Causal outflow | $\deg^{+}(v) / \max_u \deg^{+}(u)$ | 0.40 |
| Temporal priority | $1 - \dfrac{t_v - t_{\min}}{t_{\max} - t_{\min}}$ | 0.30 |
| Causal inflow | $1 - \deg^{-}(v) / \max_u \deg^{-}(u)$ | 0.20 |
| Anomaly severity | $\tilde{s}_v$ | 0.05 |
| Event correlation | best $1/(1+\Delta_{\text{hours}})$ | 0.05 |

$$\text{score}(v) = \min\!\Big(1.0,\ 0.70 \sum_k w_k\, s_k(v) \;+\; 0.30\,\text{PR}(v)\Big)$$

Confidence bands: $\ge 0.95$ Critical, $\ge 0.85$ High, $\ge 0.70$ Medium,
$\ge 0.50$ Low, else Very Low.

**The failure mode of this formula is structural and important.** With zero
edges, every node has $\deg^{+} = 0$ and $\deg^{-} = 0$, so causal outflow is 0
and causal inflow is 1 for every metric — 60% of the weight becomes a constant.
The ranking collapses onto temporal priority and severity alone, while still
printing a score and a confidence band. §8 shows this happening on real data.

### 4.15 Event correlation

For each anomalous metric and each event preceding it by
$0 < \Delta \le 24$ hours:

$$\text{correlation} = \frac{1}{1 + \Delta_{\text{hours}}}$$

Strictly $\Delta > 0$: an event at or after the anomaly cannot have caused it.

### 4.16 Process attribution

Process snapshots from 15 minutes before the first anomaly to the end of the
window are aggregated by executable name into sample count, mean CPU %, peak
RSS and total I/O bytes, ranked by mean CPU. This is descriptive, not causal —
it says what was running, not what was responsible.

---

## 5. End-to-end workflow

### 5.1 Stage 1 — training

```
 baseline_readiness(db)
        │  needs ≥ 21 h of clean samples at W=12
        ▼
 load clean segments  ──►  build windows per segment (stride 5)
        │                          │
        ▼                          ▼
 fit MinMaxScaler on baseline   concatenate across segments
        │                          │
        └──────────► train LSTM AE (Adam, MSE, best-val checkpoint)
                              │
                              ▼
                     calibrate τ_j = P99(validation error)
                              │
                              ▼
             save artifact: state_dict, τ, scaler params,
                            feature order, window size,
                            reference error, trained_at
```

The artifact is a single reloadable bundle. Saving the state dict alone was
insufficient — thresholds calibrated against a different scaler are
meaningless, and feature order silently reindexing would produce confident
nonsense.

### 5.2 Stage 2 — inference

```
 detect_incidents(db, model)          ──► merged, widened candidate list
        │
        │  user selects one (or a custom range)
        ▼
 [10%]  load model artifact
 [25%]  load telemetry for window
 [40%]  validate: largest contiguous segment ≥ W
 [55%]  scale (baseline params, clipped) → score → flag metrics
        │
        ├── no metric flagged ──► report "nothing anomalous", stop
        ▼
 [70%]  stationarise → Granger over anomalous pairs → BH → effect floor
        │              → graph → break cycles → topology prune
 [85%]  PageRank → composite ranking → process attribution
        ▼
 [100%] report: Markdown + JSON + timeline figure + causal graph figure
```

Progress percentages are emitted per stage because causal inference dominates
the runtime, and without them a long analysis is indistinguishable from a hung
one — which is precisely how the total outage in §10.1 presented.

### 5.3 What the report refuses to say

Four distinct outcomes are reported differently, and collapsing them was the
substantive dishonesty in the original system:

| Condition | Reported as |
|---|---|
| Edges survived | "supported" — causal chain stated |
| Pairs accepted, then topology-pruned | "no causal claim is made"; states that the statistics found something the map forbids |
| Pairs tested, none survived the gates | "no supported causal chain"; metrics labelled correlated |
| $N < 3L + 2$, nothing compared | "causality was **not tested**"; ranking explicitly carries no causal evidence |

The last two were previously identical output, which is the difference between
a negative result and no result at all.

---

## 6. Cost model

### 6.1 Training

Measured on 1,701 windows from 8,516 clean samples:

| Epochs | Window | Time |
|---|---|---|
| 5 | 12 | 8.5 s |
| 20 | 12 | 17.5 s |
| 30 | 12 | 24.6 s |
| 5 | 60 | 13.2 s |
| 30 | 60 | ~64 s |

Per-epoch cost is linear in window length (an LSTM traverses every timestep)
and linear in window count:

$$t_{\text{epoch}}(W, N_w) = \big(0.40 + 0.0308\,W\big)\cdot\frac{N_w}{1701}
\quad\text{seconds}$$

$$t_{\text{train}} = 2.1 + 5.0_{\text{cold}} + E \cdot t_{\text{epoch}}$$

Held-out points land within 0.05 s. The constants recalibrate against each real
training run on the host machine, so the estimate improves with use rather than
staying fixed to the development machine.

**The first epoch costs ~4.6 s against ~0.5 s for subsequent ones.** PyTorch
imports Dynamo lazily through Adam's constructor on first use. This is
invisible in aggregate timings and was found only by instrumenting per-epoch
boundaries; it is carried as an explicit cold-start term rather than smeared
across the per-epoch rate.

### 6.2 Inference

| Samples | Anomalous metrics | Time |
|---|---|---|
| 104 | 5 | 0.8 s |
| 464 | 6 | 1.4 s |
| 1,334 | 10 | 12.5 s |

$$t_{\text{rca}} = 0.7 + 6.7\times10^{-6}\,N^2$$

Cost grows with the *square* of the sample count: Granger tests every ordered
pair, and the anomalous-metric count itself rises with window width, so both
factors grow together.

### 6.3 Collector overhead

| Property | Measured |
|---|---|
| CPU | 0.78% of total (28 logical cores ≈ 1/5 of one core) |
| RAM | 27 MB |
| Database growth | ~2.2 MB/day |
| Cold first launch | 52 s |
| Warm launch | 9.5 s |

RAM and disk are comfortable. The CPU figure is higher than a background
sampler should need and is flagged for a proper profile — the 90-second
measurement window may have caught a process-sampling burst rather than steady
state.

### 6.4 What actually binds

**Training compute is not a limiting factor.** Even maximal settings complete
in about a minute. The binding constraint is the ~21 hours of clean collection
needed before training unlocks at all — which, at the coverage measured in §7,
is several days of wall-clock time.

---

## 7. Dataset and coverage

Measured on a live installation, 2026-08-11:

| Property | Value |
|---|---|
| Observation span | 13.2 days |
| System samples | 10,555 |
| Process samples | 243,089 |
| Event records | 4,194 |
| Features per sample | 29 |
| Database size | 28.9 MB |
| Clean samples available for training | 8,721 |

### 7.1 Coverage — the dominant finding

| Property | Value |
|---|---|
| **Coverage against continuous sampling** | **27.8%** |
| Contiguous segments | 46 |
| Longest unbroken segment | 14.4 hours |
| **Median segment length** | **17 samples (8.5 minutes)** |

At 30-second cadence, 13.2 days of continuous collection would yield ~38,000
samples; 10,555 were recorded. **The collector ran roughly a quarter of the
time, in fragments whose median length is under nine minutes.**

Note what the median coincides with: $N_{\min} = 3L + 2 = 17$ samples at the
default lag. The median segment sits exactly on the floor below which no
Granger pair is compared at all. Coverage and causal yield are not two problems
but one.

Part of this is measurement artefact — the collector was repeatedly terminated
during development builds — and the supervisor of §2.2 postdates the
measurement. **The figure cannot be compared against a supervised installation
and should not be read as the steady-state number.** Re-measuring over a fresh
multi-day window is the outstanding item.

---

## 8. Evaluation

The original draft's §5.3 stated that no measurement of detection quality
existed anywhere in the work, and named fault injection as the remedy. That
harness now exists as `tools/evaluate_detection.py`: it causes a known
disturbance, waits for the samples to land, runs the real pipeline over the
injection window, and scores what came back.

It manufactures the ground truth that a personal machine cannot otherwise
provide. It is not a unit test — it needs the collector running and takes tens
of minutes, because it must wait for real samples at the real cadence.

### 8.1 Results

| Run | Samples | Flagged | Accepted pairs | Edges | Attributed | Verdict |
|---|---|---|---|---|---|---|
| CPU burn, 7 min | 14 | 6 of 29 | **0 — never tested** | 0 | yes | PASS (detection only) |
| CPU burn, 30 min | 60 | 6 of 29 | 10 | **6** | yes | **PASS, explained** |
| Disk burn, 30 min | 60 | 4 of 29 | 1 | **0 (pruned)** | yes | PASS, unexplained |
| Memory hold, 30 min | 60 | 2 of 29 | 0 | 0 | **no** | **FAIL — wrong culprit** |
| Idle, 30 min | 60 | 1 of 29 | — | — | — | 3.4% false positive |

Each run found something the previous ones could not. Read in order they
form a rough ladder of what the system can and cannot do: detect (CPU, 7 min),
explain (CPU, 30 min), decline to explain honestly (disk), and — with memory —
detect the right thing while naming the wrong cause.

### 8.2 CPU: the first end-to-end explanation

Half the cores were burned for 30 minutes. The pipeline was told nothing.

```
1. cpu_pct            score 1.000   Critical
2. cpu_pct_max_core   score 0.916   High
3. swap_used_delta    score 0.719   Medium
```

The causal directions point away from CPU, which is what a genuine root cause
looks like — nothing upstream drives it:

```
cpu_pct_max_core → disk_busy_pct    lag=2   strength 0.756
cpu_pct_max_core → swap_used_delta  lag=3   strength 0.262
swap_used_delta  → disk_busy_pct    lag=2   strength 0.436
```

**The comparison against the 7-minute run is the result that matters.**
Identical fault, identical code, only duration differed. At 14 samples, against
$N_{\min} = 17$, *no pair was compared at all* and the system said so. At 60
samples, 10 pairs survived correction and 6 edges survived topology.

**The causal layer was never broken. It was starved.** That reframes the
original draft's central negative result: the remedy is operational — analyse
wider windows — not architectural. It also validates the honesty work, since
the 7-minute run reported "not tested, window too short" rather than
manufacturing a ranking from an empty graph.

### 8.3 Disk: detected, correctly ranked, and honestly unexplained

Sustained ~100 MB passes with `fsync` for 30 minutes. Detection succeeded —
`disk_write_bps` and `disk_busy_pct` both flagged, load attributed to
`python.exe` — and the ranking put the injected metric first:

```
1. disk_write_bps     score 1.000   Critical
2. net_sent_bps       score 0.818   Medium
3. cpu_pct_max_core   score 0.331   Very Low
4. disk_busy_pct      score 0.266   Very Low
```

**But the causal graph was empty, and the ranking above therefore carries no
causal evidence.** Exactly one pair passed both statistical gates:

```
net_sent_bps → cpu_pct_max_core     p=0.0033   lag=1   strength 0.136
```

and the subsystem map of §4.13 has no path from network to CPU, so it was
pruned. The graph emptied, the composite score collapsed to temporal priority
and severity as §4.14 predicts, and `disk_write_bps` reached 1.000 "Critical"
**on severity alone**.

That the top-ranked metric is also the injected fault is the right answer
arrived at by a route the system cannot claim as causal. Recording it as a
success would be precisely the overstatement this project has spent its later
half removing.

The run also exposed a **reporting defect, now fixed**: the pruned pair was
described as "no edge survived multiple-testing correction and the effect-size
floor", which blames the statistics for a decision the topology made. The two
are now distinguished, and the report states plainly that the statistics found
something the map does not permit — meaning either the relationship is
spurious, or the map is incomplete. Both remain open. A network→CPU path is
physically defensible via interrupt handling; the map does not include one.

### 8.4 Memory: detected, and attributed to four innocents

A process held 1.15 GB — bounded deliberately at half of free memory — for
30 minutes, sleeping between allocations. Memory moved from 84% to 93.3% used.

Detection worked, if narrowly: `swap_used_delta` and `disk_free_pct` flagged,
2 of 29. Notably **`mem_pct` itself did not flag**, which is discussed below.
Then attribution failed outright:

```
top processes : SearchIndexer.exe, WmiPrvSE.exe, Taskmgr.exe, MsMpEng.exe, System
ATTRIBUTED to us : no
```

Four innocent processes named; the one holding 1.15 GB absent entirely.

**The cause was structural, not statistical.** `load_process_attribution`
ordered by `avg_cpu_pct`, then `io_bytes`. `max_rss_bytes` was *selected and
never sorted on*. A process that allocates and sleeps has no CPU and no I/O, so
no quantity of held memory could place it in the top ten — **a memory-bound
cause was unreachable by construction**, in every incident this system has ever
analysed. Attribution now takes the union of the CPU-heaviest and the
RSS-heaviest, half the slots each, which is the rule `ProcessSampler` already
applies one layer down. Re-run against the recorded window, `python.exe` appears
at 1,537 MB peak RSS.

**The harness was wrong in the more dangerous direction.** It printed
`ATTRIBUTED to us: no` and returned `PASS`, because only detection gated the
verdict. The production checklist carried "assert the correct process is
attributed" as satisfied, on the strength of a check that never ran. Attribution
now fails the run: knowing that something is wrong without knowing what did it
is half a diagnosis.

Two facts about this result are worth separating. The defect was real, sat in
production, and had been invisible for the project's entire life — the CPU and
disk faults are both CPU-and-I/O-heavy, so they attributed correctly and
concealed it. And the defect was found only because the fault type was chosen
for being *different in kind*, not for being likely to pass.

That `mem_pct` did not flag is a second, unresolved observation. The injection
moved total memory use by roughly 9 percentage points; the autoencoder found
that unremarkable, while it did flag the swap movement that followed. The most
likely explanation is that this machine's baseline already spans a wide range of
memory pressure, so 93% is within learned normal — which would mean the model is
correct and the fault simply was not large enough to be abnormal *for this host*.
That is a hypothesis, not a finding; it has not been tested.

### 8.5 Idle: the false-positive floor

Thirty minutes with nothing injected flagged **1 metric of 29 (3.4%)**,
`mem_available_mb`, with Windows Search indexing visible in the process
attribution. A Windows machine is never truly idle, and flagging genuine
background load is correct behaviour rather than a false positive in the strict
sense.

Read against the injected runs — 6 of 29 under CPU load, 4 of 29 under disk
load, 2 of 29 under memory load, 1 of 29 at rest — the detector
**discriminates** rather than firing at everything. The harness tolerance is
set at 7.0%, just above the measured floor; tightening it further would fail on
real OS activity.

### 8.6 What this evaluation still does not establish

Five runs on **one machine**. The following remain undone and are not claimed:

- **A fault whose cause is not the loudest metric.** Both explaining runs put
  the injected metric at the top of a severity ranking. Neither could have
  distinguished a correct causal answer from a correct *severity* answer.
- **Repeats on other hardware.** Every number here is from one host.
- **Causal yield across many incidents**, rather than the two reported here.
- **Re-running memory with attribution fixed.** The fix is verified against the
  recorded window, which is not the same as a fresh end-to-end pass.
- **Why `mem_pct` did not flag** at 93% memory use.

The honest claim is **"demonstrated once"**, not "validated".

### 8.7 What the harness is actually for

The evaluation's most valuable output has not been a pass. Of five runs, one
explained a fault end to end, one showed the causal layer being starved rather
than broken, one caught the report blaming statistics for a topology decision,
and one caught a production defect that made an entire class of root cause
unnameable — plus a defect in the harness's own scoring.

The pattern is consistent enough to state as a working rule: **the runs that
found defects were the ones chosen for being different in kind from the runs
before them.** A second CPU fault would have passed and taught nothing. This
argues for selecting future faults by what they would exercise that nothing has
exercised yet, rather than by what seems likely to succeed.

---

## 9. Distribution

PyInstaller `--onedir`, two executables, unsigned.

| Property | Value |
|---|---|
| Installed size | 1,110 MB (from 1,538 MB) |
| Release ZIP | 433.5 MB |
| SHA256 | `AE73A5A5175507B2A4FC4E5069181B6D8DAAFFED6A52A2D1EC3CAE66D029FA45` |
| Install | extract → run → agree |

Registration is per-user and needs no elevation: a Startup-folder logon entry,
a Start menu shortcut (so Windows search can find the app), and an
Add/Remove Programs entry under `HKCU`. A ZIP that runs itself at every logon
with no entry where users look to remove things is indistinguishable from
something unwanted.

Notable packaging results:

- **A pip shadow copy inflated the build by ~400 MB.** An interrupted install
  had left `~orch` directories beside `torch`; PyInstaller walked them as real
  packages. The hook now filters `~`-prefixed paths.
- **`optree` was excluded but its metadata was not.** PyTorch treats optree as
  optional and decides availability via `importlib.metadata.version`. The
  frozen build shipped `optree-0.18.0.dist-info` *without* the package, so
  PyTorch read a version, concluded it was present, imported it, and failed
  inside Adam's constructor. Training worked from source and failed only when
  packaged. Excluding a lazily-imported dependency while shipping its metadata
  is a general hazard of static-closure exclusion lists.
- **Excludes that break the thing they trim.** `torch.export` and
  `torch._inductor` were excluded as tracing-only. Both are imported at module
  scope — by `torch/__init__.py` and `torch._dynamo.guards` respectively — so
  the exclusions would have broken `import torch` outright had they taken
  effect.
- **A windowed build starts with no valid stdout/stderr.** The packaged
  application terminated with `0xC0000409` in `Qt6Core.dll` roughly forty
  seconds after launch, with no diagnostic anywhere. It survives when file
  descriptors 1 and 2 are pointed at the null device before Qt is imported.
  The mechanism remains **unestablished**: it is *not* PySide6 escalating a
  slot exception to `qFatal`, which was measured and disproved. The fix is
  validated by outcome, not by mechanism, and the code says so.

Licensing: MIT, with LGPL obligations for bundled Qt/PySide6 documented in
`THIRD-PARTY-NOTICES.md`.

---

## 10. Defects worth recording

### 10.1 Failures invisible from source

The RCA worker stored its analysis window as `self.start`, overwriting
`QThread.start` — the method that launches the thread — with a `pandas`
Timestamp. Clicking Run raised `TypeError: 'Timestamp' object is not callable`
inside the handler, so no worker was created and the progress bar sat at 0%
indefinitely. Training was unaffected because its worker takes no such
argument, which made the outage look specific to RCA.

Every timing measurement taken from source was correct and irrelevant: they
called the pipeline directly and bypassed the worker entirely. The bug was
found only after unhandled exceptions were routed to a log file — in a windowed
build there is no stdout, so the traceback had been going nowhere.

### 10.2 Fixes that were worse than the defect

Three are worth naming because all three passed their tests.

**An erase command that destroyed its own stop signal.** `delete-all-data` was
extended to remove the whole data directory. The collector's `stop.flag` lives
*inside* that directory, so the erase retracted the stop request microseconds
after making it. The collector polls once per 30-second cycle, never saw the
flag, kept the database locked, and every 250 ms retry destroyed the trained
model and reports again — while the database, the actual privacy concern,
survived. Deletion is now gated on the database unlink succeeding, which is
proof the collector has exited.

**An incident filter that discarded every model detection.** Incidents whose
window could not be analysed were filtered out. A detector-flagged run can be
as short as three samples, and the filter judged it against a twelve-sample
model window, so *all* detector-triggered incidents were removed — leaving only
Windows Event Log faults. The incident list shrank from 26 to 9 and every
remaining entry succeeded, which resembled success. What had actually happened
is that 21 of the model's own findings were silently discarded.

**A shared log file across two processes.** The GUI was given the collector's
log. Measured: 6 records written, 3 survived. Windows file handles do not
interleave that way. The GUI now writes `desktop.log`.

All three were made confidently, passed their tests, and were caught by
adversarial review rather than by testing.

### 10.3 A harness that endangered its host

The memory fault allocated until `MemoryError`. On a machine with 0.3 GB free
of 15.7 GB that means forcing the session into swap, freezing the desktop, and
possibly taking other applications down. A tool for diagnosing a machine must
not be able to fell it. It is now bounded to half of available memory, capped
at 2 GB.

---

## 11. Limitations

1. **Collection coverage of 27.8%** with a median segment of 17 samples —
   exactly the Granger floor. Everything downstream inherits this. The figure
   predates supervision and has not been re-measured.
2. **Evaluation is five runs on one machine.** No fault has been tested whose
   cause is not also the most severe metric, so a correct causal answer and a
   correct severity answer are not yet distinguishable.
3. **Causal inference frequently produces no edge** at the window sizes real
   incidents present. The statistical gates are correct; the data are thin.
4. **The subsystem topology is hand-written and asymmetric**, and has been
   measured discarding the only surviving edge in a run. Whether the map is
   incomplete or the edge was spurious is unresolved.
5. **`model_stale` conflates** a stale model with an old analysis window.
6. **Unsigned distribution, 1,110 MB**, with no update mechanism and no path
   for a crash report to reach the developer — the latter arguably
   irreconcilable with the no-egress promise.
7. **Single-machine scope.** Nothing correlates across machines, by design.
8. **Untested at non-100% DPI, on small screens, and with a screen reader.**

---

## 12. Conclusion

The engineering is sound in the parts that were measured: collection is cheap,
training completes in under a minute, the collector is now supervised and
demonstrably restarts, the packaging failures are understood with one honest
exception, and the privacy claim — no network code anywhere in the source —
holds structurally.

The analytical claims were originally weaker than the interface suggested, and
the substantive contribution of the later work was making the system say so.
That work now has evidence behind it rather than only an argument. A 30-minute
CPU burn was detected, attributed, and explained, with six causal edges
pointing away from the injected cause. The same fault at 7 minutes tested
nothing and reported exactly that. **The causal layer was starved, not
broken**, and knowing which of those it was changes the remedy from redesign to
window width.

**The runs that failed were worth more than the one that succeeded.** The disk
run got the right answer for a reason it was not entitled to claim, and the
report said so — exposing, in the process, that a topology decision was being
reported as a statistical one. The memory run went further: it detected the
fault and named four innocent processes, because attribution ranked by CPU and
a sleeping allocator has none. That defect had been in production for the
project's entire life and was invisible to every prior test, because the CPU
and disk faults are both CPU-heavy and attributed correctly by accident of
their shape. The harness scored that run PASS, which was a second defect, and
the checklist had been claiming attribution was verified on the strength of it.

The lesson generalises past this project. Four of the five runs were chosen to
be unlike their predecessors, and those are the four that found something. A
test suite that only exercises the shape of fault it was designed against
measures its own assumptions.

One success is not a measurement. The remaining work is not a better model: it
is collector supervision measured over a fresh multi-day window, a fault whose
cause is not the loudest signal, and repeats on hardware that is not this
laptop.

---

## Appendix A · Reproducing the measurements

```powershell
python -m pytest tests -q                 # unit and integration suite
.\packaging\build.ps1                     # both executables, ~19 minutes
.\packaging\make_release.ps1              # ZIP + SHA256

# Fault injection (needs the collector running; takes as long as it says)
python tools\evaluate_detection.py --fault cpu    --minutes 30
python tools\evaluate_detection.py --fault disk   --minutes 30
python tools\evaluate_detection.py --fault memory --minutes 30
python tools\evaluate_detection.py --fault idle   --minutes 30
```

Coverage, segment statistics and readiness come from
`telemetry.analysis.baseline_status` and `contiguous_windows`. Runtime figures
come from `pipeline.engine.estimate_training_seconds` and `estimate_rca_seconds`,
whose constants were fitted to §6 and which recalibrate against each real
training run on the host machine.

## Appendix B · Symbols

| Symbol | Meaning | Default |
|---|---|---|
| $n$ | metrics per sample | 29 |
| $W$, $T$ | window length in samples / timesteps | 12 |
| $L$ | maximum Granger lag | 5 |
| $N_{\min}$ | samples needed for any Granger pair, $3L+2$ | 17 |
| $\tau_j$ | per-metric threshold, $P_{99}$ of validation error | — |
| $\tilde{s}_j$ | normalised anomaly score, 1.0 = threshold | — |
| $\rho$ | drift ratio; stale above 2.0 | — |
| $\alpha$ | FDR level | 0.05 |
| — | effect-size floor, $F/(F+N)$ | 0.10 |

## Appendix C · Threats to validity

- All measurements come from **one machine** (Windows 11, 28 logical cores,
  NVIDIA GPU present, CPU-only training). Nothing here establishes generality.
- **Coverage is depressed by the development process itself** — the collector
  was repeatedly terminated by rebuilds. The architectural weakness was real
  and is now addressed, but 27.8% is not the steady-state figure for an
  undisturbed installation, and the post-supervision figure is unmeasured.
- Timing constants were fitted while a **PyInstaller build ran concurrently**,
  so they lean pessimistic.
- The fault-injection runs are **one repetition each**. No variance is
  reported because none was measured.
- In both explaining runs the injected fault was also the highest-severity
  metric, so the evaluation **cannot separate** a correct causal ranking from a
  correct severity ranking. §8.6 states this as the most important gap.
- The attribution fix is verified by **re-running the analysis over the stored
  incident window**, not by a fresh injection. Nothing here shows it holding
  end to end on a live fault.
- Two of the five runs were scored by a harness now known to have been
  **passing runs that failed attribution**. Their detection results stand;
  their attribution results were re-checked by hand rather than by the harness
  that originally reported them.
- The idle false-positive rate was measured on a machine running Windows
  Search indexing. A quieter host would likely score lower, and a busier one
  higher.
