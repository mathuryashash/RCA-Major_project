# LocalRCA: On-Device Root Cause Analysis for Windows Endpoints

**An implementation paper**

Version 1.5.0 · revised 2026-08-22

---

## Abstract

LocalRCA is a Windows desktop application that continuously records system
telemetry on a single machine, learns that machine's normal behaviour with an
LSTM autoencoder, and — after an incident — attempts to explain what went
wrong using constrained Granger causality over the anomalous metrics. It runs
entirely on the endpoint: no telemetry leaves the machine, and the collector
opens no sockets under any configuration. The desktop application carries one
opt-in exception, off by default -- a check for a newer release that reads a
version number and sends nothing (§3.4).

This paper documents what was built, what was measured, and where the
measurements contradict the design's assumptions. Every figure comes from
instrumented runs on a live installation. Where a result is unflattering, or
where a claim made earlier in this document turned out to be wrong, both are
recorded in place rather than revised away.

**Evaluation is in two parts.** Seven controlled fault injections establish
whether the system detects, attributes and explains a disturbance whose cause
is known in advance. A population survey then runs the full pipeline over
**175 incidents discovered in real collected history**, of which 92 are long
enough to test, converting claims that rested on single runs into rates.

The headline results:

| Question | Answer | Evidence |
|---|---|---|
| Does it detect a known fault? | yes, for CPU and disk | §8.2, §8.3 |
| Does it name the responsible process? | yes, after a defect that made memory-bound causes unnameable | §8.4 |
| Does it explain a cause it was not told about? | yes, once, with six causal edges pointing away from CPU | §8.2 |
| How often does it explain anything? | **31.5%** of *analysable* incidents — but **17%** of all incidents found, which is what a user experiences | §8.5 |
| Why does it stay silent so often? | **47%** fall below the Granger sample floor — an artefact of one default, though lowering it recovers only four explanations in 210 | §8.5, §8.5.1 |
| What filters the rest? | the statistics accept 115 pairs; 26% are rejected by a hand-written subsystem map, 17% by cycle-breaking | §8.6 |
| False positives at rest? | 1 metric of 29 over 30 minutes (3.4%) | §8.7 |

**Two findings reframe the design.** The causal layer was long described as
near-silent; measured, it explains a third of what it can test, and yield
doubles from 25% to above 50% once analysis windows exceed an hour. It was
never broken — it was **starved**, and the fix is operational rather than
architectural. Separately, the subsystem map that encodes prior knowledge
about how a laptop behaves turns out to be a strict total order in which
`network` is a pure sink; it forbids `network → disk`, `disk → memory` and
`disk → cpu`, three mechanisms that demonstrably exist, and discards the
single strongest relationship the system has ever measured.

**A recurring failure mode is documented as a result in its own right.** Nine
defects were found in code that had a passing test suite around it, and in
almost every case the test asked a question adjacent to the one that mattered
— the SQL value instead of the bytes on disk, the main file instead of the
full footprint, the function instead of the path that calls it. Two fixes were
worse than the defects they repaired, one of which doubled disk usage while
claiming to bound it. They were caught by adversarial review rather than by
more testing, and §10.5 argues why.

The engineering is sound where it has been measured, and the application runs:
a packaged build was launched and observed responding with memory plateauing at
427 MB, and the pipeline processed 92 real incidents end to end in 1.7 minutes.
Version 1.5.0 was published on 2026-08-22 as a 272 MB download. What still
limits distribution is not code but procurement: the binaries are unsigned, so
every user meets a SmartScreen warning, and the no-egress promise remains
irreconcilable with crash reporting — a defect on a stranger's machine is
invisible unless they send the log themselves.

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
| `metrics` | 29 numeric features per 30 s sample | 365 days |
| `metrics.foreground_app` | the application in focus | 30 days |
| `processes` | top-15 by CPU per scan | 30 days |
| `events` | allowlisted Windows Event Log records | 365 days |
| `meta` | schema version, event watermarks, model reference error | — |

The retention asymmetry is deliberate: windows shorten as data becomes more
personal. The focus record is the shortest because it describes a person
rather than the machine (§3.4); process detail is next; the numeric readings,
which are machine state, are kept longest. This ordering is a correction — the
original design had the asymmetry backwards, protecting Event Log text while
the focus log was kept indefinitely — and the 30-day process window is the
direct cause of a defect in §10.

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
The collector contains no network code at all, so its privacy property is
structural rather than policy-based. The desktop application gained one
deliberate exception in 1.5.0: an opt-in check for a newer release, off by
default, which reads a version tag and sends nothing. That exists because a
system with no update path keeps every defect it ships forever, and this
project has documented enough of those to make that trade a bad one. The
claim is therefore stated precisely -- collected telemetry never leaves the
machine, and the collector never opens a socket -- rather than as a blanket
"no network connections", which the update check would make false.

**The protection was inverted, and has been corrected.** `foreground_app` — the
name of the application in focus — is sampled every 30 seconds alongside
`user_idle_sec`. Together they reconstruct when the machine was in use and
roughly what for, which makes them the most personal data the system holds. Yet
the two mechanisms the design was proudest of, the opt-in gate and the
redaction pass, both govern **Event Log message text**: optional, redacted, and
expiring at 365 days. The focus record was on by default, unredactable by
nature, and kept forever. Nothing was concealed — it is declared in the schema
— but the safeguards protected the less sensitive field.

It now expires at **30 days**, matched to the process-sample window because
both describe behaviour rather than machine state, and it is erased as a
*column* rather than by deleting the row: the numeric readings beside it are
machine state and stay for the full window, so coverage, gap detection and
training are untouched. `foreground_app` is not in `MODELLED_COLUMNS` and never
was, so the cost is display context on incidents older than a month and no
accuracy whatsoever — a claim now guarded by a test that fails if either field
ever enters the model's feature set.

The consent dialog also did not mention it. It listed what is recorded every 30
seconds and omitted the focus log entirely, which meant the disclosure was
accurate about the data it protected and silent about the data it did not. It
now names the field, states plainly what it can reconstruct, and gives every
retention window.

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

### 5.2.1 Where the refusal is shown

A design review of the interface found that **every refusal described below
lived in the least visible widget in the application**. The report declines to
say "root cause" without a surviving edge and states outright that an
unsupported ranking is arbitrary — and that text rendered in a
`QPlainTextEdit` on the *fourth* of four tabs, as raw unparsed markdown, in the
smallest and lowest-contrast style in the stylesheet. The tab that opens by
default showed a table of four-decimal scores whose own threshold for a
meaningful difference is 0.01.

The system was honest in its report and confident in its interface, and the
interface is what gets read. Two consequences followed:

- A **verdict banner** now sits above the results tabs and states the finding
  in a sentence before any number appears. It reads the same `causal_support`
  value the report's evidence section uses rather than recomputing the
  judgement, so the two cannot drift apart, and a test asserts that no
  non-supported case ever contains the phrase "root cause".
- The empty causal graph had **no `paper_bgcolor`**, so Plotly defaulted to
  white and the single most important honest state — no surviving edge —
  rendered as a bright rectangle in a dark application, reading as a broken
  chart rather than as a finding. It now carries the same dark ground as every
  other figure and says in words why it is empty.

The same review found that no button in the application drew a keyboard focus
indicator: the stylesheet sets a `QPushButton` background, which stops Qt
painting the native focus rectangle, so Tab moved an invisible cursor across
Run RCA, Train, Find Incidents and both consent buttons. That is a WCAG 2.4.7
failure and is now fixed, along with accessible names on the controls that a
`QFormLayout` could not buddy to their labels.

That first pass fixed the controls it enumerated and missed the two nobody had
listed — `QCheckBox` and `QSlider`, both still at zero changed pixels between
focused and blurred. The checkbox is the Event Log opt-in, which makes it the
worst control in the application to leave unindicated: a keyboard user could
not see which control they were about to toggle to store message text. All
eight focusable widget types now show an indicator, verified by rendering each
one focused and blurred and counting changed pixels, with `sizeHint` asserted
identical in both states so the ring cannot shift the layout.

The verification is worth recording because it went wrong first, in a way that
would have produced a confidently false claim. An early probe reported that the
checkbox rule did not render, and successive experiments appeared to show that
Qt required a background declaration alongside the border, rejected the
`transparent` keyword, ignored `border-color` in favour of the full `border`
shorthand, and silently discarded declarations following an inline comment.
**All four of those conclusions were artefacts.** Qt paints the `:focus`
pseudo-state only for the *active* window, and the probe never called
`activateWindow()`; whichever measurement happened to run while some earlier
window still held activation passed, and the rest failed. Each failure invited
a plausible explanation, and each explanation survived exactly as long as the
next experiment took to contradict it. The elaborate workarounds were reverted
and the rule is four lines. The regression test activates the window
explicitly, and was confirmed to fail when either rule is removed.

### 5.2.2 Layout, and three attempts to fix it

The interface work in this project has a poorer record than the pipeline work,
and the reason is worth stating: **none of it is reachable by the test suite in
the form the user meets it.** 130 tests pass with the layout correct and with
it visibly broken.

Three defects arrived in sequence, each introduced by the fix before it.

**The window could not render its declared minimum.** `setMinimumSize(1024,
640)` was set, while Stage 2 asked for 1168px of height and the Captured Data
table for 1752px of width, and no view sat in a scroll area — so at the minimum
size content was simply clipped, with no way to reach it. The same arithmetic
reaches anyone at 150% display scaling, which is the common configuration on a
Windows laptop rather than an edge case.

**Fixing that broke the header.** An unwrapping label in the header set the
window's real floor at 1155px, so it was given `setWordWrap(True)` and a
minimum width of 1 — while a trailing spacer continued to take all the spare
width. Qt duly handed the label its minimum and wrapped it to **one word per
line**, a thin vertical column beside the title. The instinct was right and the
layout arithmetic was not.

**Fixing that exposed nested scrollbars.** With every tab in a scroll area, any
content taller than the window produced a page-level bar beside the ones the
tables and figures already carry — two bars in the same corner, which reads as
the panel sliding under itself. The causes were a channels table demanding
430px and two `QWebEngineView` figures asking for roughly 700px each, putting
the results panel at 775px on a display with room to spare.

The resolution is that a preferred size is not a minimum: the figures now ask
for 340px and expand to fill whatever room exists — measured, 480px on a
1900×1000 window — so the page fits when it can and scrolls only when it must.

| window | outer scrollbars |
|---|---|
| 1900×1000 | none on any tab |
| 1366×768 | all three tabs |
| 1024×640 | all three tabs |

**What made this expensive was measuring the wrong thing.** Two of the three
were "verified" against offscreen geometry that did not reflect what a screen
would show — `resize()` before `show()` is silently ignored offscreen, so an
early check ran at 796px and reported wrapping that does not occur at 1900px.
The defects were found by rendering the window to an image and looking at it,
which is the only method here that has worked reliably.

The regression test asserts on `sizeHint` rather than on whether a scrollbar is
currently painted. Visibility depends on how far the event loop has run, which
made the first version pass in isolation and fail in the full suite — a test
that reports on scheduling rather than on layout.

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
| Database growth | **3.33 MB/day**, unbounded |
| Cold first launch | 52 s |
| Warm launch | 9.5 s |

RAM is comfortable. The CPU figure is higher than a background sampler should
need and is flagged for a proper profile — the 90-second measurement window may
have caught a process-sampling burst rather than steady state.

**Disk is not comfortable, and the earlier figure in this paper was wrong.**
Re-measured over a longer span:

| Property | Value |
|---|---|
| Span | 18.2 days |
| `samples` / `proc_samples` / `events` | 21,257 / 515,427 / 4,677 |
| Database | 60.6 MB |
| Growth | **3.33 MB/day** |

The previously published 2.2 MB/day was taken over a shorter, quieter window
and understated the real rate by 51%. Extrapolated, year one is roughly 1.2 GB.

Two design facts made that a trajectory rather than a number. The `samples`
table had **no retention at all** — `proc_samples` expired at 30 days and
`events` at 365, but the metric history was kept forever. And there was no
`VACUUM` and no `auto_vacuum` pragma, so when the 30-day process purge began to
fire it would delete rows into free pages that sqlite reuses internally and
**never returns to the filesystem**. The file could shrink in content but not
in size. Neither behaviour had been observed in the wild only because the
installation was younger than the retention window it would trip.

A diagnostic tool that quietly consumes the disk it is diagnosing has a defect
of the same family as the evaluation harness that could exhaust memory (§10.3):
correct in intent, unbounded in practice.

### 6.3.1 What was done about it

Metric history now expires at **365 days**, matching the event window so that
an incident still visible in the Event Log always has telemetry left to explain
it. A shorter figure would leave the application listing year-old faults it can
no longer analyse, which is a worse failure than the disk cost.

Reclamation is the half that makes retention visible, and it is gated twice:

$$\text{vacuum} \iff \text{free pages} \ge 2000 \;\wedge\; \text{disk free} \ge 2 \times \text{db size}$$

with the free space also required to be worth at least a tenth of the file.
The absolute floor alone was the wrong shape: at the measured growth rate a
daily purge frees roughly 8 MB, which clears 2,000 pages — so on a 1.2 GB
database `VACUUM` would have fired every two or three days to rewrite the
entire file for **0.7% of it**, which is exactly the churn the floor was
introduced to prevent. Both gates apply, so a small database is governed by
the constant and a large one by the proportion.

The first gate exists because a full rewrite for a few kilobytes is not worth
the daily churn. The second exists because `VACUUM` builds a complete second
copy before swapping it in — starting one without room for it is how a cleanup
task becomes the outage it was meant to prevent, which is the same rule the
memory-fault harness had to learn in §10.3. A failed or skipped vacuum loses
nothing: the space stays on the free list and the next daily pass retries.

**This cannot be demonstrated on the development machine, and is not claimed
to be.** Measured at the time of the change:

| | |
|---|---|
| Database | 71.8 MB |
| Reclaimable right now | **0.0 MB** |
| Rows past retention | **0 of 24,695 samples, 0 of 612,734 process samples** |

The installation is 21 days old and the shortest retention window is 30 days,
so no purge has ever fired and there is nothing yet to reclaim. The mechanism
is covered by a test that fills a database, deletes from it, and asserts the
**file on disk** shrinks by more than half — which fails if reclamation is
removed — but the live path first executes around day 30. The honest status is
*implemented and unit-tested, not yet observed in production*.

Sequencing was deliberate. Corruption recovery (§10.4) landed **before** this,
because `VACUUM` rewrites the entire database and is the single most
dangerous routine operation the application performs.

### 6.3.2 The fix was worse than the defect, and its test hid that

An adversarial review of the change above found a fourth instance of the
pattern catalogued in §10.2 — and this one was written two commits after
documenting it.

**`VACUUM` in WAL mode writes the entire new database into the WAL.** Without a
checkpoint nothing reaches the filesystem, and the collector holds its
connection for the whole logon session, so the space never returned. Measured
on a 45 MB database:

| | main | wal | total |
|---|---|---|---|
| before purge | 45.0 | 0.0 | 45.0 MB |
| after delete | 45.0 | 44.7 | 89.7 MB |
| **after `reclaim()` — as shipped** | 45.0 | 44.7 | **89.7 MB** |
| after checkpoint | 0.0 | 0.0 | 0.1 MB |

The change titled "bound the database so collection cannot fill the disk"
**doubled the footprint** at the moment it claimed to reduce it, and because
the Captured Data tab sums `main + wal + shm`, the user would have watched
disk usage rise immediately after the cleanup.

**The test passed because it issued `PRAGMA wal_checkpoint(TRUNCATE)` between
`reclaim()` and the measurement** — a statement that appears nowhere in `src/`
— and measured only the main file. It asserted a shrink the application never
performed, on half the footprint. That is the §10.2 signature exactly: a test
that does something production does not, and therefore cannot fail for the
reason it exists.

`reclaim()` now checkpoints, and the test measures `main + wal + shm` with no
checkpoint of its own. Re-measured after the fix: **89.7 MB → 0.1 MB.**

A second finding from the same review is worse, because it made the consent
dialog untrue. `PRAGMA secure_delete` defaults to off, so
`UPDATE samples SET foreground_app = NULL` unlinked the cell and left the
string readable in the page freeblock — and since an in-place shrink frees no
pages, `VACUUM` was gated off by the free-page threshold and never compacted
it away. Measured: **745 copies of a test application name survived a purge
that reported 2,000 rows blanked**, while §3.4 and the dialog both told the
user it had been erased. The test asserted the *SQL value* was NULL, which is
the wrong question. `secure_delete` is now on and the test greps the file
itself: **745 → 0**.

Three further findings are recorded in §10.5.

### 6.3.3 Removing a browser to draw two charts

The largest single component of the installed application was not the model,
the runtime or the data. It was a web browser.

Figures were built with Plotly, which renders to HTML and therefore needs
something to display HTML. That something was QtWebEngine:

| Component | Size |
|---|---|
| WebEngine DLLs | 258 MB |
| WebEngine resources | 29 MB |
| Qt translations | 53 MB |
| `opengl32sw.dll` (software fallback) | 20 MB |
| **total** | **~360 MB** |

About a third of a 1,110 MB installation existed so that two charts could be
drawn. Matplotlib renders the same two figures into a native Qt canvas for the
28 MB it costs, and the application already had matplotlib available.

| | before | after |
|---|---|---|
| Download (ZIP) | 433 MB | **272 MB** |
| Installed | 1,110 MB | **731 MB** |
| PySide6 alone | 461 MB | **92 MB** |
| GUI resident memory | 542 MB | **473 MB** |

Two properties improved as side effects rather than by design. Figures had
been written to temporary HTML files and loaded over `file://`, which left
**rendered metric values in the user's temp directory** and required
`delete-all-data` to clean them up explicitly; that path no longer exists. And
pan and zoom now come from the standard matplotlib toolbar rather than from
JavaScript. The measurable loss is hover tooltips.

**Three packaging defects surfaced, all with the same signature.** The build
succeeded, the application launched, the collector ran, and the failure waited
for someone to draw a figure:

1. `matplotlib` was on the exclude list, left from when the application used
   Plotly and matplotlib was dead weight.
2. `cycler`, `contourpy`, `fontTools` and `kiwisolver` — matplotlib's own
   dependencies — were excluded for the same reason.
3. The fix for (2) **silently missed one**. The entry is `fontTools` with a
   capital T; a case-sensitive replacement matched nothing and reported
   success having done nothing.

None is reachable by launching the application and looking at it, which is
precisely the check that would otherwise have been used to call the build
good. The first two were found by reading `desktop.log` after launch; the
third by a test.

**The guard test is the part worth keeping.** Its first version named three
modules by hand and passed while four required packages were still excluded.
Rewritten to ask `importlib.metadata` what matplotlib actually declares, and
to compare case-insensitively against the exclude list, it immediately caught
the `fontTools` case bug that the fix had missed. The difference is between a
test encoding what the author remembered and one that interrogates the system.

A claim made during this work was also wrong and is corrected here.
`matplotlib.backends.backend_qtagg` was reported missing from the bundle on
the evidence that `_internal/matplotlib/backends/` held a single file. Pure
Python modules live in the PYZ archive rather than as loose files, so looking
for them on disk finds nothing whether they are present or not; the build
manifest confirms it was always there. The hidden import added in response
remains as insurance against a future refactor moving the import behind
`matplotlib.use()`, not as a fix for a defect that existed.

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

The original draft stated that no measurement of detection quality existed
anywhere in the work, and named fault injection as the remedy. Evaluation now
proceeds on two levels, which answer different questions and have different
weaknesses.

**Fault injection** (`tools/evaluate_detection.py`) manufactures the ground
truth a personal machine cannot otherwise provide: cause a specific known
disturbance, wait for the samples to land, run the real pipeline over the
injection window, and score what came back. It is the only way to ask *"is the
answer correct"*, because only here is the answer known in advance. It is also
expensive — the collector must be running and each run takes tens of minutes at
the real sampling cadence — so it yields single observations, and a single
observation cannot distinguish a property of the system from an accident of the
run.

**Population survey** (`tools/measure_causal_yield.py`,
`tools/audit_topology_map.py`) runs the same pipeline over every incident the
detector finds in real collected history — 175 of them, 92 long enough to test
— in under two minutes, with no injection and no privileges. It cannot say
whether any individual answer is right, because nothing here has a known cause.
What it gives instead is *rates*: how often the causal layer produces anything,
how that varies with window width, and which filter removes what. Several
claims in earlier revisions of this paper rested on one or two runs and are
corrected below by numbers with a sample size.

The two are complementary and neither substitutes for the other. Injection
establishes correctness on a handful of cases; the survey establishes frequency
across many. §8.2 to §8.4 report the injections, §8.5 and §8.6 the survey.

### 8.1 Results

| Run | Samples | Flagged | Accepted pairs | Edges | Attributed | Verdict |
|---|---|---|---|---|---|---|
| CPU burn, 7 min | 14 | 6 of 29 | **0 — never tested** | 0 | yes | PASS (detection only) |
| CPU burn, 30 min | 60 | 6 of 29 | 10 | **6** | yes | **PASS, explained** |
| Disk burn, 30 min | 60 | 4 of 29 | 1 | **0 (pruned)** | yes | PASS, unexplained |
| Memory hold, 30 min | 60 | 2 of 29* | 0 | 0 | **no** | **FAIL — wrong culprit** |
| Memory hold, 30 min (after fix) | 60 | 3 of 29* | 0 | 0 | yes | PASS, unexplained |
| Memory hold, 30 min (clean baseline) | 60 | **0 of 29** | — | — | — | **not measurable here (§8.4.1)** |
| Idle, 30 min | 60 | 1 of 29 | — | — | — | 3.4% false positive |

\* Both earlier memory runs were scored on a machine already at 84% and 98%
memory and already swapping; what they flagged was `swap_used_delta`, which may
have been ambient rather than injected. Read their detection results as
unproven — see §8.4.1.

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

**Re-run live, the fix holds — and exposed a second, milder version of itself.**
A fresh 30-minute injection of 0.78 GB was detected and attributed: the harness
reported `ATTRIBUTED to us: yes` under its newly-enforced gate, and the process
holding **1,135 MB** appeared in the table where before it could not. But it
appeared *tenth of ten*, below nine innocent processes, because selection now
used both dimensions while the ordering still used CPU alone — and its CPU was
0.0002%. Naming the culprit in last place is most of a fix rather than a fix.

Ordering now scores each process by its share of the window's maximum in
*either* dimension, and sorts on that:

$$\text{prominence}(p) = \max\!\left(
\frac{\overline{\text{cpu}}_p}{\max_q \overline{\text{cpu}}_q},\;
\frac{\text{rss}_p}{\max_q \text{rss}_q}\right)$$

Measured against both stored windows: on the memory incident the 1,135 MB
process moves from tenth to second, and on the CPU incident the busiest process
still leads on CPU as before. The rule needs no knowledge of which metrics were
anomalous, which is why it is a sort and not a branch.

That `mem_pct` did not flag is a second, unresolved observation, and the re-run
**reproduced it**. Both injections moved total memory use substantially — 9
points in the first, 5 in the second — and in both the autoencoder found that
unremarkable while flagging the swap movement that followed. Twice is a pattern
rather than an accident, so the reading that this host's baseline already spans
a wide range of memory pressure, leaving 93% inside learned normal, is now
better supported. It remains a hypothesis: nothing has tested it directly, and
the alternative — that memory features are under-weighted somewhere in scaling
or thresholding — has not been ruled out.

In both runs the detection that *did* fire came from `swap_used_delta`, a
derived rate, rather than from the level metric a person would look at. If the
level metric is systematically insensitive on hosts that habitually run near
full, that is a real gap for the machines most likely to need this tool.

### 8.4.1 The memory fault cannot be measured on this machine, and the harness said FAIL

A third memory run was performed on a deliberately cleared machine — 4.51 GB
free, 71.3% used, swap at 18.3%, against the 84% and 98% of the earlier two.
A **1.91 GB** hold over 30 minutes produced:

```
samples analysed : 60
metrics flagged  : 0 -> []
DETECTED         : NO
FAIL
```

Nothing flagged at all. Comparing the injection window against the preceding
half hour explains why, and it is not a detector defect:

| metric | 30 min before | during fault | delta |
|---|---|---|---|
| `mem_pct` | 93.8 | 87.4 | **−6.3** |
| `mem_available_mb` | 1,006 | 2,023 | +1,017 |
| `swap_pct` | 44.8 | 17.8 | −27.0 |

**Memory was freer during the fault than before it.** Freeing 4 GB to make the
test possible moved the machine further from its own normal than the injection
then moved it back. Against the learned distribution the window is not
anomalous in the slightest:

```
mem_pct over 30,199 collected samples
p25 93.2%   p50 95.2%   p75 96.9%   p95 98.1%   max 99.9%
fault window average 87.4%  ->  the 9th percentile
```

The model was right. An 87.4% window on a host whose median is 95.2% is
unusually *calm*, and flagging it would have been a false positive.

**The defect is in the harness, and it is structural.** The injection is
bounded to half of available memory — a safety rule adopted after an earlier
version allocated until `MemoryError` (§10.3). On a machine that habitually
runs near full, that bound is also a guarantee of failure: half of what is
free cannot take usage past a level the machine already reaches unaided. When
little is free the budget is tiny; when much is free it is because usage just
dropped, so the injection only claws back toward normal. **A bounded memory
injection is undetectable on this host by construction**, and no duration or
retry changes that.

Two consequences follow. The harness now checks reachability before injecting
and reports `INCONCLUSIVE` with the arithmetic rather than `FAIL`:

```
a 1.17 GB hold would reach 92.5% memory, under this machine's own p95 of
98.1% -- the injection cannot exceed normal, so nothing here could detect it
```

Blaming the detector for a limit of the test is the same error as §6.3.2's
test that checkpointed where production did not, arriving from the opposite
direction: there a test passed for a reason the product did not earn, here a
test failed for a reason the product did not deserve.

And it **retroactively weakens the two earlier memory runs**. Both were
scored on a machine already at 84% and 98% memory, already swapping, and what
they flagged was `swap_used_delta` — plausibly the ambient thrashing rather
than the injection. Their attribution results stand, having been re-checked by
hand, but their *detection* results should be read as unproven rather than as
passes. The evaluation table is annotated accordingly.

**What this does not say.** It is not evidence that memory detection is
broken; it is evidence that this machine cannot test it. A host with a normal
distribution of memory usage would answer the question in one run. That is now
the strongest argument in this paper for repeating the evaluation on second
hardware — not generality for its own sake, but because one measurement is
unobtainable here at all.

### 8.5 Population evidence: causal yield across 92 real incidents

Every causal claim so far rested on two injected faults. The collected history
contains far more, so the pipeline was run over all of it — no injection, read
only, `tools/measure_causal_yield.py`. **175 incidents found, 92 analysable,
1.7 minutes.**

| Outcome | Count | Share of analysable |
|---|---|---|
| no supported causal chain | 58 | 63.0% |
| **supported** | **29** | **31.5%** |
| pruned by topology | 2 | 2.2% |
| no anomaly detected | 2 | 2.2% |
| not tested — window too short | 1 | 1.1% |

**Quote the funnel, not the survivors.** 31.5% is the share of incidents that
were *analysable*, which is the flattering denominator and not the one a user
lives in:

```
incidents the detector found        175   100%
...long enough for Granger           92    53%
...that produced any causal edge     29    17%
```

**One in six.** Someone who sees an incident has roughly a 17% chance of
getting any causal explanation at all. Earlier revisions of this paper, and its
abstract, quoted 31.5% without the denominator; that is corrected here and
there. "Supported" also means only that at least one edge survived — the survey
has no ground truth, so it is a ceiling on usefulness rather than a measure of
correctness.

Three results follow, and two of them change what this paper claims.

**Starvation is confirmed at scale, and quantified.** 83 of 175 incidents —
**47%** — sit below the Granger floor and cannot be tested at any setting.
Among those that can be, yield climbs with window width exactly as the CPU
7-versus-30-minute comparison predicted from a single pair:

| window | n | explained | rate |
|---|---|---|---|
| 0–30 min | 48 | 12 | 25% |
| 30–60 min | 20 | 5 | 25% |
| 60–120 min | 10 | 5 | **50%** |
| 120–360 min | 9 | 5 | **56%** |
| 360 min+ | 5 | 2 | 40% |

The rate doubles once windows pass an hour. The dip in the last row is five
incidents and should not be read as a trend.

**The subsystem map is doing far more work than anyone knew.** Across all 92
incidents the statistics accepted **115** pairs after FDR correction and the
effect-size floor, and only **65 (57%)** reached the final graph. §8.6 breaks
down what removed the other 50, and audits whether it should have.

**The layer is not as silent as previously described.** Earlier text
characterised it as producing nothing on the majority of incidents, inferred
from a handful of cases. Measured, it explains **31.5%** — a minority, but a
substantial one, and above an hour of window it is a coin flip. Where it does
explain, the median chain is 2 edges and the leading metric is overwhelmingly
disk or swap: `disk_busy_pct` (8), `swap_used_delta` (4), `disk_read_bps` (4),
`disk_free_pct` (3), `swap_used_bytes` (3).

The survey costs nothing to repeat and should be re-run whenever the gates,
the map, or the model change — it is the only measurement here with a sample
size worth the name.

### 8.5.1 The sample floor is a parameter artefact — and removing it recovers little

The 47% figure above invites an obvious reading: incidents are too short, the
data are thin, nothing can be done. That reading is wrong on the mechanism and,
it turns out, also wrong about the remedy.

**Where the cliff comes from.** A detector-flagged run can be as short as three
samples, so the pipeline widens it before offering it for analysis — to exactly
the model's window size, **15 samples**. Granger at the default lag of 5 needs
`3L + 2` = **17**. Two constants chosen independently in the same pipeline, and
the widening target lands two samples under the causal floor. The incident
duration distribution has a hard spike at 7.0 minutes for precisely this reason:
p10 and p25 are both exactly 7.0, which is 15 samples at a 30-second cadence.

The cliff is therefore an artefact of one default, not a property of the data:

```
lag 2  floor  4.0 min  ->  202/202 testable (100%)
lag 3  floor  5.5 min  ->  202/202 testable (100%)
lag 4  floor  7.0 min  ->  202/202 testable (100%)
lag 5  floor  8.5 min  ->  107/202 testable ( 53%)   <- the default
```

**What removing it actually buys.** The survey was re-run end to end at lags 3,
4 and 5 on an identical 210-incident population, which is the measurement that
settles it:

| lag | testable | explained | of all incidents | median edges | max edges |
|---|---|---|---|---|---|
| 3 | 210 | 32 | 15.2% | 1 | 11 |
| 4 | 210 | 34 | 16.2% | 1 | 13 |
| **5** | **113** | **30** | **14.3%** | **2** | **7** |

Lowering the lag to 4 makes *every* incident testable — 113 to 210, nearly
double — and yields **four more explanations**. The incidents hidden below the
floor were overwhelmingly ones that produce nothing when tested. The hypothesis
that a parameter mismatch was concealing recoverable insight is measured and
mostly false.

There is also a cost that runs the other way. Median surviving edges falls from
**2 to 1** while the maximum rises from 7 to 13: shorter lags admit more
relationships, and the ones they admit are shallower. Accepted pairs rise from
116 to 130 and the topology map's rejections rise with them, 50 to 60 — the
extra findings are not obviously better findings.

**So the honest gain is in reporting, not in insight.** At lag 5, 47% of
incidents are reported as "not tested — window too short", which tells a user
nothing about their machine and everything about a constant they cannot see. At
lag 4 those same incidents return "tested, nothing survived", which is a real
negative result. The user learns no more about the cause either way, but one
answer is evasive and the other is not — and this project has spent most of its
later effort on exactly that distinction.

The default is left at 5 pending a decision, because the trade is genuine:
richer chains and an evasive silence, against flatter chains and an honest one.
What is no longer defensible is describing the floor as thin data. It is a
constant sitting one notch above a length the pipeline manufactures for itself.

### 8.6 Population evidence: auditing the subsystem map

The first attempt at this measurement got the attribution wrong, and the error
is worth stating because it is the same shape as the others in this paper. The
50 removed pairs were all attributed to the topology map. They were not: cycles
are broken inside `CausalGraphBuilder.build()` **before** `refine_causal_graph`
ever sees an edge, so the figure conflated two filters. The tell was in the
data and was initially read past — `disk → disk` appeared as "rejected" 8
times, and the map permits same-subsystem edges by construction, so something
else was removing them. Asking the map directly rather than inferring its
verdict from the survivors gives:

| | pairs | share |
|---|---|---|
| accepted by the statistics | 115 | — |
| survived to the final graph | 65 | 57% |
| **rejected by the subsystem map** | **30** | **26%** |
| removed by cycle-breaking | 20 | 17% |

26%, not 43%. Still the largest single filter after the statistical gates, and
still never validated — but a quarter rather than a half.

**What the map rejects.** Every rejected transition has *zero* survivors,
because the map forbids whole classes rather than individual edges:

| transition | rejected | kept | strongest rejected pair |
|---|---|---|---|
| `disk → memory` | 13 | 0 | `disk_busy_pct → swap_used_bytes` (0.758, lag 2) |
| `memory → cpu` | 7 | 0 | `swap_pct → cpu_freq_mhz` (0.138, lag 1) |
| `network → disk` | 4 | 0 | `net_recv_bps → disk_write_bps` (**0.982**, lag 2) |
| `disk → cpu` | 3 | 0 | `disk_busy_pct → cpu_pct` (0.322, lag 3) |
| `memory → process` | 3 | 0 | `swap_used_bytes → process_count` (0.109, lag 1) |

**The map is a strict total order**, and that is the root of it:

```
power, process  →  cpu  →  memory  →  disk  →  network
```

`network` has out-degree zero — a pure sink. Nothing in this system can ever be
reported as caused by network activity. `power` and `process` have in-degree
zero and can never be caused by anything. Because `is_path_possible` uses
`nx.has_path`, the order is transitive: every "upstream" direction is forbidden
outright.

That encodes an assumption — resource pressure flows one way, from power
through compute to I/O — which is defensible as a first sketch and wrong as a
description of a real machine. Judged individually:

- **`network → disk` is almost certainly a real mechanism the map cannot
  express.** A download arrives over the network and is written to disk. The
  strongest single relationship in the entire dataset, at **0.982**, is
  `net_recv_bps → disk_write_bps`, and it is discarded because network is a
  sink. The map has `disk → network` — the upload direction — and not its
  mirror.
- **`disk → memory` is real on Windows.** Sustained I/O grows the standby file
  cache, which consumes available memory. 13 rejections, the most of any
  transition.
- **`disk → cpu` is real.** Interrupt and DPC handling, and I/O wait, put
  kernel time on the CPU during heavy disk activity.
- **`memory → cpu` is plausible but confounded.** Page-fault handling does cost
  CPU, but the strongest instance targets `cpu_freq_mhz`, which is governed by
  thermal and power policy; both sides are more likely driven by overall load
  than by each other.
- **`memory → process` looks spurious.** Strength 0.109 against a floor of
  0.10 — it barely cleared the gate, and memory pressure causing process count
  to change is hard to argue.

So of the five rejected classes, three describe mechanisms that genuinely exist
and one is probably noise. **The map is not wrong so much as one-directional**:
it models resource pressure flowing downhill and has no vocabulary for the
feedback that makes I/O expensive. Adding the reverse edges would be simple,
but it would also make the graph cyclic — and cycle-breaking already removes
17% of pairs, so the two mechanisms would begin to fight. That is a design
question, not a patch, and it is recorded here rather than answered.

None of this establishes that the rejected pairs are causal. Granger causality
over correlated resource metrics finds a great deal that is confounded, and the
map exists precisely to encode prior knowledge the statistics do not have. The
finding is narrower and firmer: **the prior it encodes forbids mechanisms that
demonstrably exist, and one of them accounts for the strongest relationship
this system has ever measured.**

### 8.7 Idle: the false-positive floor

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

### 8.8 What this evaluation still does not establish

Six runs on **one machine**. The following remain undone and are not claimed:

- **A fault whose cause is not the loudest metric.** Both explaining runs put
  the injected metric at the top of a severity ranking. Neither could have
  distinguished a correct causal answer from a correct *severity* answer.
- **Repeats on other hardware.** Every number here is from one host.
- **Causal yield across many incidents**, rather than the two reported here.
- **Why `mem_pct` does not flag** under a memory injection, reproduced twice,
  while `swap_used_delta` does.
- **Whether the prominence ordering is right when both dimensions are loaded.**
  It was measured on incidents dominated by one or the other.

The honest claim is **"demonstrated once"**, not "validated".

### 8.9 What the harness is actually for

The evaluation's most valuable output has not been a pass. Of six runs, one
explained a fault end to end, one showed the causal layer being starved rather
than broken, one caught the report blaming statistics for a topology decision,
one caught a production defect that made an entire class of root cause
unnameable — plus a defect in the harness's own scoring — and the re-run of
that last fault caught the *fix* being incomplete, since a culprit named tenth
of ten is not much use to whoever is reading.

The pattern is consistent enough to state as a working rule: **the runs that
found defects were the ones chosen for being different in kind from the runs
before them.** A second CPU fault would have passed and taught nothing. This
argues for selecting future faults by what they would exercise that nothing has
exercised yet, rather than by what seems likely to succeed.

---

### 8.10 Three product changes, and a promise that had to be rewritten

Three changes were made for the sake of someone who is not the author, and one
of them required restating a claim this project had made absolutely.

**An off switch.** The application records which programs are in use every 30
seconds, and the only way to stop it was to uninstall. For software whose case
rests on privacy that is not defensible, and the mechanism already existed:
both the collector and its supervisor poll a stop flag. It had simply never
been offered to the person being recorded. Resuming deliberately restores
*supervision* rather than just the collector, because the flag ends the
supervisor's loop too, and bringing back a bare collector would leave it
unsupervised until the next logon — working, but quietly weaker than before.

**A first run that does not look broken.** With no database yet — the normal
state for the first minutes after installing — Stage 1 displayed
`python -m telemetry install` beside a raw `unable to open database file`.
Someone who downloaded a packaged ZIP has neither Python nor the module, so
the first screen a new user saw was an impossible instruction next to what
reads as a crash, while the installation was working exactly as intended. A
frozen build now says collection starts on its own and to check back in about
a day; a source build still gets the command, because there it is correct.
The 21-hour wait also shows progress — *"25% collected — about 0.7 days to
go"* — since a screen that says "wait" and never visibly moves is
indistinguishable from one that has stopped working.

**An update check, and the cost of having one.** A system with no update path
keeps every defect it ships. Given the nine defects documented in §10, that
trade was a bad one, so the desktop application gained an opt-in check that
reads the newest release tag and nothing else: off until enabled, asked once
in plain terms, run only on a button press, no identifier, no download, no
request body and no query string. Tests assert each of those properties,
because a version check is exactly the kind of feature that becomes telemetry
by accretion.

**It made a documented claim false, so the claim changed.** Five documents
stated that the application "makes no network connections". That is no longer
true, and the honest repair is not to hide the socket but to describe the
system precisely: *collected telemetry never leaves the machine, and the
collector opens no sockets under any configuration*, with the opt-in exception
named wherever the guarantee is given. The collector's property remains
structural — there is no network code in the collection path at all — while
the desktop application's is now conditional on a choice the user makes. That
distinction is worth more than the simpler sentence it replaced.

---

## 9. Distribution

PyInstaller `--onedir`, two executables, unsigned.

| Property | Value |
|---|---|
| Version | 1.5.0 |
| Installed size | **731 MB** (from 1,110 MB; see §6.3.3) |
| Release ZIP | **272.3 MB** |
| SHA256 | `24D1D091DDB60A27D5EA79F0E90A5CDECABD7F213DCA3AEC459BA8B386720B65` |
| Install | extract → run → agree |

**Published 2026-08-22**, at
`github.com/mathuryashash/RCA-Major_project/releases/tag/v1.5.0`. Before that
point the project had produced twelve releases and published exactly one, an
early v1.0.0 that was deleted when 1.5.0 went out: it predated the attribution
fix that made memory-bound causes nameable at all, bounded storage, corruption
recovery, the pause control and every layout fix, and it had been sitting on
the repository as "Latest" throughout.

Four finished builds were discarded rather than shipped, on a single
principle: an artifact whose checksum appears in a document, while its code has
since moved, is the kind of thing someone later trusts by mistake. 1.2.0 was
superseded by the process-ordering fix, a first cut of 1.3.0 by the storage
corrections, 1.4.0 by three layout regressions, and 1.4.1 by the size work.
Throwing away a 433 MB build four times is cheaper than publishing a checksum
that describes something else.

One packaging error survived that discipline and is worth recording because it
is exactly what the discipline exists to prevent: the release notes for 1.4.1
carried **1.4.0's SHA256**, because the notes were copied and the filenames
updated while the hash was not. It was caught before publication, and the
notes for 1.5.0 were written with a `PENDING_SHA256` placeholder that the
build fills in, so the value cannot be inherited from a previous release
again.

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

### 10.4 A security and storage audit, including what it got wrong

A full-codebase audit was run against the shipped 1.2.1 build. Its most useful
output was a finding that **did not survive checking**.

`torch.load` appears four times, loading a model artifact from a user-writable
directory. That is the textbook Python deserialization vulnerability, and it
was written up as one. Then it was verified rather than asserted:

```
torch 2.12.0+cpu
inspect.signature(torch.load).parameters['weights_only'].default  →  None
```

In torch ≥ 2.6, `None` resolves to `True`: the restricted unpickler is already
in force, and the artifact was confirmed to contain only `OrderedDict`, `list`,
`str`, `int` and `float`, so it loads cleanly under it. **There was no
vulnerability.** Reporting one would have been a false positive of exactly the
kind the audit was supposed to filter. The calls now pass `weights_only=True`
explicitly — which changes nothing today, and prevents a torch downgrade from
silently turning "load a model file" back into "execute whatever is in it".

The same fate met the SQL finding. One f-string interpolation exists, at
`analysis.py:214`, and it draws its table name from a hardcoded tuple, so it is
not injectable. Everything else is parameterised.

What did survive:

- **The storage trajectory** of §6.3 — unbounded `samples`, no vacuum.
- **`foreground_app`, recorded every 30 seconds and never purged.** 21,247
  records across 36 applications on the audited machine. This is the most
  sensitive field the system collects and it receives the *least* protection:
  the opt-in gate and the redaction pass both apply to Event Log message text,
  which is optional and expires in a year, while the focus log is on by
  default, unredactable by nature, and permanent. Combined with
  `user_idle_sec` it reconstructs when the machine was in use and what it was
  used for, at 30-second resolution, indefinitely. Nothing is concealed — it is
  declared in the schema — but the protection is inverted relative to the
  sensitivity.
- **An apostrophe in a Windows username breaks collection entirely.** The
  generated supervisor assigns the collector path into a PowerShell
  single-quoted string, which ends at the first apostrophe. A user named
  O'Brien got a script that would not parse, so nothing started at logon and
  nothing said why. No privilege boundary is crossed, so this is a correctness
  defect rather than a vulnerability — and it is the same shape as the space in
  a profile path that broke the original `schtasks` registration.
- **No corruption handling.** There was not one `sqlite3.Error` handler in the
  codebase. An unclean shutdown that damaged the database meant the application
  opened to a traceback with no route back except deleting data by hand.

The corruption fix is worth stating precisely, because the obvious version of
it destroys data. `sqlite3.OperationalError` — which is what "database is
locked" raises — is a **subclass** of `sqlite3.DatabaseError`. Catching the
parent and treating it as damage would move a healthy database aside every time
the collector held a write lock. The damaged file is now renamed rather than
deleted, and contention is explicitly excluded from the recovery path, with a
test that fails if a locked database is ever quarantined.

---

### 10.5 What adversarial review caught that testing did not

Two reviewers were pointed at the previous section's changes with one
instruction: this project has a documented history of fixes that pass their
own tests while destroying data, so find the next one. They found five defects
between them, in code that had 119 passing tests.

| Defect | Why the tests missed it |
|---|---|
| `VACUUM` doubled the on-disk footprint (§6.3.2) | the test checkpointed; production never does |
| `foreground_app` was not erased, only unlinked (§6.3.2) | the test asserted the SQL value, not the file |
| `quarantine()` moved the database aside, failed to remove the poisoned WAL, and suppressed the log line naming where the history went | no test injected a sharing violation on the sidecar |
| Up to twelve full-size `.corrupt-*` copies, uncounted in "size on disk" | nothing tested repeated quarantine |
| The verdict banner was never cleared on failure | the test called `_set_verdict` directly, never the failure path |

The banner case is the most instructive. `workers.py` routes "no anomalies were
detected" through the **failure** signal, so the commonest benign outcome left
the previous run's *"Likely root cause: cpu_pct — supported by 6 causal edges"*
displayed above a line reading "Failed: No anomalies were detected". The test
exercised the function in isolation and could not see it.

Two smaller ones are worth recording because both inverted their own intent.
`setAccessibleName` on a `QLabel` **replaces** the label's text for assistive
technology, so naming the banner "Analysis verdict" meant a screen reader
announced two words and none of the finding — the accessibility fix made the
headline feature less accessible. And one new assertion read
`assert view.verdict.isVisible() or True`, which is true unconditionally; it
would have passed with the feature deleted.

Four statements in the consent dialog were also found to be false: process
sampling tightens to 30 seconds under load rather than the stated 5 minutes;
"nothing is kept indefinitely" is untrue once the collector is stopped, since
every purge runs only from its loop; quarantined copies obey no retention rule;
and the focus record was described as the shortest-lived when it is tied with
process detail. All four are corrected, and the dialog now states the caveat
about retention depending on a running collector rather than implying it is a
property of the data.

**The general lesson is not "write more tests".** All five defects sat behind
tests that existed and passed. They were missed because each test asked a
question adjacent to the one that mattered — the SQL value instead of the
bytes, the main file instead of the footprint, the function instead of the
path that calls it. Adversarial review found them in a single pass because it
was asked to assume the fix was wrong rather than to confirm it was right.

---

## 11. Limitations

Ordered by how much they constrain what this system can claim.

**Evidence**

1. **Everything is one machine.** Seven injections and a 92-incident survey,
   all from a single Windows 11 host. Nothing here establishes generality, and
   one measurement — memory detection — is provably *unobtainable* on this
   host (§8.4.1), which makes second hardware a prerequisite rather than a
   nicety.
2. **No fault has been tested whose cause is not also the loudest metric.**
   Both explaining runs put the injected fault at the top of a severity
   ranking, so a correct causal answer and a correct severity answer remain
   indistinguishable. The memory route to testing this is closed on this
   machine; `process_count` is the best remaining candidate.
3. **The survey measures frequency, not correctness.** Its 92 incidents have
   no known cause, so a 31.5% explanation rate says how often the layer speaks,
   never whether it is right.

**Method**

4. **47% of incidents fall below the Granger floor at the default lag** —
   an artefact of the pipeline widening short incidents to 15 samples while
   the causal layer needs 17, not a property of the data. Measured, lowering
   the lag makes every incident testable and recovers four explanations in
   210, so the gain is honest reporting rather than insight (§8.5.1).
5. **The subsystem map is a strict total order** in which `network` is a sink,
   and it forbids three mechanisms that demonstrably exist (§8.6). Adding the
   reverse edges makes the graph cyclic, where cycle-breaking already removes
   17% of accepted pairs — the two filters would begin to fight. Unresolved by
   design rather than by neglect.
6. **`model_stale` conflates** a stale model with an old analysis window.
7. **Collection coverage was 27.8%** with a median segment of exactly the
   Granger floor. The figure predates supervision and has not been re-measured.

**Unobserved in production**

8. **Retention has never actually run.** Metric history expires at 365 days,
   the focus record at 30, and freed space is returned to the filesystem — all
   unit-tested, none observed, because the installation is younger than the
   shortest window (§6.3.1).
9. **Schema migration has never been exercised.** `SCHEMA_VERSION` has stood
   at 1 for the life of the project.

**Distribution**

10. **Unsigned**, confirmed `NotSigned`, so every user meets a SmartScreen
    warning on a 272 MB download from an unknown publisher. This is the single
    largest barrier to distribution and it is not an engineering problem.
11. **No update mechanism.** A defect shipped is a defect that stays, which
    given the defect rate documented in §10 is the risk worth weighing most.
12. **No crash reporting, irreconcilably.** `desktop.log` never leaves the
    machine, and cannot without breaking the promise the design is built on.

**Interface**

13. **Interface defects are not reachable by the test suite** in the form a
    user meets them. 130 tests pass with the layout correct and with it
    visibly broken; three layout defects in a row were each introduced by the
    fix before them, and were found by rendering the window to an image rather
    than by any assertion (§5.2.2). Structural contrast now passes SC 1.4.11 at
    3.07:1, up from 1.32:1.
14. **Untested at non-100% DPI, on small screens, and with a screen reader.**
    Keyboard focus is now verified on all eight focusable control types by
    rendering, but nothing has been tried with assistive technology.
15. **The first day is still empty**, though it now says so usefully.
    Roughly 21 hours of clean collection are required before training unlocks;
    progress is shown as a percentage rather than a static "wait" (§8.10), but
    there is no sample dataset and nothing to explore meanwhile.
16. **`torch` is now half the build.** After removing QtWebEngine it accounts
    for 351 MB of 731 MB, to run a 0.52 MB model on the CPU. A CPU-only wheel
    or an ONNX runtime for inference could plausibly halve the build again and
    has not been attempted.

## 12. Conclusion

The engineering is sound where it has been measured. Collection is cheap,
training completes in under a minute, the collector is supervised and
demonstrably restarts, the packaging failures are understood with one honest
exception, and the privacy claim holds structurally rather than by policy —
there is no network code anywhere in the source. The application runs: a
packaged build was launched and observed responding with memory plateauing at
511 MB, and the pipeline processed 92 real incidents end to end in 1.7 minutes.

**On the analytical claims, the picture is now quantified rather than
argued.** The causal layer explains 31.5% of the incidents it can test, and
yield doubles once windows exceed an hour. It was never broken; it was
starved, and the remedy is window width rather than redesign. That was first
suspected from one pair of runs at 7 and 30 minutes, and it is the survey
across 92 incidents that turned a plausible story into a rate.

**The most consequential finding was not about the model at all.** The
subsystem map — fifteen lines encoding intuition about how a laptop behaves —
rejects a quarter of everything the statistics accept, and is a strict total
order in which `network` is a sink. It forbids `network → disk`, `disk →
memory` and `disk → cpu`: a download writing to disk, a file cache consuming
memory, interrupt handling costing CPU. All three exist. One of them accounts
for the strongest relationship this system has ever measured. The map is not
wrong so much as one-directional — it models pressure flowing downhill and has
no vocabulary for the feedback that makes I/O expensive — and correcting it is
a design question about representing cycles, not a patch.

**The failures were worth more than the successes, and there is a pattern to
why.** Nine defects were found in code with a passing test suite around it.
Attribution ranked by CPU, so a sleeping allocator could never be named — in
production for the project's entire life, invisible because the CPU and disk
faults are both CPU-heavy and attributed correctly by accident of their shape.
A storage fix doubled disk usage while claiming to bound it, and its test
concealed that with a checkpoint the application never issues. A memory
injection was scored FAIL when the truth was that a bounded injection cannot
exceed normal usage on a habitually-full host, so the test was unanswerable
rather than the product broken.

In almost every case the test asked a question *adjacent* to the one that
mattered: the SQL value instead of the bytes on disk, the main file instead of
the full footprint, the function instead of the path that calls it, the
stylesheet instead of the rendered pixels. More tests would not have helped.
Adversarial review — being asked to assume the fix was wrong rather than to
confirm it was right — found five in a single pass.

**What would change the conclusions.** Running the evaluation on second
hardware, because one measurement is provably unobtainable here. Testing a
fault whose cause is not also the loudest metric, because nothing yet
distinguishes a correct causal answer from a correct severity answer. And
observing a retention purge actually run, because everything about bounded
storage is currently unit-tested and unwitnessed.

**What still limits it** is largely not engineering. Two of the three barriers
named in earlier revisions have been addressed: the application can now check
for a newer release (§8.10), and the download has fallen by a third (§6.3.3).
The one that remains is a purchase. The binaries are unsigned, so every user
meets a SmartScreen warning from an unknown publisher, and no amount of
engineering substitutes for a certificate. Crash reports still cannot reach
anyone without breaking the promise the system is built on, which is a
deliberate trade rather than an omission: a defect on a stranger's machine is
invisible unless that person sends the log.

The most durable result here is methodological. A tool that reports "no causal
chain was supported" when the data cannot support one is more useful than a
confident ranking that changes with the window — and the work of making it say
so, honestly and in the place users actually read, was larger than the work of
making it compute an answer in the first place.

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
  correct severity ranking. §8.8 states this as the most important gap.
- The attribution fix is verified by **re-running the analysis over the stored
  incident window**, not by a fresh injection. Nothing here shows it holding
  end to end on a live fault.
- Every run before the memory re-run was scored by a harness now known to have
  been **passing runs that failed attribution**. Their detection results stand;
  their attribution results were re-checked by hand rather than by the harness
  that originally reported them.
- The harness's attribution check matches any process whose name contains
  "python", and the development machine runs several. In the re-run the match
  was confirmed by hand to be the injecting worker, at 1,135 MB against 618 MB
  for the next largest — but the check as written is **weaker than the result
  it reported**.
- The idle false-positive rate was measured on a machine running Windows
  Search indexing. A quieter host would likely score lower, and a busier one
  higher.
