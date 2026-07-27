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
| Failure families with full RCA | all *acute* events; not slow drift | The split is acute vs drift, not subsystem — see RCA scope |
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
samples      (ts INTEGER PRIMARY KEY,      -- wall clock, seconds
              elapsed_ms INTEGER NOT NULL, -- monotonic delta from previous tick
              ...~25 REAL metric columns...,
              -- counters for attribution, in physical units
              cpu_busy_s_delta REAL,        -- core-seconds
              mem_used_bytes INTEGER,
              disk_read_bytes_delta INTEGER,
              disk_write_bytes_delta INTEGER)
proc_samples (ts INTEGER, pid INTEGER, create_time REAL, name TEXT,
              cpu_pct REAL,                 -- display only
              cpu_time_delta_s REAL,        -- core-seconds, for attribution
              rss INTEGER,
              io_read_delta INTEGER, io_write_delta INTEGER)
events       (ts INTEGER, record_id INTEGER, provider TEXT, event_id INTEGER,
              level TEXT,
              message_redacted TEXT,   -- NULL unless the user opts in
              UNIQUE(provider, event_id, ts, record_id))
meta         (key TEXT PRIMARY KEY, value TEXT)
```

`(pid, create_time)` is the process identity, not `pid` alone — Windows reuses
PIDs aggressively and a reused PID would otherwise fabricate a huge delta.

I/O columns store **per-tick deltas**, not the cumulative counters psutil
returns. Cumulative counters reset when a process restarts, which would produce
a large negative delta; the collector computes the delta at write time and
writes 0 when the previous value is missing or larger than the current one.

#### Gap semantics

A gap is any consecutive pair of `samples.ts` more than **45 s** apart, i.e.
1.5x the 30 s cadence. The earlier 90 s figure was wrong: one dropped tick
produces a 60 s interval, which would have passed a 90 s test and let a model
window silently span missing data.

The rule is defined relative to cadence — `gap_threshold = 1.5 x cadence` — so
it stays correct if the cadence is ever changed.

No training or inference window may span a gap. Sleep, hibernate, shutdown,
collector crashes, and single dropped ticks all present identically, which is
the desired behaviour.

#### Rate metric calculation

Rate metrics (`disk_read_bps`, `net_sent_bps`, `swap_in_rate`,
`battery_drain_rate`, ...) are computed as `Δcounter / elapsed_ms` using
`time.monotonic()`, **not** the assumed 30 s cadence. A delayed or slow tick
would otherwise inflate every rate on that row. `elapsed_ms` is persisted so the
computation is auditable after the fact.

The first tick after collector start has no predecessor: all rate columns are
written `NULL` and that row is excluded from training windows. Same for the
first tick after any gap.

#### Single-collector lock

The collector acquires an exclusive lock (named mutex, released on exit) before
its first write. A second instance exits immediately rather than double-writing
the same timestamps. Without this, a stale Task Scheduler entry plus a manual
run would produce duplicated rows and doubled rates.

| Table | Rate | Year 1 | Steady state |
|---|---|---|---|
| `samples` | 2,880 rows/day | ~263 MB | grows, never purged |
| `proc_samples` | ~8,100 rows/day | ~206 MB | **~17 MB** (30-day purge) |
| `events` | hundreds/day | ~10 MB | grows slowly |

The `proc_samples` rate is measured, not assumed: "top 15 by CPU union top 15 by
RSS" yields **28.2 rows per tick** on a real machine (max 29 of a theoretical
30) — the two lists barely overlap. At 288 ticks/day that is ~8,100 rows/day,
before burst-mode ticks.

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
| Context (stored, excluded from the model) | `on_battery`, `user_idle_sec`, `foreground_app` (executable name only — never window titles, see Privacy) |

Temperature is deliberately absent: `psutil.sensors_temperatures()` returns
nothing on most Windows laptops. `cpu_freq_ratio` (current ÷ max) is the
reliable throttle proxy and needs no extra dependency.

### Per-process (`proc_samples`, every 5 min)

Top 15 processes by CPU union top 15 by RSS — measured at ~28 rows per tick,
since the two lists barely overlap. Bursts to every 30 s while
`cpu_pct > 80`, `mem_pct > 85`, or `disk_busy_pct > 80` — free when the machine
is healthy, dense exactly when attribution will be read.

### Events (`events`, polled every 5 min)

Read via `win32evtlog` from the System and Application logs using a stored
bookmark. System and Application are readable without elevation; the Security
log requires it and is not used.

**The bookmark must advance in the same SQLite transaction that inserts the
events.** A bookmark stored separately, or committed in a second transaction,
guarantees a bug: a crash between the two operations either replays events
already recorded or drops events never recorded, depending on the order. Since
the bookmark lives in `meta` in the same database, one transaction covers both:

```
BEGIN
  INSERT INTO events ...        -- the batch just read
  UPDATE meta SET value = ?     -- WHERE key = 'evtlog_bookmark_system'
COMMIT
```

Events also carry a uniqueness constraint on `(provider, event_id, ts,
record_id)` so a replay after an unclean shutdown is idempotent rather than
duplicating rows.

**Bookmark invalidation.** A bookmark becomes unusable if the log is cleared,
wraps past the bookmarked record, or the channel is recreated. On `EvtSeek`
failure the collector does not silently restart from the beginning (which would
re-ingest the entire retained log as if newly observed) nor silently skip. It
resets the bookmark to the current end of log, and writes a
`collection_gap` marker row recording the channel and the interval that was
lost. Incident analysis touching an interval containing such a marker reports
that event coverage is incomplete for that window.

| Purpose | Provider / ID |
|---|---|
| Unexpected shutdown | Kernel-Power 41 |
| Application crash | Application Error 1000 |
| Application hang | Application Hang 1002 |
| Disk fault | `disk` 7, 51, 153 |
| Hardware error | WHEA-Logger |
| Resource exhaustion | Resource-Exhaustion-Detector 2004 |
| Change events (deployment analogue) | WindowsUpdateClient, MsiInstaller |

## Privacy and data retention

This design continuously records what the user is doing on their own machine and
keeps it indefinitely. Two fields carry real sensitivity, and they are the two
most useful ones:

- **`foreground_app`** — reveals activity patterns: which applications, when,
  for how long.
- **Event Log `message`** — free text routinely containing file paths, usernames,
  installed software, and occasionally URLs or document names.

Process names in `proc_samples` also disclose installed software.

**Minimisation, applied at write time — the collector never stores the raw form:**

| Field | Rule |
|---|---|
| `foreground_app` | Executable name only (`chrome.exe`). **Window titles are never captured** — they are the field that leaks document names, URLs, and message contents. |
| Event `message` | **Not stored at all by default.** See below. |
| Event providers | Allowlist only (the table above). Everything else is discarded at read time, not filtered later. |
| `user_idle_sec` | Duration only, never input content. |

**Event message text is opt-in, not redacted-by-default.** Everything RCA
actually needs from an event — correlation in time, and what kind of failure it
was — comes from `provider`, `event_id`, `level`, and `ts`. The free-text
message adds diagnostic colour and carries essentially all of the privacy risk,
so the default is to discard it at read time.

If the user opts in, messages are truncated to 512 chars and passed through
redaction covering:

- user profile paths on any drive — `[A-Za-z]:\Users\<name>\` → `<drive>:\Users\<redacted>\`
- UNC paths — `\\server\share\...` → `\\<redacted>\`
- URLs — `http(s)://...` → `<url redacted>`
- email addresses
- the current username wherever it appears literally

**Residual risk is stated plainly in the opt-in dialog:** regex redaction is
best-effort and cannot catch application-specific identifiers, document names
embedded in error strings, or paths in formats not listed above. Anything that
survives redaction will appear in exported reports. Users who need certainty
should leave message capture off, which is the default and costs only report
readability.

**No network egress.** The collector opens no sockets and the app makes no
outbound requests. All data stays in one local file. This is an explicit
non-goal, not merely an omission — no telemetry, no crash reporting, no update
check.

**Storage and access.** `%LOCALAPPDATA%\RCA\telemetry.db`, created with a
user-only ACL.

**Retention and deletion.**

| Data | Default retention |
|---|---|
| `samples` | Indefinite (needed for retraining breadth), configurable |
| `proc_samples` | 30 days |
| `events` | 1 year |

The app provides **Delete all collected data**: stops the collector, deletes the
database and all trained models, and restarts collection from empty. It also
provides **Disable collection**, which unregisters the Task Scheduler entry.

**Disclosure.** On first run the app states exactly what is collected, where it
is stored, that it never leaves the machine, and how to delete it. The collector
is not registered until the user acknowledges this. Consent is a precondition of
collection, not a setting discovered afterwards.

**Exported reports are the real egress path.** The Markdown and JSON reports
contain process names and redacted event text, and reports are the artifact a
user actually shares — with a colleague, in a bug tracker, in a submission. The
export dialog states what the file contains before writing it.

## Baseline and retraining

A bad event is an *outcome*. The pathology that produced it is in the minutes
**before** the event, so excluding only windows that overlap the event timestamp
would train the model on exactly the degradation it needs to detect. Exclusion
therefore uses an asymmetric buffer around each bad event:

```
bad event at T  ->  exclude [T - 60 min, T + 15 min]
```

The long pre-buffer captures the lead-up; the short post-buffer captures
recovery and reboot settling.

Training data is all retained history **except**:

- any window intersecting `[T - 60 min, T + 15 min]` for a bad event T (crash,
  hang, unexpected shutdown, disk error, WHEA, resource exhaustion),
- any window intersecting `[start - 10 min, end + 10 min]` for a **confirmed
  detector-discovered incident** (see circularity bound below),
- any window containing a gap (>45 s interval), and
- any window containing a `NULL` rate row (first tick after start or gap).

The 99th-percentile reconstruction-error threshold absorbs residual
contamination.

**Circularity bound.** Excluding detector-discovered incidents means the model's
own output shapes its next training set, which if unbounded makes it
progressively narrower and more trigger-happy — each retrain would find the
previous notion of normal even more normal. Three constraints:

1. The **first** training run excludes only Event-Log-derived windows. The
   detector has produced nothing to exclude yet, and this anchors the baseline
   to external evidence rather than to itself.
2. Only incidents at confidence High or above are excluded on retrain.
3. If total exclusions would exceed **20%** of retained history, exclusion stops
   at the 20% bound, dropping the lowest-severity candidates first, and the app
   warns that the machine has been unhealthy often enough that the baseline may
   be unrepresentative.

The excluded fraction is recorded in the model artifact so a suspicious model
can be diagnosed after the fact.

Retraining:

- First train when ≥3 days of clean baseline exist — 8,640 samples, which at
  window 60 / stride 5 yields ~1,716 training windows. Until then Stage 1 shows
  "collecting — N days remaining".
- On demand from Stage 1.
- **Staleness alarm:** if the rolling 7-day median reconstruction error drifts
  more than 2x from its value at training time (`reference_recon_error` in the
  model artifact), warn that the model no longer matches current usage and offer
  a retrain. This is drift detection on the error signal itself — no new model.

  This is a **new** component, not a reuse. An earlier draft of this spec
  described `src/models/concept_drift_handler.py` as an existing seam; that was
  wrong. It imports `PrometheusDataIngestion` and `DeploymentEventListener`
  (both deleted here), is built around a deployment soak period that has no
  laptop meaning, and fabricates its retraining data at line 110 with
  `np.random.normal(0.6, 0.1, ...)` in the class method itself — not in a demo
  block. It is deleted, not adapted.
- Retraining uses the full retained history minus the exclusions above. Subject
  to the circularity bound, this broadens the model over time; the 20% cap is
  what stops it narrowing instead.

### Model artifact

The model is **not** a bare `.pt` file. `torch.save` of a `state_dict` records
none of the preprocessing needed to score data compatibly, and the scaler is
currently refit on whatever data is at hand — so after a restart, a schema
change, or a grown baseline, the same model would silently score against
different scaling. That produces plausible-looking, wrong anomaly scores, which
is the worst failure mode available.

A model is a versioned bundle, written atomically (temp file then rename):

| Field | Why |
|---|---|
| `schema_version` | refuse to load against an incompatible `samples` schema |
| `feature_order` | column order is positional in the tensor; a reorder silently corrupts scoring |
| `scaler_params` | per-feature min/max from `MinMaxScaler`, persisted not refit |
| `thresholds` | per-metric calibrated thresholds |
| `window_size`, `stride`, `cadence_s` | a model trained at 30 s cadence is invalid at 60 s |
| `training_range` | first and last `ts` used |
| `excluded_fraction` | from the circularity bound |
| `reference_recon_error` | median at training time; the staleness alarm compares against it |
| `torch_version`, `created_at`, `model_id` | provenance |

Load refuses, with an explicit message, when `schema_version`,
`feature_order`, or `cadence_s` disagree with the current store. It does not
silently coerce. Stage 2 stays disabled until a compatible model exists.

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
                                (causal analysis extends to
                                 t-60min if gap-free — see
                                 statistical guardrails)
drop runs <3 windows
  (min duration, not the gap rule)
  ↓                               ↓
        └──────── Incident ────────┘
   {start, end, peak_severity, trigger, metrics[]}
```

The 3-window minimum suppresses single-sample noise; the 5-minute merge stops
one episode fragmenting into several reports.

The event-triggered path is what makes crash diagnosis work at all: a BSOD
produces no gradual metric anomaly — the machine stops — so a detector-only
design would miss it. The event defines the window and RCA asks what was
abnormal beforehand — 30 minutes for attribution and reporting, extended to 60
where gap-free data exists, because Granger needs the sample count (see
statistical guardrails).

#### RCA scope by failure family

The scope line is **acute versus slow drift**, not subsystem. Anything that
produces a bounded incident window gets full treatment, regardless of which
family it belongs to:

| Failure | Trigger | Treatment |
|---|---|---|
| Performance degradation | detector | Full: mechanism + attribution |
| App crash / hang | event 1000, 1002 | Full: mechanism + attribution |
| Unexpected shutdown / BSOD | event 41, WHEA | Full: mechanism + attribution |
| Disk fault | event 7, 51, 153 | Full: mechanism + attribution |
| Resource exhaustion, **acute** | event 2004 | Full: mechanism + attribution |
| Resource exhaustion, **slow drift** | — | **Out of scope.** Collected and threshold-flagged only; no drift detector |
| Battery / power | — | Collected and threshold-flagged; appears in mechanism chains via `POWER`, but has no per-process signal so attribution is skipped |

This resolves the apparent conflict with the decision table: disk faults and
acute exhaustion *are* fully analysed, because an event gives them a window.
What is deferred is detecting the *gradual* version — a disk filling over three
weeks, or a memory leak with a rising floor — which needs a trend detector that
a 30-minute window cannot provide and which should be tuned against real
history rather than guessed.

### Real events replace the fabricated one

`engine.run_causal_inference` currently invents a "deployment" 20 minutes before
every incident, and that fiction feeds the ranker's `event_correlation` term.
Real events fill the same slot with the same `events_df` shape;
`EventCorrelator.correlate()` is unchanged. Windows Update, driver installs, and
MSI installs are the genuine laptop analogue of a code deployment.

### Statistical guardrails on Granger inference

The topology prior removes *implausible* edges. It does nothing about *spurious*
ones, and the sample counts here are small enough that spurious edges are the
default outcome without explicit correction.

An event-triggered window is 35 minutes — **70 samples** at 30 s cadence, 69
after the differencing in `_make_stationary`. With `k` anomalous metrics and
`max_lag` lags the pipeline runs `k(k-1) x max_lag` tests: for 8 metrics at lag
5 that is **280 tests**, of which roughly **14 will pass p<0.05 by chance
alone**. Those false edges then feed a ranker whose dominant term is out-degree.

Four guardrails, applied in order:

1. **Minimum observations.** Require `n >= max(30, 10 x (max_lag + 1))` usable
   samples after differencing, with no gap in the window. At `max_lag=5` that is
   60. Event-triggered windows therefore extend to **60 minutes** (120 samples)
   where gap-free data exists, rather than the bare 35. If the requirement
   cannot be met, no causal inference is attempted for that incident.
2. **Lag bounded by sample count.** `max_lag = min(5, floor(n/10))`, so a short
   window automatically tests fewer lags instead of overfitting.
3. **Multiple-testing correction.** Benjamini–Hochberg FDR at `q=0.05` applied
   **once across the entire pair x lag test set**, not per pair. Raw p<0.05 is
   never used as the edge criterion.
4. **Effect size floor.** A surviving edge must also reduce the restricted
   model's residual variance by **>=5%**. Significance on 120 samples is easy;
   a lag that explains almost none of the variance is not a mechanism.

Only the best surviving lag per ordered pair contributes an edge, so one pair
cannot inflate out-degree by appearing at five lags.

**If nothing survives, say so.** When no edge passes all four gates the report
states **"no supported causal chain"** and presents the anomalous metrics ranked
by severity alongside process attribution, explicitly labelled as correlation
without a causal claim. This is a common and legitimate outcome for short
incidents, and it is a better answer than a ranked list of statistical noise.

### Topology prior

Every metric maps to a subsystem, and only physically plausible directed edges
survive Granger:

The prior is defined at **subsystem** level, not metric-pair level, so it is
exhaustive by construction. Enumerating metric pairs would leave gaps every time
a column is added, and a metric absent from the table gets pruned into
isolation — which, as shown below, actively distorts ranking.

Every modelled metric maps to exactly one subsystem:

| Subsystem | Metrics |
|---|---|
| `LOAD` | `process_count`, `thread_count` |
| `CPU` | `cpu_pct`, `cpu_pct_max_core`, `cpu_freq_mhz`, `cpu_freq_ratio` |
| `MEM` | `mem_pct`, `mem_available_mb` |
| `SWAP` | `swap_pct`, `swap_in_rate`, `swap_out_rate` |
| `DISK` | `disk_read_bps`, `disk_write_bps`, `disk_busy_pct`, `disk_free_pct` |
| `NET` | `net_sent_bps`, `net_recv_bps` |
| `POWER` | `battery_pct`, `battery_drain_rate`, `power_plugged` |

Adding a metric column requires adding it here; a startup assertion fails if any
modelled feature has no subsystem, so the mapping cannot silently go stale.

Allowed directed edges between subsystems:

```
LOAD  ──> CPU        more runnable work raises utilisation
LOAD  ──> MEM        more processes consume memory
MEM   ──> SWAP       pressure forces paging
SWAP  ──> DISK       paging is disk traffic
DISK  ──> CPU        io wait presents as cpu time
NET   ──> DISK       downloads land on disk
POWER ──> CPU        power-saving and thermal limits throttle frequency
CPU   ──> POWER      sustained load drains battery
```

**Intra-subsystem edges are always allowed** — `cpu_pct → cpu_freq_ratio` and
`mem_pct → mem_available_mb` are legitimate and both endpoints share a
subsystem. A metric edge survives if it is intra-subsystem, or if its
subsystem pair appears above.

So `net_recv_bps → mem_pct` (NET → MEM) is pruned however well it fits
statistically, while every collected metric still has somewhere to attach.

**Isolated nodes are dropped from ranking, not scored.** After pruning, any node
with in-degree 0 *and* out-degree 0 is removed from the graph before
`RootCauseRanker.rank()` runs. This is not cosmetic: `causal_inflow` is computed
as `1.0 - in_degree/max_in` ([causal_engine.py:370](src/causal_inference/causal_engine.py#L370)),
so an isolated node scores **1.0** — the maximum — on a 20%-weighted term, and
pruning would promote precisely the metrics it disconnected. Such metrics are
still listed in the report as "anomalous, no causal linkage", which is
informative without polluting the ranking. This reuses the existing
`DynamicGraphGenerator.refine_causal_graph()` seam; the Jaeger service lookup is
replaced by a static subsystem adjacency table. Same interface, no new
component.

Without this, the 40%-weighted `causal_outflow` term crowns whichever metric has
the most outgoing edges — on a laptop always CPU or temperature, because
everything is thermally coupled. Physically true, diagnostically useless.

### Process attribution

The metric graph explains mechanism; attribution answers who.

**Percentages require unit-compatible numerator and denominator.** Process CPU
percent can exceed 100 on a multicore machine while system CPU percent cannot;
process RSS is bytes and cannot be divided by `mem_pct`; process I/O is bytes
and cannot be divided by `disk_busy_pct`. Attribution therefore never uses the
percentage columns. Both sides are compared in a shared physical unit, which
requires the collector to store the underlying counters:

| Resource | Process numerator | System denominator | Shared unit |
|---|---|---|---|
| CPU | `Σ cpu_time_delta_s` (user+system) | `cpu_busy_s_delta` | core-seconds |
| Memory | `Δrss` | `Δmem_used_bytes` | bytes |
| Disk | `io_read_delta + io_write_delta` | `disk_read_bytes_delta + disk_write_bytes_delta` | bytes |
| Network, Power | — | — | no per-process signal; attribution skipped, mechanism only |

The percentage columns (`cpu_pct`, `mem_pct`, `disk_busy_pct`) remain in
`samples` for the model and for display; the counter columns above are added
alongside them specifically so attribution has aligned units.

Procedure:

1. Take the top-ranked metric, map its subsystem to a resource above.
2. Pull `proc_samples` for the incident window and the 30 minutes preceding it.
   Process identity is `(pid, create_time)`, never `pid` alone.
3. For each process, baseline = median of its pre-window samples, peak = max
   within the window, `delta = peak - baseline` (CPU and disk use summed
   deltas over the window rather than a peak, since they are flow quantities).
4. Rank by share of the **system delta in the same unit**.
5. Report **"diffuse — no single process responsible"** if the top process
   explains <30%.

**Reconciliation check.** Per-process sums do not have to equal the system
figure, and can legitimately exceed it: RSS double-counts pages shared between
processes, and per-process I/O counters include logical reads served from cache
that never reached the disk. So before any percentage is displayed:

```
ratio = Σ(observed clamped deltas) / system_delta
```

- `0 < ratio ≤ 1.2` — percentages shown, remainder line included.
- `ratio > 1.2` or `system_delta ≤ 0` — **no percentages are displayed at all.**
  The result is labelled "attribution unreconciled" and shows absolute
  per-process deltas ranked, with the system figure alongside.

A ranked list of absolute deltas is still useful. A percentage that exceeds
100%, or one computed against a denominator that does not measure the same
thing, is worse than no percentage — it looks authoritative and is wrong.

Rule 5 matters: memory pressure from forty browser tabs is genuinely diffuse,
and a system that always names a culprit will confidently name the wrong one.

**Edge cases, all of which otherwise corrupt the percentages:**

| Case | Handling |
|---|---|
| Process started during the incident | No pre-window samples, so baseline = 0 and the full peak is its delta. Marked `[started]`. |
| Process exited during the incident | Delta measured to its last observation. Marked `[exited]` — its contribution is a lower bound. |
| PID reused | Different `create_time` makes it a different process. Without this a reused PID fabricates a huge delta. |
| Negative delta (process released memory) | Clamped to 0. A process freeing memory is not a cause of memory pressure, and negatives would inflate everyone else's share. |
| Cumulative I/O counters | Already stored as per-tick deltas by the collector; counter resets on process restart are written as 0, not negative. |
| Process outside top-N | Unobserved. Never silently ignored — see remainder. |

**Unattributed remainder.** Shares are computed against the system-level metric
delta, so observed processes and the remainder sum to 100%:

```
remainder = system_delta - Σ(observed clamped deltas)
```

The remainder is displayed as its own line. If it exceeds **40%**, attribution
confidence is reported as low and the mechanism chain is presented as the
primary result. This is what stops "chrome.exe, 78% of delta" being asserted
when the real denominator was only the fifteen processes that happened to be
sampled.

At 5-minute cadence a 6-minute incident yields ~2 snapshots, which is thin.
Burst mode (30 s under load) is what makes attribution usable in practice, and
incidents that never triggered a burst are marked as coarsely sampled.

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
| `models/concept_drift_handler.py:110` | **`np.random.normal` in the class method, not a demo block** | delete file; replaced by the staleness monitor |
| `anomaly_detection/dimensionality_reduction.py:147` | `np.random` in `__main__` demo | delete block |
| `causal_inference/granger_causality.py:130-132` | `np.random` in `__main__` demo | delete block |
| `causal_inference/pc_algorithm.py:93-95` | `np.random` in `__main__` demo | delete block |
| `tests/test_pipeline_engine.py` | 3 tests call `generate_data` | rewrite against fixture |
| `models/lstm_autoencoder.py:178` | stale print string | edit |

Dead once real telemetry lands, all server-infrastructure sources with no laptop
meaning: `jaeger_connector.py`, `deployment_listener.py`,
`cloudwatch_connector.py`, `prometheus_connector.py`.

**Verification of this inventory must grep for two things, not one.** Searching
for `SyntheticMetricsGenerator|generate_data|inject_failure` finds the obvious
sites but misses code that fabricates data with raw `np.random` — which is how
`concept_drift_handler.py:110` was initially missed. The completeness check is:

```
grep -rn "SyntheticMetricsGenerator|synthetic_generator|generate_data|inject_failure" src/
grep -rn "np\.random|numpy\.random|torch\.randn|torch\.rand\(" src/
```

Both must return empty (outside tests using recorded fixtures) before the
no-synthetic-data requirement is met.

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
| Laptop slept mid-window | gap >45 s (1.5x cadence) splits the series; no window spans it |
| Single dropped tick | 60 s interval counts as a gap; window is split, not bridged |
| Second collector launched | exits on the mutex; no double-writing |
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

- gap detection splits windows across a real sleep gap, **and across a single
  dropped tick** (60 s interval — the case the original 90 s threshold missed)
- rate metrics use measured `elapsed_ms`, not assumed cadence: a synthetic
  delayed tick must not inflate the computed rate
- first tick after start and after a gap writes `NULL` rates and is excluded
- baseline filter excludes the full `[T-60min, T+15min]` buffer around a seeded
  bad event, not merely the overlapping window
- circularity bound caps exclusions at 20% of history
- model artifact refuses to load when `feature_order`, `cadence_s`, or
  `schema_version` disagree with the store
- incident segmentation merges runs <5 min apart and drops runs <3 windows
- every modelled feature has a subsystem (startup assertion)
- topology prior prunes a known-implausible edge, and isolated nodes are dropped
  from ranking rather than scored 1.0 on `causal_inflow`
- attribution: PID reuse with differing `create_time` is treated as two
  processes; negative deltas clamp to 0; observed shares plus remainder sum to
  100%; the <30% case reports "diffuse"; the >40% remainder case reports low
  confidence
- attribution units: a fixture where per-process RSS sums above system used
  bytes (shared pages) yields `ratio > 1.2` and produces **no percentages**,
  only ranked absolute deltas
- event ingestion is transactional: killing the process between insert and
  bookmark advance leaves neither a duplicate nor a lost event; replaying the
  same batch is idempotent via the uniqueness constraint
- an invalidated bookmark writes a `collection_gap` marker rather than
  re-ingesting the whole log
- Granger guardrails: a window below the minimum observation count attempts no
  inference; BH-FDR is applied across the whole test set; an edge that is
  significant but reduces residual variance <5% is dropped; when nothing
  survives the report says "no supported causal chain"
- event message text is absent from the store unless opted in; with opt-in,
  redaction covers non-C: drive letters, UNC paths, and URLs

## Explicitly out of scope

- Drift detector for **slow** resource exhaustion (disk filling over weeks,
  memory leak with a rising floor). Acute exhaustion (event 2004) is fully in
  scope. Data is collected; the drift detector gets its own spec once real
  history exists to tune thresholds against, rather than being guessed blind.
- Regime-conditioned modelling. Context columns are collected so this stays
  possible; enabling it is a modelling change, not a collection change.
- Always-on live monitoring. The architecture keeps it available as a second
  consumer of the store.
- Battery and power causal analysis. Collected and threshold-flagged only.
- Cross-machine or fleet analysis. Single machine, single store.
