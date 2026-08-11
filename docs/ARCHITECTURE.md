# Architecture

How LocalRCA is put together, and why the seams sit where they do.

---

## 1. The shape of the system

Two processes, one database, one direction of data flow.

```
        ┌──────────────────────────────┐
        │      RCA-Collector.exe       │   console, starts at logon
        │  ────────────────────────    │
        │  every 30s  → system metrics │
        │  every 300s → process sample │
        │  every 300s → Event Log poll │
        └──────────────┬───────────────┘
                       │ writes
                       ▼
        ┌──────────────────────────────┐
        │  %LOCALAPPDATA%\RCA\          │
        │    telemetry.db  (SQLite/WAL) │
        │    telemetry_model.pt         │
        │    reports\                   │
        │    collector.log, desktop.log │
        └──────────────┬───────────────┘
                       │ reads
                       ▼
        ┌──────────────────────────────┐
        │      RCA-Desktop.exe         │   windowed GUI
        │  ────────────────────────    │
        │  Captured Data  — coverage   │
        │  Stage 1        — training   │
        │  Stage 2        — inference  │
        └──────────────────────────────┘
```

**Why two processes rather than one.** Collection must continue when the
window is closed; a diagnostic tool that only records while you are watching
it records nothing useful. Keeping them separate also means a GUI crash
cannot take collection down with it.

That split has a sharp edge, and it drew blood. The desktop app starts the
collector if one is not running, and originally located it via
`sys.executable`. In a frozen build `sys.executable` *is* the GUI, so
"start the collector" opened a second window, which opened a third — a fork
bomb. The collector is now found by explicit binary name
(`RCA-Collector.exe`), and **refuses to launch if that sibling is absent**
rather than falling back to the running executable.

---

## 2. Module layout

```
src/
├── telemetry/            The collector. No ML, no UI.
│   ├── collector.py      Loop, consent gate, singleton mutex
│   ├── sampler.py        psutil + NVML readings
│   ├── eventlog.py       Windows Event Log ingest, allowlisted
│   ├── redaction.py      Best-effort scrub for opted-in message text
│   ├── store.py          SQLite schema, WAL, migrations
│   ├── analysis.py       Segments, gaps, readiness, incidents
│   ├── schedule.py       Startup entry, Start menu, Add/Remove Programs
│   ├── config.py         Paths, cadences, retention
│   └── __main__.py       CLI: install, status, uninstall, delete-all-data
│
├── models/
│   └── lstm_autoencoder.py    The detector
│
├── anomaly_detection/
│   ├── anomaly_scorer.py      Thresholds, scoring
│   └── ensemble_detector.py   Optional secondary detector
│
├── causal_inference/
│   ├── causal_engine.py       Granger, FDR, effect size, ranking
│   └── dynamic_graph.py       Graph refinement
│
├── reporting/
│   └── report_generator.py    Markdown and JSON reports
│
├── pipeline/
│   ├── engine.py         The seam: everything the UI and CLI both need
│   └── visualizations.py Plotly figures
│
└── desktop/              PySide6. No analysis logic.
    ├── main.py           Entry, std handles, crash logging, consent
    ├── main_window.py    Tab shell
    ├── workers.py        QThread wrappers around engine calls
    ├── consent.py        First-run disclosure
    └── views/            Captured Data, Stage 1, Stage 2, figures
```

### The one rule that keeps this honest

**`pipeline/engine.py` is the only place a pipeline phase is implemented.**
Both the desktop app and `train_and_run.py` call into it. Nothing in
`desktop/` computes anything analytical; nothing in `telemetry/` imports
torch.

That boundary is what makes the pipeline testable without a GUI, and it is
why 93 tests run headless in about a minute.

---

## 3. Data model

`telemetry.db`, SQLite in WAL mode so the GUI can read while the collector
writes.

| Table | Row | Retention |
|---|---|---|
| `samples` | One 30-second system observation, 29 metric columns | indefinite |
| `proc_samples` | Top-15 process by CPU at that moment | 30 days |
| `events` | One allowlisted Windows event | 365 days |
| `collection_gaps` | A recorded discontinuity | indefinite |
| `meta` | Consent flag, schema version, watermarks | — |

**Gaps are first-class.** A machine sleeps, reboots, and the collector dies.
Rather than pretend the series is continuous, discontinuities are detected
(`elapsed_ms` against a 1.5× cadence threshold) and every downstream consumer
works in *contiguous segments* rather than on the raw table.

This single decision propagates everywhere: training windows never span a
gap, incident windows are validated against segments, and coverage is
reported honestly rather than implied by span.

---

## 4. The three stages

### Captured Data
Reads coverage, gap counts and span. Exists because the honest answer to
"why can't I train yet" is usually "your collector has only been running a
quarter of the time", and that should be visible rather than inferred.

### Stage 1 — Baseline & Training
Gates training on **250 clean windows** existing across all clean segments.
Trains the LSTM autoencoder, records a reference reconstruction error, saves
the artifact with its scaler and feature list.

### Stage 2 — Inference
Selects an incident (detector-flagged or Event-Log-triggered), scores the
window, runs causal inference, ranks candidates, renders figures and a
report.

---

## 5. Threading

Qt's event loop must never block. All three long operations run in `QThread`
subclasses in `desktop/workers.py`, communicating by signal:

```
TrainWorker            progress(int, str) → finished_ok(payload) | failed(str)
DetectIncidentsWorker                     → finished_ok(list)    | failed(str)
InferenceWorker        progress(int, str) → finished_ok(payload) | failed(str)
```

`engine` functions take an optional `progress` callback, and the worker
passes its signal's `emit` directly. The engine stays UI-agnostic — the CLI
passes nothing — while the GUI gets per-stage updates.

**A caution learned here:** `InferenceWorker` originally stored its analysis
window as `self.start`, which overwrote `QThread.start` — the method that
launches the thread — with a `pandas.Timestamp`. Clicking Run raised
`TypeError: 'Timestamp' object is not callable`, no worker was created, and
the progress bar sat at 0% looking like a hang. Subclassing a framework
class means its attribute namespace is not yours.

---

## 6. Packaging

PyInstaller, two `--onedir` executables, ~1.5 GB, unsigned.

- `packaging/rca_desktop.spec` — the GUI, `console=False`
- `packaging/build.ps1` — drives both builds
- `packaging/excludes.txt` — ~600 modules deliberately dropped
- `packaging/hooks/hook-torch.py` — bundles Dynamo's Python sources
- `packaging/runtime_hook.py` — survives torch's missing-source imports

**The excludes list is generated from a static import closure, which is
precisely why it has bitten repeatedly.** Anything imported lazily is invisible
to it. `optree` was excluded while its `.dist-info` still shipped, so PyTorch
read a version, concluded the package was present, imported it, and failed
inside Adam's constructor — in the packaged build only. `torch.export` and
`torch._inductor` were excluded as "tracing only" when both are imported at
module scope by torch itself.

A windowed frozen build also starts with **no valid stdout/stderr**, which
crashed the app with `0xC0000409` in `Qt6Core.dll` about forty seconds in,
silently. Descriptors 1 and 2 are pointed at the null device before Qt is
imported. The mechanism is still not established — it is *not* PySide6
escalating a slot exception, which was measured and disproved.

---

## 7. Where the design is weakest

Stated plainly, because an architecture document that only lists strengths is
a sales brochure.

1. **The collector has no supervisor.** It is a Startup-folder entry; if it
   dies mid-session it stays dead until the next logon. Measured coverage was
   **27.8%** over 13.2 days, median unbroken segment **8.5 minutes**.
   Everything downstream inherits this.
2. **Causal inference rarely has enough data.** Granger needs `max_lag × 3`
   aligned samples; real incident windows often hold fewer.
3. **No ground truth**, so no measured precision or recall anywhere.
4. **1.5 GB, unsigned, no update mechanism.**
