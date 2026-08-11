# Library and Technology Decisions

Every dependency here earns its place or gets cut — the frozen build is
already 1.5 GB. This records what was chosen, what it was chosen over, and
where the choice has since caused pain.

---

## Storage — SQLite

**Chosen over:** flat CSV/Parquet files, DuckDB, a local Postgres.

SQLite is in the Python standard library, needs no server, no install step
and no admin rights — which matters for a tool distributed as a ZIP that a
user extracts and runs. It handles concurrent read-while-write through **WAL
mode**, which the architecture depends on: the collector writes every 30
seconds while the GUI reads on a timer.

Flat files were rejected because a partially-written CSV during a power cut
corrupts the tail, and because "give me the samples between these two
timestamps" is a query, not a file scan. DuckDB is a better analytical
engine, but the workload is small (10,555 rows after 13 days) and it is
another 50 MB in a build already too large. Postgres is absurd for a
single-user desktop tool.

**Cost:** SQLite gives no built-in encryption, so the database is plaintext.
Acceptable here — it lives in a per-user profile directory protected by
Windows ACLs, and the threat model is not "an attacker already has your
files".

---

## UI — PySide6 (Qt 6)

**Chosen over:** Tkinter, Electron, a local web app, PyQt6.

Requirements were: native Windows desktop, embed interactive charts, and run
long jobs without freezing.

- **Tkinter** ships with Python but cannot embed an interactive Plotly chart,
  and its widgets look like 2005.
- **Electron** means shipping Chromium *and* Python — worse than the current
  1.5 GB, plus an IPC layer between the UI and the ML code.
- **A local web app** (Flask + browser) was genuinely tempting and would have
  been smaller. Rejected because it needs a browser window that looks like a
  website, a port to bind, and a story for "what if the port is taken" — and
  because a diagnostic tool that opens a localhost URL feels less like an
  application people trust with system telemetry.
- **PyQt6 vs PySide6:** functionally near-identical. PySide6 is the official
  Qt binding under **LGPL**, PyQt6 is GPL-or-commercial. LGPL permits shipping
  a closed binary provided the Qt libraries stay replaceable; GPL would force
  the whole application open. PySide6 kept the licensing option open.

`QThread` gives the worker model, `QWebEngineView` embeds Plotly output.

**Cost:** QtWebEngine is most of the Qt bulk, and it brought the
`0xC0000409` windowed-build crash that took a long time to pin down.

---

## Charts — Plotly

**Chosen over:** Matplotlib, PyQtGraph.

The causal graph needs hover text on edges (lag, strength) and the timeline
needs pan and zoom across hours. Matplotlib renders a static image — fine for
a paper, poor when a user wants to inspect a spike. PyQtGraph is faster and
native but its API is awkward for annotated network graphs.

Plotly emits self-contained HTML with the JS inlined, so **the charts work
with no network access** — which the privacy promise requires absolutely.

**Cost:** each figure is ~4.4 MB of HTML written to a temp directory. Those
files outlived the process (cleaned only at `atexit`) and held metric values
outside the data directory, so `delete-all-data` walked past them. Now
cleaned explicitly.

---

## Detection model — LSTM autoencoder (PyTorch)

**Chosen over:** Isolation Forest, One-Class SVM, statistical thresholds,
Prophet, a Transformer.

See [MODEL_RATIONALE.md](MODEL_RATIONALE.md) for the full argument. In short:
the data is multivariate and temporal, "normal" is machine-specific and
unlabelled, and reconstruction error gives a **per-metric** anomaly score,
which the causal stage needs as its input.

**PyTorch over TensorFlow:** better CPU-only story, far simpler packaging,
and a training loop you can read. TensorFlow's frozen-build footprint is
worse and its lazy imports are no friendlier.

**Cost:** PyTorch is most of the 1.5 GB, and its lazy imports have caused
three separate packaging failures. On a CPU-only workload of this size, that
is a real price paid for an architecture that could arguably be smaller.

---

## Causality — statsmodels (Granger) + NetworkX

**Chosen over:** correlation only, PCMCI/tigramite, DoWhy, causal discovery
via PC/GES.

Correlation alone cannot distinguish "disk filled because logs grew" from
"logs grew because disk filled". Granger causality asks whether one series
improves prediction of another — a testable, explainable claim on time series
that suits a tool that must justify itself to a user.

**tigramite/PCMCI** is the stronger method and would handle the multivariate
confounding better. Rejected on dependency weight and because its output is
harder to explain in a report a non-specialist reads.
**DoWhy** targets interventional questions with a known causal graph — we are
trying to *discover* the graph.

`statsmodels` supplies both `grangercausalitytests` and the ADF stationarity
test needed before it. **NetworkX** builds the DAG, breaks cycles and supplies
PageRank for topology centrality — the standard library of graph work, pure
Python, no build step.

**Cost, and it is the biggest honest one:** Granger needs `max_lag × 3`
aligned samples and differencing consumes more. Real incident windows are
often too short, so the layer frequently tests *nothing*. That is a data
problem, not a library problem, but the choice of a data-hungry method made
the system's weakest input its binding constraint.

---

## System sampling — psutil (+ pynvml)

**Chosen over:** WMI queries, raw Performance Counters, `typeperf`.

psutil is one cross-platform call per metric, is well maintained, and avoids
the COM overhead of WMI — which matters when sampling every 30 seconds in a
background process meant to be unnoticeable. `pynvml` adds GPU readings when
an NVIDIA card is present and degrades quietly when it is not.

---

## Windows events — pywin32

Not really a choice: the Event Log API is Win32, and `pywin32` is the binding.
Only an **allowlist** of providers is retained (Kernel-Power 41, Application
Error 1000, disk 7/51/153, WHEA, Resource-Exhaustion 2004, and update
activity). Message text is stored only under an explicit opt-in and is
redacted first.

**Why an allowlist rather than a filter list:** a denylist fails open. Any
event family nobody thought about would be captured by default, which is the
wrong default for privacy.

---

## Preprocessing — pandas, NumPy, scikit-learn

pandas because everything is a time-indexed table and gap detection is
`.diff()` plus a `cumsum` grouping idiom. NumPy underneath. scikit-learn
purely for `MinMaxScaler` — the scaler is **saved inside the model artifact**,
because a model applied with a different scaling than it was trained under
produces confident nonsense.

---

## Packaging — PyInstaller

**Chosen over:** Nuitka, cx_Freeze, shipping a Python install, MSIX.

PyInstaller has by far the best hook ecosystem for the scientific stack —
torch, sklearn and statsmodels all have community hooks that work. Nuitka
compiles and would be faster and smaller, but its torch support is fragile.

**Cost, extensively paid:** `--onedir` means users must not move the `.exe`
out of its folder. The excludes list is derived from a *static* import
closure, so every lazily-imported dependency is a latent failure that only
appears in the packaged build — `optree`, `torch.export`, `torch._inductor`
and the missing-stdout crash all trace to this. A ~19-minute build makes each
of those expensive to diagnose.

---

## Things deliberately not used

| Not used | Why |
|---|---|
| A cloud backend | The entire premise is that nothing leaves the machine |
| Docker | Shipped a Streamlit dashboard that no longer exists; deleted |
| A database server | No install step is a feature |
| Telemetry/analytics on the app itself | Would contradict the privacy claim outright |
| An auto-updater | Not built. A real gap for a distributed app, recorded as such |
| Code signing | Costs money; absence is disclosed in the README and installer notes |

---

## The decision I would revisit first

**PyTorch for a model this small.** The autoencoder trains in 8.5 seconds on
1,701 windows. Most of a 1.5 GB download and every packaging failure but one
traces back to the deep-learning dependency. A hand-written autoencoder in
NumPy, or a classical detector, would fit the actual scale of this problem —
at the cost of the per-metric reconstruction error that makes the causal
stage possible. That trade is worth re-examining rather than assuming.
