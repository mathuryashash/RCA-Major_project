# Resume Material

Everything from this project you can legitimately put on a CV, with the
numbers that back each claim. Nothing here is inflated — every figure was
measured, and the ones you should *not* claim are listed at the end.

---

## 1. The project line

**Title options** (pick by the role you're applying to):

- *LocalRCA — On-Device Root Cause Analysis for Windows* — ML / data roles
- *LocalRCA — Privacy-First System Diagnostics Desktop Application* — product / full-stack
- *LocalRCA — Anomaly Detection & Causal Inference Pipeline* — research / analytics

**Stack line:**
`Python · PyTorch · PySide6 (Qt6) · SQLite · statsmodels · scikit-learn · NetworkX · Plotly · PyInstaller`

---

## 2. Bullets — pick 3 or 4, matched to the job

### If the role is ML / Data Science

- Built an **LSTM autoencoder** in PyTorch for unsupervised multivariate anomaly detection over **29 system metrics**, trained per-machine on unlabelled telemetry; reconstruction error yields a **per-metric** anomaly score rather than a single scalar, enabling downstream causal analysis.
- Implemented a **Granger causality** pipeline with ADF stationarity testing, **Benjamini-Hochberg FDR correction** across ~110 hypotheses, and an F-statistic effect-size floor — designed so the system reports *no supported causal chain* rather than a spurious one.
- Engineered **model drift detection** by persisting reference reconstruction error with the artifact; measured **57× drift** on a week-old model and validated retraining restored it to **1.54×** against recent data.
- Diagnosed and corrected a **degenerate ranking failure**: with zero surviving causal edges, PageRank and outflow terms become uniform and the top two candidates separated by **0.0004**, with the leading candidate changing at every window width — reworked the reporting layer to refuse a root-cause claim in that regime.
- Derived empirical **cost models** for training and inference (per-epoch cost linear in sequence length; inference quadratic in window size), fitted to measurements and **self-calibrating against each run** on the host machine.

### If the role is Software / Backend / Systems

- Designed a **two-process architecture** (background collector + desktop GUI) sharing a **WAL-mode SQLite** store, so telemetry collection survives GUI closure and a UI crash cannot stop recording.
- Built a Windows telemetry collector sampling **29 metrics at 30s**, process activity at 300s, and allowlisted Event Log ingestion, with **singleton mutex** enforcement and first-class **collection-gap detection** so no analysis window spans a discontinuity.
- Packaged the application with **PyInstaller** into two signed-ready executables; resolved frozen-build-only failures including lazily-imported dependencies excluded by static analysis, missing standard file descriptors in windowed builds, and a **self-replicating process bug** caused by executable path resolution.
- Wrote **93 automated tests** running headless in ~1 minute against real SQLite schemas, with regression tests pinning each production defect found.
- Implemented full **install/uninstall lifecycle** — per-user startup registration, Start menu shortcut, Add/Remove Programs entry — requiring **no administrator privileges**.

### If the role is Full-Stack / Product

- Shipped a **PySide6 desktop application** with a three-stage workflow (data → training → inference), background `QThread` workers with per-stage progress reporting, and embedded interactive **Plotly** visualisations via `QWebEngineView`.
- Designed a **privacy-first data model**: all processing on-device, **zero network calls**, allowlisted event capture, opt-in redacted message storage, and a first-run consent dialog stating what is recorded, retained and how to erase it.
- Built **runtime estimators** surfacing predicted training and analysis time before execution, calibrated per-machine — predictions within **~5%** of measured runtimes.
- Authored end-user documentation (install guide, architecture, decision log) and an implementation paper reporting measured results **including negative findings**.

---

## 3. Verified metrics

Use these freely — each was measured, not estimated.

| Claim | Value |
|---|---|
| Source lines (`src/`) | ~5,570 across 42 modules |
| Automated tests | 93, headless, ~1 minute |
| Metrics collected | 29 per sample |
| Sampling cadence | 30 s system / 300 s process / 300 s events |
| Dataset observed | 10,555 samples, 243,089 process rows, 4,194 events over 13.2 days |
| Training time | 8.5 s (default) → 24.6 s (30 epochs) → ~64 s (max window) |
| Inference time | 0.2 s – 12.5 s, growing quadratically with window size |
| Incident detection | 0.6 s across 7 days of history |
| Drift correction | 57× → 1.54× after retraining |
| Estimator accuracy | 18/14/26 s predicted vs 17.5/13.2/24.6 s measured |
| Distribution | Two executables, 1,531 MB |
| Git history | 84 commits |

---

## 4. Skills this project genuinely evidences

**Machine Learning** — unsupervised anomaly detection, autoencoders, LSTMs,
sequence modelling, time-series preprocessing, train/validation splitting,
model persistence and versioning, drift detection, hyperparameter reasoning.

**Statistics** — Granger causality, ADF stationarity testing, differencing,
multiple-hypothesis correction (FDR), effect size, PageRank centrality.

**Software Engineering** — multi-process architecture, concurrency
(QThread, Qt signals), SQLite/WAL, schema migration, structured logging,
crash reporting, exception routing, dependency injection via callbacks,
93-test suite, regression-driven fixes.

**Desktop / Windows** — PySide6/Qt6, `QWebEngineView`, per-user registry
(Add/Remove Programs), Startup registration, COM shortcut creation, Windows
Event Log API via pywin32, PyInstaller freezing and hook authoring.

**Data Engineering** — time-series segmentation, gap detection, retention
policies, coverage measurement, incremental ingestion with watermarks.

**Professional practice** — privacy-by-design, consent flows, LGPL compliance
for bundled Qt, licence and third-party notice authoring, technical writing,
release packaging with checksums.

---

## 5. ATS keywords

```
Python, PyTorch, LSTM, Autoencoder, Anomaly Detection, Unsupervised Learning,
Time Series Analysis, Granger Causality, Causal Inference, Statistical
Hypothesis Testing, FDR Correction, scikit-learn, statsmodels, NetworkX,
pandas, NumPy, SQLite, WAL, PySide6, Qt6, Multithreading, Desktop Application,
Windows API, pywin32, PyInstaller, Plotly, Data Visualisation, Telemetry,
Observability, Root Cause Analysis, Privacy by Design, Unit Testing, pytest,
Git, Technical Documentation
```

---

## 6. If you have room for one line of impact

Most student projects on a CV describe what was built. This one can describe
a *finding*, which is rarer and more interesting:

> Measured that the causal layer produced no statistically supported chain on
> the majority of real incidents, traced it to insufficient window length
> against Granger's sample requirement, and reworked the reporting layer to
> distinguish "no relationship found" from "not enough data to test" — rather
> than presenting an unsupported ranking.

That single bullet demonstrates measurement, root-cause reasoning, statistical
literacy and engineering judgement at once.

---

## 7. Do NOT claim these

Putting any of these on a CV will collapse under one follow-up question.

| Don't say | Why |
|---|---|
| "95% accuracy" or any accuracy figure | **There is no ground truth and no labelled test set.** No precision or recall was ever measured. |
| "Predicts system failures" | It explains incidents *after* they occur. There is no forecasting component. |
| "Proves the root cause" | Granger causality is predictive precedence, not causation. |
| "Real-time analysis" | Sampling is every 30 seconds; analysis is manual and post-hoc. |
| "Production deployment" / "used by N users" | It has not been deployed to anyone. |
| "Scalable to enterprise fleets" | Explicitly single-machine by design. |
| "Reduced downtime by X%" | Never measured, and unmeasurable without a baseline. |

If asked "how accurate is it?", the correct and stronger answer is: *"I can't
claim an accuracy figure — there's no ground truth on a personal machine. The
honest next step is fault injection: run a known stressor and confirm it's
detected and attributed. That's the main gap."*

That answer consistently reads better than an invented number.

---

## 8. Two compact formats

### Long (2–3 lines, detailed CV)

> **LocalRCA — On-Device Root Cause Analysis** · *Python, PyTorch, PySide6, SQLite*
> Privacy-first Windows desktop application diagnosing system slowdowns and crashes
> entirely on-device. LSTM autoencoder learns per-machine normal behaviour across 29
> metrics; Granger causality with FDR correction and effect-size gating ranks root-cause
> candidates. 93 tests; measured 8.5 s training, sub-second to 12 s inference. Reports
> explicitly state when evidence is insufficient rather than asserting a cause.

### Short (1 line, space-constrained)

> **LocalRCA** — On-device Windows diagnostics: LSTM autoencoder anomaly detection over 29 metrics + Granger causal ranking with FDR correction; PySide6 desktop app, 93 tests, fully offline. *(Python, PyTorch, Qt6, SQLite)*
