# Repository Overview

This project performs root-cause analysis from locally collected Windows
telemetry.

- `src/telemetry/` collects opt-in metrics and Windows events into SQLite.
- `src/pipeline/engine.py` loads clean baseline data, persists model bundles,
  scores observed windows, and coordinates RCA.
- `src/models/lstm_autoencoder.py` provides the anomaly detector.
- `src/causal_inference/` applies guarded Granger tests and laptop-subsystem
  topology constraints.
- `src/desktop/` is the PySide6 workflow for model training and observed-window
  analysis.
- `src/reporting/report_generator.py` writes Markdown and JSON incident reports.

Synthetic generators, Streamlit dashboards, and Jaeger integration were
removed as part of the real-telemetry migration.
