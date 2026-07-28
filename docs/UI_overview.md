# Desktop UI Overview

The PySide6 desktop application is a local-only interface for the telemetry
collector and RCA workflow. It does not generate failures or send telemetry to
an external service.

1. The collector stores opt-in system metrics, process snapshots, and an
   allowlisted/redacted subset of Windows events in local SQLite.
2. **Stage 1** filters gaps and event-adjacent samples, then trains an LSTM
   baseline from one uninterrupted clean three-day segment.
3. **Stage 2** scores a selected observed time window, runs constrained Granger
   analysis, correlates stored events, and ranks root-cause candidates.
4. Reports contain the causal graph, anomaly times, event correlations, and
   process attribution from retained snapshots.

The model bundle includes feature order, scaler parameters, thresholds, and
weights, so inference always uses the training-time schema.
