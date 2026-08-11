# Why This Model, and Why This Approach

The reasoning behind the LSTM autoencoder, the causal layer, and the
constraints that forced both — including where the approach does not work.

---

## 1. The problem the model actually has to solve

Three properties of this setting rule out most of the obvious answers before
any model is chosen.

**There are no labels.** Nobody records "my laptop was slow at 14:32 because
Windows Update was indexing". There is no training set of faults, so anything
supervised — a classifier over known failure types — is impossible from the
start.

**"Normal" is machine-specific.** A gaming laptop at 80 °C under load is
fine; an idle office machine at 80 °C is not. A model trained on someone
else's telemetry, or on an average of many machines, would be wrong about
this one. Normal has to be learned locally, per install.

**The signal is multivariate and temporal.** 29 metrics, and what matters is
their behaviour *together over time*. A CPU spike is unremarkable; a CPU
spike with rising swap and falling disk-free is a different story.

This is anomaly detection with no negative examples, on a multivariate time
series, learned per machine.

---

## 2. Why an autoencoder

An autoencoder is trained to reconstruct its input. Train it only on clean
data and it becomes good at reproducing normal behaviour and bad at
reproducing anything else. **Reconstruction error becomes the anomaly score**,
without ever needing an example of an anomaly.

That property — learning only from normal — is what the "no labels"
constraint demands.

The second property is the one that actually decided it. Reconstruction error
is available **per metric**, not just per timestep. The model does not merely
say "something is wrong at 14:32"; it says *which of the 29 metrics* were
poorly reconstructed. That per-metric score is the direct input to the causal
stage, which needs to know which series to test against which.

A model producing only a scalar anomaly score would have left the causal
layer with nothing to work on.

---

## 3. Why LSTM rather than a dense autoencoder

A dense autoencoder over a single timestep sees a snapshot. It cannot tell a
CPU sitting at 90% for an hour from one that jumped there a second ago, and
the difference is the whole diagnosis.

The LSTM consumes a **window** of consecutive samples (default 12 = 6
minutes) and carries state across timesteps, so it models the *shape* of a
metric's behaviour. Sequential structure is exactly what the data has.

**Windows never span a collection gap.** A window bridging a two-hour sleep
would teach the model that a discontinuity is normal. Windows are built
inside contiguous segments and concatenated, which is why gap detection is
first-class in the storage layer.

---

## 4. Alternatives considered

| Approach | Why not |
|---|---|
| **Static thresholds** (CPU > 90%) | Cannot express "unusual *for this machine*", and would fire constantly on a machine that is legitimately busy. No temporal structure at all. |
| **Isolation Forest** | Genuinely strong for tabular anomalies and far cheaper. Treats rows as independent — no sequence modelling — and gives a per-*row* score, not per-metric, so the causal stage would have nothing to consume. |
| **One-Class SVM** | Scales poorly with sample count, needs careful kernel choice, and again yields one score per observation. |
| **Prophet / ARIMA** | Forecast a single series. We have 29 that matter jointly, and forecasting is not the task. |
| **Transformer autoencoder** | Better long-range modelling, but wants far more data than a single machine produces in days, and adds parameters and training time for a sequence length of 12. |
| **A supervised classifier** | No labels. Would require a fault taxonomy nobody has. |

**Isolation Forest deserves the honest note:** on pure detection quality per
unit of complexity it may well beat the autoencoder here. It was not chosen
because of the per-metric requirement, not because it would detect worse.

---

## 5. Why Granger causality on top

Detection answers *what* was abnormal. Users want *why*.

With 11 metrics flagged, listing all 11 is not a diagnosis. Something has to
distinguish the metric that moved first and drove others from the ones that
merely followed.

Granger causality asks a specific, testable question: **does the past of X
improve prediction of Y beyond Y's own past?** It is a statement about
predictive precedence, not true causation — and the reports are careful to say
so. It suits this problem because the data is exactly what it needs: regularly
sampled, multivariate time series.

### The statistical guards, and why they matter more than the test

Testing every ordered pair of 11 metrics is 110 hypotheses. At p < 0.05 you
would expect ~5 false edges by chance alone, and a graph of spurious arrows
looks more authoritative than no graph at all.

1. **Stationarity** — ADF test, differencing up to twice. Granger on a
   trending series finds structure that is only the trend.
2. **FDR correction** — Benjamini-Hochberg across all pairs.
3. **Effect-size floor** — `f / (f + n)`. With enough samples a negligible
   improvement becomes "significant"; the F-statistic gives a bounded,
   explainable measure of how much it actually helps.

**These gates are why the system usually reports nothing, and that is the
correct behaviour.** A tool that always produces a confident causal chain
would be lying most of the time.

---

## 6. Ranking

Surviving edges form a DAG (cycles broken). Candidates are scored on a
composite:

- **causal outflow** — how many metrics it drives
- **causal inflow** — how many drive it (a pure effect is not a cause)
- **temporal priority** — did it deviate first
- **anomaly severity** — how far from normal
- **event correlation** — did a Windows event coincide

plus PageRank for topology centrality.

### The failure mode this creates

**With no surviving edges, the ranking degenerates.** Every metric gets
identical graph influence and zero outflow, so the score collapses to timing
and severity alone. Measured on a real incident: top two candidates scored
**0.4620 and 0.4616** — a gap of 0.0004 — and the leading candidate *changed
at every window width tested* (`disk_free_pct` → `battery_drain_rate` →
`cpu_freq_mhz` → `disk_read_bps`).

The system was presenting an arbitrary ordering with a confidence percentage
while its own evidence section said no causal claim was supported. The report
now refuses to name a "primary root cause" when no edge survives, calls the
leading metric *correlated*, and declares the ranking arbitrary when
candidates fall within a hundredth of each other.

---

## 7. Hyperparameters, and why they are what they are

| Parameter | Default | Reasoning |
|---|---|---|
| Sampling cadence | 30 s | Fine enough to catch a spike, coarse enough to be unnoticeable. 2,880 rows/day. |
| Window size | 12 (6 min) | Long enough for shape, short enough that real incidents contain one. Adjustable 6–60. |
| Training stride | 5 | Overlapping windows multiply training data from limited history. |
| Minimum windows | 250 | Below this the model memorises rather than generalises. Implies ~21 h of clean collection. |
| Epochs | 5 (max 30) | Validation loss flattens early; 5 epochs = 8.5 s, 30 = 24.6 s. Cheap either way. |
| Staleness ratio | 2.0× | Reconstruction error doubling means "normal" has moved. |
| Granger max lag | 5 (2–10) | 5 samples = 2.5 min of propagation. Higher lags need proportionally more data. |

**Window size is the parameter with the sharpest trade-off.** Larger windows
model longer behaviour but require longer unbroken collection — and at 27.8%
coverage with a median segment of 8.5 minutes, a 60-sample window (30 min)
disqualifies most of the collected history.

---

## 8. What this approach cannot do

Stated plainly.

1. **It cannot be evaluated.** No ground truth means no precision, no recall.
   Every correctness claim is a plausibility judgement. The fix is fault
   injection — run a known stressor, confirm it is detected and attributed —
   and it is not built.
2. **It cannot explain what it never saw.** An event at 03:00 with no
   telemetry (collector not running) is unanalysable, and coverage is 27.8%.
3. **Granger is not causation.** It is predictive precedence, and both the
   report and the graph legend say so.
4. **Unobserved confounders are invisible.** If a thermal event drives both
   CPU frequency and fan behaviour and is not measured, an edge may be drawn
   between the two symptoms.
5. **The model describes one machine.** That is the design, but it means
   nothing transfers and every install starts cold.

---

## 9. If starting again

Keep the autoencoder — the per-metric error is genuinely what makes the rest
possible.

**Fix the data before touching the model.** Coverage is the binding
constraint on everything: training data, incident analysability, causal
window length, drift measurement. A supervised collector at >90% coverage
would improve results more than any change of architecture.

**Build the evaluation harness first.** Without it there is no way to tell
whether a change helped, which means every subsequent modelling decision is
made on impression rather than evidence.
