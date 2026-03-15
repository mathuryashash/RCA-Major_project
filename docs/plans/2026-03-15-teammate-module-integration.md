# Teammate Module Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all three teammates' modules (Utsav M1B/M2B, Aditya M3, Shresth M4) with full infrastructure stack and integrate into the existing RCA pipeline.

**Architecture:** 
- Utsav (M1B/M2B): Prometheus metric scraping → Redis Streams buffer → TimescaleDB persistence → Z-score normalization → 50%-overlap windowing → HDF5 arrays
- Aditya (M3): LogBERT (DistilBERT fine-tuned on log corpus) for semantic log anomaly detection + Temporal Transformer for multi-step metric forecasting + Unified anomaly scoring (alpha=0.80 blend)
- Shresth (M4): Benjamini-Hochberg FDR correction on Granger tests + PC Algorithm (causal-learn) for DAG orientation + KHBN (pgmpy) Bayesian network with topology prior + 5-factor RCA scoring formula

**Tech Stack:** Python 3.11, TimescaleDB 2.x (PostgreSQL 15), Redis 7, Prometheus, HuggingFace Transformers (DistilBERT), PyTorch, causal-learn, pgmpy, h5py, Docker Compose

---

## Task 1: Synthetic Metric Generator (Foundation)

**Why first:** Before building metric ingestion, we need synthetic metric data that correlates with our failure scenarios. This parallels how `generate_failures.py` creates synthetic logs.

**Files:**
- Create: `scripts/generate_metrics.py`
- Create: `data/metrics/` directory

**What it does:**
- For each of the 10 failure scenarios, generates correlated time-series metrics (CPU, memory, disk, network, query latency, connections, etc.)
- Outputs CSV files per scenario + a combined metrics file
- Metrics have realistic baseline patterns (diurnal, weekly) with injected anomalies that correlate with the log failure timestamps
- Exposes metrics via a Prometheus-compatible `/metrics` endpoint for scraping

---

## Task 2: TimescaleDB Schema + Redis Setup (Infrastructure)

**Files:**
- Create: `infra/init-db.sql` — TimescaleDB schema (metrics hypertable, events table)
- Create: `infra/redis.conf` — Redis config for Streams
- Create: `config/prometheus.yml` — Prometheus scrape config
- Modify: `docker-compose.yml` — Enable TimescaleDB, Redis, Prometheus services
- Modify: `config/config.yaml` — Add Redis config

---

## Task 3: Metric Ingestion Module (Utsav M1B)

**Files:**
- Create: `src/ingestion/metric_ingest.py` — PrometheusCollector + CloudWatchCollector + MetricIngestionService
- Create: `src/ingestion/redis_buffer.py` — Redis Streams buffer layer
- Create: `src/ingestion/timescaledb_store.py` — TimescaleDB writer
- Create: `tests/unit/test_metric_ingest.py`

**Key interfaces:**
- `PrometheusCollector.scrape() -> List[MetricPoint]`
- `RedisBuffer.push(metrics)` / `RedisBuffer.consume(batch_size) -> List[MetricPoint]`
- `TimescaleDBStore.write_batch(metrics)` / `.query_range(start, end, metric_names) -> pd.DataFrame`

---

## Task 4: Metric Preprocessing Module (Utsav M2B)

**Files:**
- Create: `src/preprocessing/metric_preprocessor.py` — missing data handling, z-score normalization, windowing
- Create: `tests/unit/test_metric_preprocessor.py`

**Key interfaces:**
- `MetricPreprocessor.fill_gaps(df) -> df` (forward-fill <5min, interpolate 5-30min, sentinel >30min)
- `MetricPreprocessor.normalize(df) -> df` (rolling 7-day z-score)
- `MetricPreprocessor.create_windows(df, window_size=60, overlap=0.5) -> np.ndarray` shape (batch, 60, n_metrics)
- `MetricPreprocessor.save_hdf5(windows, path)` / `.load_hdf5(path) -> np.ndarray`

---

## Task 5: LogBERT Anomaly Detector (Aditya M3 — Log Stream)

**Files:**
- Create: `src/models/logbert.py` — DistilBERT fine-tuned log anomaly detector
- Create: `tests/unit/test_logbert.py`

**Key interfaces:**
- `LogBERTDetector.__init__(model_name='distilbert-base-uncased', max_seq_length=128)`
- `LogBERTDetector.train(log_templates: List[str], epochs=3)` — fine-tunes on healthy log templates with masked token prediction
- `LogBERTDetector.score(log_templates: List[str]) -> np.ndarray` — anomaly score [0,1] per template based on prediction loss
- Uses HuggingFace Transformers + PyTorch
- Fallback: if GPU not available, uses CPU with smaller batch size

---

## Task 6: Temporal Transformer (Aditya M3 — Metric Stream)

**Files:**
- Create: `src/models/temporal_transformer.py` — Multi-head attention for metric forecasting
- Create: `tests/unit/test_temporal_transformer.py`

**Architecture per PRD:**
- 4 attention heads, d_model=128, 2 encoder layers
- Forecasting horizon = 10 timesteps
- Anomaly score = normalized MAE between forecast and actuals
- Secondary signal to LSTM AE — catches step-changes autoencoders miss

**Key interfaces:**
- `TemporalTransformer.__init__(n_features, d_model=128, n_heads=4, n_layers=2, forecast_horizon=10)`
- `TemporalTransformer.train(windows: np.ndarray, epochs=20)`
- `TemporalTransformer.score(window: np.ndarray) -> np.ndarray` — anomaly score per metric

---

## Task 7: Unified Anomaly Scorer + MLflow (Aditya M3)

**Files:**
- Create: `src/models/unified_scorer.py` — combines TF-IDF, LogBERT, LSTM AE, Temporal Transformer scores
- Modify: `src/models/deep_learning.py` — update LSTM AE encoder to 128→64 per PRD
- Create: `tests/unit/test_unified_scorer.py`

**Formula (PRD FR-19):** `score = alpha * model_score + (1 - alpha) * rarity_prior`
- alpha = 0.80 from config
- rarity_prior for logs: level weights (DEBUG=0.1, INFO=0.2, WARN=0.5, ERROR=0.8, CRITICAL=1.0)
- Threshold: score > 0.5 = anomalous

**MLflow integration:**
- Log training configs, data hashes, loss curves, precision/recall/F1

---

## Task 8: Enhanced Causal Inference — FDR + PC Algorithm (Shresth M4)

**Files:**
- Modify: `src/models/causal_inference.py` — add BH-FDR correction + PC Algorithm + majority-vote edge weighting
- Create: `tests/unit/test_enhanced_causal.py`

**Enhancements:**
1. **Benjamini-Hochberg FDR** on Granger p-values (replacing raw alpha threshold)
2. **PC Algorithm** via `causal-learn` for DAG orientation  
3. **Majority-vote** combination: edge present if both Granger + PC agree, weight = average
4. **Cycle breaking:** remove weakest-weighted edge if cycles detected

---

## Task 9: KHBN — Knowledge-Informed Hierarchical Bayesian Network (Shresth M4)

**Files:**
- Create: `src/models/khbn.py` — Bayesian network with topology prior using pgmpy
- Create: `config/service_topology.yaml` — service dependency DAG (structural prior)
- Create: `tests/unit/test_khbn.py`

**What KHBN does:**
- Takes service dependency topology as structural prior
- Fits a BayesianNetwork using pgmpy with anomaly evidence
- Posterior inference re-ranks root cause candidates
- Suppresses spurious causal edges that violate known topology

---

## Task 10: 5-Factor RCA Scoring Formula (Shresth/Yashash)

**Files:**
- Modify: `src/models/causal_inference.py` — replace simple PageRank with full formula
- Create: `tests/unit/test_rca_scoring.py`

**Formula (PRD Line 454):**
```
rca_score = 0.35 * pagerank_score 
          + 0.25 * temporal_priority 
          + 0.20 * anomaly_score 
          + 0.10 * rarity_prior 
          + 0.10 * event_bonus
```

---

## Task 11: Pipeline Integration

**Files:**
- Modify: `src/pipeline.py` — integrate all new modules into the pipeline flow
- Modify: `src/api/server.py` — add metric-related endpoints

**New pipeline flow:**
1. Read logs (M1A — existing)
2. **Read metrics from TimescaleDB (M1B — new)**
3. Extract log templates (M2A — existing)
4. **Preprocess metrics: fill gaps, normalize, window (M2B — new)**
5. Detect log anomalies via **LogBERT + TF-IDF ensemble** (M3 — enhanced)
6. **Detect metric anomalies via LSTM AE + Temporal Transformer (M3 — new)**
7. **Compute unified anomaly scores (M3 — new)**
8. Build unified signal DataFrame (logs + metrics)
9. Build causal graph via **Granger + FDR + PC Algorithm** (M4 — enhanced)
10. **KHBN posterior refinement** (M4 — new)
11. **5-factor RCA scoring** (M4/M5 — new)
12. Generate NLG narrative (M5 — existing)
13. Get remediation plan (M7 — existing)

---

## Task 12: Docker + Dependencies Update

**Files:**
- Modify: `docker-compose.yml` — enable TimescaleDB, Redis, Prometheus + metric generator init
- Modify: `requirements.txt` — add transformers, h5py, redis, boto3
- Modify: `requirements-docker.txt` — same additions
- Create: `infra/` directory with init scripts

---

## Task 13: Full Test Suite Update

**Files:**
- Update existing test files for compatibility
- Add new unit tests for all new modules
- Update system E2E tests to cover metric stream
- Verify all 152+ existing tests still pass

---

## Execution Order

1 → 2 → 3 → 4 (Utsav's infra chain)
5 → 6 → 7 (Aditya's models, can parallel with 1-4)
8 → 9 → 10 (Shresth's causal, can parallel with 5-7)
11 (integration — needs all above)
12 (Docker — needs 11)
13 (tests — final verification)
