# Automated Root Cause Analysis (RCA) System

An 8-module AI pipeline for automated root cause identification of production system failures. Ingests logs and metrics in real time, detects anomalies using ensemble ML models, performs causal inference with Bayesian networks, ranks root causes, generates natural language reports, and provides tiered remediation plans.

## Architecture

```
                         ┌─────────────────────────────────┐
                         │        Data Sources              │
                         │  Logs (watchdog) │ Metrics (Prom) │
                         └────────┬─────────┬──────────────┘
                                  │         │
                    ┌─────────────▼─┐   ┌───▼──────────────┐
                    │ M1A: Log      │   │ M1B: Metric      │
                    │ Ingestion     │   │ Ingestion        │
                    │ (Watchdog)    │   │ (Prom→Redis→TSDB)│
                    └───────┬───────┘   └───────┬──────────┘
                            │                   │
                    ┌───────▼───────┐   ┌───────▼──────────┐
                    │ M2A: Log      │   │ M2B: Metric      │
                    │ Preprocessing │   │ Preprocessing    │
                    │ (Drain3)      │   │ (Z-score, HDF5)  │
                    └───────┬───────┘   └───────┬──────────┘
                            │                   │
                    ┌───────▼───────────────────▼──────────┐
                    │ M3: Anomaly Detection (Ensemble)     │
                    │ TF-IDF │ LogBERT │ LSTM AE │ Transf. │
                    │         Unified Anomaly Scorer        │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │ M4: Causal Inference + RCA Ranking    │
                    │ Granger │ PC Algorithm │ FDR │ KHBN   │
                    │         5-Factor RCA Scoring          │
                    └───────────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
      ┌───────▼───────┐   ┌───────▼───────┐   ┌─────────▼─────────┐
      │ M5: NLG       │   │ M6: API +     │   │ M7: Remediation   │
      │ Report Gen    │   │ Dashboard     │   │ Engine            │
      │ (Jinja2+spaCy)│   │ (FastAPI+React│   │ (3-Tier + Safety) │
      └───────────────┘   └───────────────┘   └───────────────────┘
```

## Modules

| # | Module | Description | Key Tech |
|---|--------|-------------|----------|
| M1A | Log Ingestion | Filesystem watcher for log files | watchdog |
| M1B | Metric Ingestion | Prometheus/CloudWatch/CSV collection via Redis Streams to TimescaleDB | redis, psycopg2, boto3 |
| M2A | Log Preprocessing | Online template extraction | Drain3 |
| M2B | Metric Preprocessing | Gap filling, z-score normalization, windowing to HDF5 | numpy, h5py |
| M3 | Anomaly Detection | Ensemble of 4 models with unified scoring | transformers, PyTorch |
| M4 | Causal Inference | Multi-method causal graph + Bayesian network + 5-factor ranking | causal-learn, pgmpy |
| M5 | NLG Report Generator | Structured narrative reports from analysis results | Jinja2, spaCy |
| M6 | API + Dashboard | REST API (17 endpoints) + React SPA with D3.js visualizations | FastAPI, React, D3.js |
| M7 | Remediation Engine | 3-tier remediation with safety classification | YAML rule engine |

### M3: Anomaly Detection Models

- **TF-IDF Anomaly Detector** -- Baseline log anomaly detection via template frequency analysis
- **LogBERT** -- DistilBERT fine-tuned with masked language modeling on log messages; anomaly score from MLM loss
- **LSTM Autoencoder** -- 2-layer encoder (128 -> 64 units) trained on metric time-series windows
- **Temporal Transformer** -- Multi-head self-attention forecaster for metric anomaly detection
- **Unified Anomaly Scorer** -- Combines all model outputs with configurable weights

### M4: Causal Inference

- **Granger Causality** -- Baseline pairwise time-series causality testing
- **Benjamini-Hochberg FDR** -- False discovery rate correction for multiple hypothesis testing
- **PC Algorithm** -- Constraint-based causal structure learning from observational data
- **KHBN (Knowledge-Hybrid Bayesian Network)** -- Combines learned structure with service topology priors
- **5-Factor RCA Scoring**: `score = 0.35*pagerank + 0.25*temporal_priority + 0.20*anomaly_score + 0.10*rarity_prior + 0.10*event_bonus`

### M7: Remediation Tiers

| Tier | Action | Example |
|------|--------|---------|
| Tier 1 | Auto-executable commands | `kubectl rollout restart deployment/api-gateway` |
| Tier 2 | Guided walkthrough | Step-by-step manual remediation with verification |
| Tier 3 | Escalation | Stakeholder notification with context |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (for frontend)
- Docker & Docker Compose (for full stack)

### Option 1: Docker (Recommended)

Run the entire stack (7 services) with a single command:

```bash
docker compose up --build
```

This starts:

| Service | Port | Description |
|---------|------|-------------|
| TimescaleDB | 5432 | Metric storage (hypertable) |
| Redis | 6379 | Metric stream buffer |
| Prometheus | 9090 | Metric scraping |
| Backend | 8000 | FastAPI API server |
| Frontend | 3000 | React dashboard (Nginx) |

Two init containers (`generate`, `metric-generator`) run first to create synthetic failure data.

Once running:
- **Dashboard**: http://localhost:3000
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

### Option 2: Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Generate synthetic failure data
python scripts/generate_failures.py --scenario all --count 20 --seed 42
python scripts/generate_metrics.py

# Start the backend
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# In a separate terminal, start the frontend
cd frontend
npm install
npm run dev
```

Note: Local mode uses CSV files for metrics instead of TimescaleDB/Redis/Prometheus.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Liveness check |
| `GET` | `/health` | System diagnostics (models loaded, log sources) |
| `POST` | `/analyze` | Trigger RCA analysis (async, returns 202) |
| `GET` | `/report/{id}` | Full RCA report |
| `GET` | `/incidents` | List all incidents |
| `GET` | `/logs/{id}` | Annotated log lines sorted by anomaly score |
| `GET` | `/metrics/{service}` | 60-min anomaly time series |
| `GET` | `/graph/{id}` | Causal DAG (D3.js node-link JSON) |
| `POST` | `/remediate/{id}` | Generate remediation plan |
| `POST` | `/remediate/{id}/execute` | Execute Tier 1 auto-fix actions |
| `GET` | `/remediate/{id}/walkthrough` | Tier 2 step-by-step guide |
| `GET` | `/remediate/{id}/prevention` | Prevention checklist (3 horizons) |
| `GET` | `/audit/{id}` | Immutable audit log |
| `POST` | `/events/deploy` | Ingest deployment events |
| `POST` | `/events/config` | Ingest config change events |
| `GET` | `/events` | List all events |
| `GET` | `/events/{id}` | Events correlated to an incident |

## Dashboard

The React dashboard has 6 tabs:

- **Incidents** -- List of all analyzed incidents with status
- **RCA Report** -- Full narrative report with confidence scores and evidence
- **Causal Graph** -- Interactive D3.js DAG (square nodes = root causes, circles = affected services)
- **Anomaly Timeline** -- Time-series anomaly score visualization
- **Remediation** -- Tiered remediation plan with execution controls
- **Audit Log** -- Immutable record of all remediation actions

## Testing

```bash
# Run all 269 tests
pytest

# Run by category
pytest -m unit          # 199 unit tests
pytest -m integration   # 27 integration tests
pytest -m system        # 43 system/E2E tests

# Run specific test file
pytest tests/unit/test_logbert.py -v
```

### Test Coverage

| Suite | Tests | Description |
|-------|-------|-------------|
| Unit | 199 | Individual module tests (models, ingestion, preprocessing, NLG, remediation) |
| Integration | 27 | API endpoint tests, pipeline orchestration |
| System/E2E | 43 | Full pipeline scenarios, accuracy validation, remediation verification |

### Accuracy Targets (from PRD)

| Metric | Target | Status |
|--------|--------|--------|
| Log parse rate | 100% | Pass |
| Drain3 template recall | >95% | Pass |
| Log anomaly detection (P/R) | >85% / >75% | Pass |
| Cross-file causality (p<0.05) | >=80% cases | Pass |
| End-to-end Top-1 accuracy | >70% | Pass |
| End-to-end Top-3 accuracy | >88% | Pass |
| Remediation completeness | All steps + correct commands | Pass |

## Project Structure

```
testingrca/
├── src/
│   ├── api/                # FastAPI server (17 endpoints)
│   ├── common/             # Config loader (YAML + env overrides)
│   ├── ingestion/          # Log watcher + metric collectors
│   ├── models/             # ML models (7 files)
│   │   ├── anomaly_detector.py    # TF-IDF + Z-score
│   │   ├── logbert.py             # DistilBERT log anomaly
│   │   ├── deep_learning.py       # LSTM Autoencoder
│   │   ├── temporal_transformer.py # Metric forecaster
│   │   ├── unified_scorer.py      # Ensemble scoring
│   │   ├── causal_inference.py    # Granger + PC + FDR + RCA
│   │   └── khbn.py                # Bayesian network
│   ├── preprocessing/      # Drain3 log parser + metric preprocessor
│   ├── pipeline.py         # 13-stage orchestrator
│   ├── reporting/          # NLG narrative generator
│   └── remediation/        # 3-tier remediation engine
├── frontend/
│   ├── src/
│   │   ├── Dashboard.jsx          # Main tab orchestrator
│   │   └── components/            # 8 React components
│   ├── Dockerfile                 # Node build + Nginx serve
│   └── nginx.conf                 # SPA routing + API proxy
├── config/
│   ├── config.yaml                # System configuration
│   ├── remediation_rules.yaml     # 10 failure scenario rules
│   ├── safety_rules.yaml          # Safety classification
│   ├── service_topology.yaml      # Service dependency DAG (KHBN)
│   └── prometheus.yml             # Prometheus scrape config
├── infra/
│   ├── init-db.sql                # TimescaleDB schema
│   └── redis.conf                 # Redis Streams config
├── templates/
│   └── reports/                   # Jinja2 report templates
├── scripts/
│   ├── generate_failures.py       # 10 failure scenarios (log gen)
│   └── generate_metrics.py        # Metric generator (Prometheus format)
├── tests/
│   ├── unit/                      # 199 tests
│   ├── integration/               # 27 tests
│   └── system/                    # 43 tests
├── data/
│   ├── logs/                      # Generated log files
│   ├── metrics/                   # Generated metric CSVs
│   └── labels/                    # Ground truth labels
├── docker-compose.yml             # 7 services
├── Dockerfile                     # Backend (Python 3.11-slim)
├── requirements.txt               # 25 pinned dependencies
└── requirements-docker.txt        # CPU-only Docker dependencies
```

## Configuration

Primary config file: `config/config.yaml`

Key settings:
- `log_sources` -- Log file paths and formats (application, database, system)
- `metric_sources` -- Prometheus URL and scrape interval
- `database` -- TimescaleDB connection (overridable via `RCA_DB_*` env vars)
- `redis` -- Redis Streams config (overridable via `RCA_REDIS_*` env vars)
- `anomaly_detection.alpha` -- Model score weight (default: 0.80)
- `causal_inference.fdr_alpha` -- FDR correction threshold (default: 0.05)
- `causal_inference.lags` -- Granger causality lag order (default: 6)

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RCA_DB_HOST` | localhost | TimescaleDB host |
| `RCA_DB_PORT` | 5432 | TimescaleDB port |
| `RCA_DB_USER` | rca_user | Database user |
| `RCA_DB_PASSWORD` | rca_pass | Database password |
| `RCA_DB_NAME` | rca_metrics | Database name |
| `RCA_REDIS_HOST` | localhost | Redis host |
| `RCA_REDIS_PORT` | 6379 | Redis port |
| `RCA_PROMETHEUS_URL` | http://localhost:9090 | Prometheus URL |

## Tech Stack

**Backend**: Python 3.11, FastAPI, PyTorch (CPU), Transformers (DistilBERT), scikit-learn, spaCy, pgmpy, causal-learn, Drain3, h5py, Redis, psycopg2

**Frontend**: React 18, Vite, D3.js, Nginx

**Infrastructure**: TimescaleDB (PostgreSQL 15), Redis 7, Prometheus 2.51

**Testing**: pytest (269 tests across unit, integration, and system suites)

## Failure Scenarios

The system includes 10 pre-built failure scenarios for testing and demonstration:

1. Database Connection Pool Exhaustion
2. Memory Leak in Application Server
3. Disk I/O Saturation
4. Network Partition
5. Certificate Expiration
6. DNS Resolution Failure
7. Database Migration Failure
8. Cascading Service Failure
9. Resource Quota Exceeded
10. Configuration Drift

Each scenario has corresponding log generation scripts, metric patterns, ground truth labels, and remediation rules.

## Authors

Aditya Prakash, Shresth Modi, Utsav Upadhyay, Yashash Mathur
