# Automated RCA System - Comprehensive Project Documentation

**Generated:** March 2026
**Project:** Automated Root Cause Analysis System
**Authors:** Aditya Prakash, Shresth Modi, Utsav Upadhyay, Yashash Mathur
**Status:** All 8 phases complete, Docker verified

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Module Breakdown](#3-module-breakdown)
4. [Technical Implementation Details](#4-technical-implementation-details)
5. [API Documentation](#5-api-documentation)
6. [Frontend Architecture](#6-frontend-architecture)
7. [Testing Strategy](#7-testing-strategy)
8. [Deployment & Infrastructure](#8-deployment--infrastructure)
9. [Configuration Reference](#9-configuration-reference)
10. [Security Analysis](#10-security-analysis)
11. [Performance Analysis](#11-performance-analysis)
12. [Areas for Improvement](#12-areas-for-improvement)
13. [Technical Debt](#13-technical-debt)
14. [Future Roadmap](#14-future-roadmap)

---

## 1. Executive Summary

### 1.1 Project Overview

The **Automated Root Cause Analysis (RCA) System** is a production-quality, AI-powered pipeline that ingests multivariate time-series metrics, event logs, and system dependency graphs to automatically identify, rank, and narrate the root causes of production failures.

### 1.2 Key Achievements

- ✅ **8-Module AI Pipeline** fully implemented
- ✅ **269 tests passing** (152 originally, expanded to 269)
- ✅ **Docker deployment verified** with 7 services
- ✅ **10 failure scenarios** implemented with ground truth labels
- ✅ **17 REST API endpoints** implemented
- ✅ **6-tab React Dashboard** with D3.js causal graph visualization
- ✅ **Top-1 accuracy >70%** (passing PRD target)
- ✅ **5-factor RCA scoring** with PageRank + KHBN

### 1.3 Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11, FastAPI, PyTorch (CPU), Transformers |
| **ML Models** | LSTM AE, Temporal Transformer, LogBERT, TF-IDF |
| **Causal Inference** | Granger Causality, PC Algorithm, KHBN (pgmpy) |
| **Frontend** | React 18, Vite, D3.js, Recharts, lucide-react |
| **Infrastructure** | TimescaleDB, Redis, Prometheus, Docker |
| **Testing** | pytest (269 tests) |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                                 │
│    Logs (watchdog)  │  Metrics (Prometheus/CloudWatch/CSV)      │
└─────────────────────┴───────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                 ┌──────────────────┐
    │ M1A: Log     │                 │ M1B: Metric     │
    │ Ingestion    │                 │ Ingestion       │
    │ (Watchdog)   │                 │ (Prom→Redis→TS) │
    └───────┬───────┘                 └───────┬──────────┘
            │                                   │
    ┌───────▼───────┐                 ┌───────▼──────────┐
    │ M2A: Log      │                 │ M2B: Metric      │
    │ Preprocessing │                 │ Preprocessing    │
    │ (Drain3)      │                 │ (Z-score, HDF5) │
    └───────┬───────┘                 └───────┬──────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
            ┌─────────────────────────────────────┐
            │ M3: Anomaly Detection (Ensemble)     │
            │ TF-IDF │ LogBERT │ LSTM AE │ Transf. │
            │         Unified Anomaly Scorer        │
            └─────────────────┬───────────────────┘
                              │
            ┌─────────────────▼───────────────────┐
            │ M4: Causal Inference + RCA Ranking    │
            │ Granger │ PC Algorithm │ FDR │ KHBN   │
            │         5-Factor RCA Scoring          │
            └─────────────────┬───────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
┌───▼───────┐         ┌──────▼──────┐         ┌────────▼────────┐
│ M5: NLG   │         │ M6: API +   │         │ M7: Remediation │
│ Report Gen│         │ Dashboard   │         │ Engine          │
│ (Jinja2)  │         │ (FastAPI)   │         │ (3-Tier)        │
└───────────┘         └─────────────┘         └─────────────────┘
```

### 2.2 Data Flow

1. **Log Stream**: File tailing → Drain3 parsing → TF-IDF/LogBERT scoring
2. **Metric Stream**: Prometheus/CSV → Preprocessing → LSTM AE/Transformer scoring
3. **Convergence**: Unified anomaly scores → Granger + PC → KHBN → PageRank
4. **Output**: NLG narrative → FastAPI → React Dashboard + Remediation

### 2.3 Project Structure

```
testingrca/
├── src/
│   ├── api/server.py              # FastAPI server (17 endpoints)
│   ├── common/config.py           # YAML config loader
│   ├── ingestion/
│   │   ├── log_ingest.py         # Log file watcher
│   │   ├── metric_ingest.py       # Prometheus/CloudWatch collector
│   │   ├── redis_buffer.py       # Redis Streams buffer
│   │   └── timescaledb_store.py   # TimescaleDB writer
│   ├── models/
│   │   ├── anomaly_detector.py    # TF-IDF + Z-score baseline
│   │   ├── logbert.py            # DistilBERT MLM anomaly detection
│   │   ├── deep_learning.py      # LSTM Autoencoder
│   │   ├── temporal_transformer.py # Transformer forecaster
│   │   ├── unified_scorer.py     # Multi-model ensemble scorer
│   │   ├── causal_inference.py   # Granger + PC + FDR
│   │   └── khbn.py               # Knowledge-Hierarchical Bayesian Network
│   ├── preprocessing/
│   │   ├── log_parser.py         # Drain3 template extraction
│   │   └── metric_preprocessor.py # Gap filling, normalization
│   ├── pipeline.py                # 13-stage orchestrator
│   ├── reporting/nlg_generator.py # Jinja2 + spaCy NLG
│   └── remediation/
│       └── remediation_engine.py  # 3-tier safety engine
├── frontend/
│   ├── src/
│   │   ├── Dashboard.jsx          # Main tab orchestrator
│   │   └── components/            # 8 React components
│   │       ├── CausalGraph.jsx    # D3.js DAG visualization
│   │       ├── RCAReportPanel.jsx # Markdown renderer + cause cards
│   │       ├── AnomalyTimeline.jsx # Time-series visualization
│   │       ├── AnomalyHeatmap.jsx  # Heatmap grid
│   │       ├── IncidentsList.jsx  # Sidebar incident list
│   │       ├── RemediationPanel.jsx # Tiered action UI
│   │       ├── AuditLog.jsx       # Immutable audit trail
│   │       └── Header.jsx         # Top nav + analyze button
│   ├── Dockerfile                 # Node 20 + Nginx
│   └── nginx.conf                 # SPA routing + /api proxy
├── config/
│   ├── config.yaml                # System configuration
│   ├── remediation_rules.yaml     # 10 failure scenario rules (794 lines)
│   ├── safety_rules.yaml          # Safety classification patterns
│   ├── service_topology.yaml      # KHBN structural prior
│   └── prometheus.yml             # Scrape config
├── infra/
│   ├── init-db.sql               # TimescaleDB schema
│   └── redis.conf                # Redis Streams config
├── scripts/
│   ├── generate_failures.py       # 10 scenario log generator (1101 lines)
│   └── generate_metrics.py         # Prometheus-format metric generator
├── tests/
│   ├── unit/                     # 15 test files (199 tests)
│   ├── integration/               # API endpoint tests
│   └── system/                   # E2E scenario tests
├── data/
│   ├── logs/                     # Generated log files
│   ├── metrics/                  # Generated metric CSVs
│   └── labels/                   # Ground truth labels
├── docker-compose.yml             # 7 services
├── Dockerfile                     # Backend (Python 3.11-slim)
├── requirements.txt               # 25 dependencies
└── requirements-docker.txt        # CPU-only Docker deps
```

---

## 3. Module Breakdown

### 3.1 Module 1A: Log File Ingestion (M1A)

**Owner:** Utsav Upadhyay
**Implementation:** `src/ingestion/log_ingest.py`

**Responsibilities:**
- Monitor configured file paths using Python `watchdog`
- Support multiple formats: plaintext, syslog, JSON
- Parse timestamps, log levels, and raw messages
- Handle file rotation automatically
- Buffer parsed events with timestamps

**Supported Formats:**
| Format | Example | Parser |
|--------|---------|--------|
| Plaintext | `2026-03-01 10:05:32 ERROR Connection refused` | Regex |
| Syslog | `Mar 1 10:05:32 hostname app[1234]: Connection refused` | Regex |
| JSON | `{"ts": "...", "level": "error", "msg": "..."}` | json.loads() |
| ISO 8601 | `2026-03-01T10:05:32.456Z [ERROR]` | Regex |

**Key Configuration (config.yaml):**
```yaml
log_sources:
  - label: application
    path: ./data/logs/app.log
    format: plaintext
  - label: database
    path: ./data/logs/db.log
    format: plaintext
```

### 3.2 Module 1B: Metric Ingestion (M1B)

**Owner:** Utsav Upadhyay
**Implementation:** `src/ingestion/metric_ingest.py`

**Responsibilities:**
- Pull metrics from Prometheus API
- Query CloudWatch GetMetricData API (optional)
- Receive CI/CD webhook events
- Write to Redis Streams buffer
- Support CSV fallback for local development

**Architecture:**
```
Prometheus/CloudWatch/CSV → Redis Streams → TimescaleDB
```

**Environment Variables:**
- `RCA_PROMETHEUS_URL`: http://localhost:9090
- `RCA_REDIS_HOST`, `RCA_REDIS_PORT`: Redis connection
- `RCA_DB_*`: TimescaleDB connection

### 3.3 Module 2A: Log Preprocessing (M2A)

**Owner:** Utsav Upadhyay
**Implementation:** `src/preprocessing/log_parser.py`

**Technology:** Drain3 online log template extractor

**Features:**
- Online learning (no retraining for new templates)
- Variable token replacement (IDs, IPs, usernames → `*`)
- Persistent state (`data/drain3_state.bin`)
- Event ID assignment per template

**Example Transformation:**
| Raw Log | Template | Event ID |
|---------|----------|----------|
| `User 12345 failed login from 192.168.1.1` | `User * failed login from *` | EVT_047 |
| `User 67890 failed login from 10.0.0.5` | `User * failed login from *` | EVT_047 (same) |

### 3.4 Module 2B: Metric Preprocessing (M2B)

**Owner:** Utsav Upadhyay
**Implementation:** `src/preprocessing/metric_preprocessor.py`

**Features:**
- Gap filling (< 5 min: forward-fill, 5-30 min: linear interpolation, > 30 min: sentinel)
- Z-score normalization (rolling 7-day window)
- Sliding window segmentation (60-step windows, 50% overlap)
- HDF5 output format

### 3.5 Module 3: Anomaly Detection (M3)

**Owner:** Aditya Prakash
**Implementation:** `src/models/` (7 files)

#### 3.5.1 TF-IDF Anomaly Detector
- **File:** `anomaly_detector.py`
- **Strategy:** Cosine distance from healthy TF-IDF profile
- **Threshold:** 99th percentile
- **Use Case:** MVP baseline, CPU-friendly

#### 3.5.2 LogBERT Detector
- **File:** `logbert.py`
- **Strategy:** DistilBERT fine-tuned with MLM on healthy logs
- **Anomaly Score:** Masked token prediction loss
- **Threshold:** 95th percentile of training loss
- **Sampling:** 2000 message cap per source for CPU feasibility

#### 3.5.3 LSTM Autoencoder
- **File:** `deep_learning.py`
- **Architecture:** 2-layer encoder (128→64) + symmetric decoder
- **Training:** MSE reconstruction loss on healthy data
- **Scoring:** Sigmoid-normalized reconstruction error

#### 3.5.4 Temporal Transformer
- **File:** `temporal_transformer.py`
- **Architecture:** 4-head Transformer encoder, 2 layers, d_model=128
- **Forecasting:** 10-step horizon
- **Scoring:** MAE between forecast and actual, normalized

#### 3.5.5 Unified Anomaly Scorer
- **File:** `unified_scorer.py`
- **Formula:** `score = α × model_score + (1-α) × rarity_prior`
- **Default α:** 0.80
- **Log Weights:** TF-IDF 0.3, LogBERT 0.7
- **Metric Weights:** LSTM AE 0.6, Transformer 0.4

### 3.6 Module 4: Causal Inference (M4)

**Owner:** Shresth Modi
**Implementation:** `src/models/causal_inference.py`, `src/models/khbn.py`

#### 3.6.1 Granger Causality
- Pairwise time-series causality testing
- Benjamini-Hochberg FDR correction
- Default: max_lag=6, α=0.05

#### 3.6.2 PC Algorithm
- Constraint-based causal structure learning
- Uses Fisher-Z test for conditional independence
- Majority-vote combination with Granger

#### 3.6.3 KHBN (Knowledge-Hierarchical Bayesian Network)
- Service topology as structural prior
- Topology-aware edge filtering
- pgmpy-based posterior inference (fallback: topology distance)
- VariableElimination for posterior probability

#### 3.6.4 5-Factor RCA Scoring
```
rca_score = 0.35 × pagerank 
           + 0.25 × temporal_priority 
           + 0.20 × anomaly_score 
           + 0.10 × rarity_prior 
           + 0.10 × event_bonus
```

### 3.7 Module 5: NLG Report Generator (M5)

**Owner:** Yashash Mathur
**Implementation:** `src/reporting/nlg_generator.py`

**Features:**
- Jinja2 templates + spaCy NLP
- Markdown + plain-text output
- Causal chain narration with causal verbs
- Confidence-adaptive language
- Prevention checklist (3 horizons)

**Key Classes:**
- `NLGGenerator`: Main generator with template rendering
- Confidence labels: High (≥0.85), Moderate-High (≥0.70), Moderate (≥0.50)
- Severity labels: Critical (≥0.90), High (≥0.70), Medium (≥0.50)

### 3.8 Module 6: API & Dashboard (M6)

**Owner:** Yashash Mathur
**Implementation:** `src/api/server.py` + `frontend/`

#### API Server Features:
- FastAPI with 17 endpoints
- CORS middleware for React dev server
- ThreadPoolExecutor for CPU-bound pipeline
- In-memory incident store (dict)
- Background pipeline execution via asyncio

#### React Dashboard Features:
- 6-tab interface: Timeline, Heatmap, Causal Graph, RCA Report, Remediation, Audit
- D3.js force-directed graph with zoom/pan
- Markdown rendering with custom parser
- Polling-based incident loading
- Responsive layout with CSS Grid/Flexbox

### 3.9 Module 7: Remediation Engine (M7)

**Owner:** Yashash Mathur
**Implementation:** `src/remediation/remediation_engine.py`

#### 3-Tier Safety Classification:

| Tier | Action Type | Example | Automation |
|------|-------------|---------|------------|
| **Tier 1** | Auto-executable, reversible | `kubectl rollout restart` | 30s countdown + cancel |
| **Tier 2** | Guided walkthrough, human verification | DB migration rollback | Step-by-step UI |
| **Tier 3** | Advisory, architectural | Code fix recommendation | Inform only |

#### 10 Failure Scenarios (remediation_rules.yaml):
1. `db_migration_applied` - Query plan regression
2. `memory_leak_detected` - WebSocket connection leak
3. `network_partition_detected` - Redis split-brain
4. `thread_pool_exhaustion` - Background job starvation
5. `dns_ttl_misconfiguration` - DNS propagation delay
6. `cpu_runaway_process` - CPU saturation
7. `connection_leak_bug` - DB connection leak
8. `cache_ttl_expiry` - Thundering herd stampede
9. `log_rotation_disabled` - Disk exhaustion
10. `consumer_service_crash` - Message queue backlog

#### Safety Features:
- Confidence gate (default: 0.70 threshold)
- Audit log (append-only, immutable)
- Reversibility tracking per action
- Escalation rules (Tier 1 fail → Tier 2)

---

## 4. Technical Implementation Details

### 4.1 Pipeline Orchestrator (`pipeline.py`)

The `PipelineOrchestrator` class orchestrates all modules in a 13-stage flow:

```
Stage 1A: Read logs from config sources
Stage 1B: Read metrics (CSV/TimescaleDB/Redis)
Stage 2A: Drain3 template extraction
Stage 2B: Metric preprocessing (gap fill + normalize)
Stage 3A: TF-IDF + LogBERT log anomaly scoring
Stage 3B: LSTM AE + Transformer metric anomaly scoring
Stage 3C: Unified scoring via UnifiedAnomalyScorer
Stage 4:  Build signal DataFrame (anomaly time series)
Stage 5:  Causal graph (Granger + PC + FDR)
Stage 6:  KHBN posterior refinement (optional)
Stage 7:  5-factor RCA ranking
Stage 8:  Determine top root cause + remediation key
Stage 9:  Generate remediation plan
Stage 10: Build NLG report data
Stage 11: Render NLG narrative
Stage 12: Annotate log lines with scores
Stage 13: Assemble final report
```

**Graceful Degradation:**
- LogBERT unavailable → TF-IDF only
- LSTM AE / Transformer unavailable → Statistical fallback
- KHBN unavailable → Topology distance fallback
- Metric data unavailable → Log-only analysis

### 4.2 Causal Inference Engine (`causal_inference.py`)

**Key Algorithms:**

1. **Granger Causality Test:**
   - `statsmodels.tsa.stattools.grangercausalitytests`
   - Max lag: configurable (default 6)
   - Pairwise testing across all signal pairs

2. **Benjamini-Hochberg FDR:**
   - `statsmodels.stats.multitest.multipletests`
   - Method: `fdr_bh`
   - Controls false discovery rate across all tests

3. **PC Algorithm:**
   - `causallearn.search.ConstraintBased.PC`
   - Independence test: Fisher-Z
   - Returns skeleton graph + orientation

4. **Edge Combination:**
   - Majority vote: union of Granger + PC edges
   - Weight = average of Granger weight + PC confidence

5. **Cycle Breaking:**
   - Weakest edge removal in each cycle
   - Weight-based: remove lowest confidence edge

### 4.3 Service Topology (`service_topology.yaml`)

**Purpose:** KHBN structural prior for Bayesian Network

**Service Dependency Graph:**
```
load_balancer
    └── web_server
            └── app_server
                    ├── database
                    ├── cache
                    ├── message_queue
                    └── storage
dns (root)
network (root)
os_kernel (root)
```

**Mappings:**
- `source_to_service`: Log labels → service names
- `metric_to_service`: Metric prefixes → service names

### 4.4 Synthetic Failure Generator (`generate_failures.py`)

**10 Failure Scenarios with Realistic Propagation:**

| Scenario | Root Cause | Propagation | Severity Scaling |
|----------|------------|-------------|------------------|
| DB Migration | Index regression | 0-7 hours | Configurable |
| Memory Leak | WebSocket leak | 4-48 hours | Configurable |
| Network Partition | BGP split-brain | 0-5 min | Configurable |
| Thread Pool | Background job | 30-90 sec | Configurable |
| DNS TTL | Cache propagation | 0-30 min | Configurable |
| CPU Saturation | Runaway process | Immediate | Configurable |
| Connection Leak | DB pool exhaustion | 15-60 min | Configurable |
| Cache Stampede | TTL expiry | At expiry | Configurable |
| Disk Exhaustion | Log rotation disabled | Gradual | Configurable |
| MQ Backlog | Consumer crash | Variable | Configurable |

---

## 5. API Documentation

### 5.1 Endpoint Summary

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | Liveness check | `{"message": "...", "version": "0.3.0"}` |
| `GET` | `/health` | System diagnostics | Models loaded, log sources, parse failures |
| `POST` | `/analyze` | Trigger RCA analysis | 202 + incident_id |
| `GET` | `/report/{id}` | Full RCA report | Complete incident analysis |
| `GET` | `/incidents` | List all incidents | Array of incident summaries |
| `GET` | `/logs/{id}` | Annotated log lines | Sorted by anomaly score |
| `GET` | `/metrics/{service}` | Anomaly time series | 60-min metric data |
| `GET` | `/graph/{id}` | Causal DAG (D3.js) | Node-link JSON |
| `POST` | `/remediate/{id}` | Generate remediation plan | 3-tier action plan |
| `POST` | `/remediate/{id}/execute` | Execute Tier 1 actions | Audit trail |
| `GET` | `/remediate/{id}/walkthrough` | Step-by-step guide | Tier 2 steps |
| `GET` | `/remediate/{id}/prevention` | Prevention checklist | 3 horizons |
| `GET` | `/audit/{id}` | Immutable audit log | Action history |
| `POST` | `/events/deploy` | CI/CD deploy event | 202 + event_id |
| `POST` | `/events/config` | Config change event | 202 + event_id |
| `GET` | `/events` | List all events | Deploy + config events |
| `GET` | `/events/{id}` | Events for incident | ±30 min window |

### 5.2 Request/Response Examples

#### POST /analyze
```json
// Request
{
  "incident_id": null,           // Auto-generated if null
  "start_time": "2026-03-15T10:00:00Z",
  "end_time": "2026-03-15T12:00:00Z",
  "services": ["application", "database"],
  "priority": "high"
}

// Response (202 Accepted)
{
  "incident_id": "INC-1710500000",
  "status": "processing",
  "estimated_completion_seconds": 180,
  "poll_url": "/report/INC-1710500000"
}
```

#### GET /report/{id}
```json
{
  "incident_id": "INC-1710500000",
  "status": "complete",
  "detected_at": "2026-03-15 11:45:32",
  "root_cause": "database",
  "confidence": 0.82,
  "narrative": "# RCA Report\n\nA High-severity incident...",
  "causal_graph": {
    "nodes": [...],
    "edges": [...]
  },
  "ranked_causes": [
    {
      "rank": 1,
      "cause": "database",
      "confidence": 0.82,
      "pagerank_score": 0.95,
      "temporal_priority": 0.90,
      "anomaly_score": 0.91,
      "rarity_prior": 0.80,
      "event_bonus": 1.0
    }
  ],
  "remediation_plan": {
    "tier1_auto_actions": [...],
    "tier2_walkthrough": {...},
    "tier3_advisory": [...]
  }
}
```

### 5.3 Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2026-03-15T11:45:32",
  "version": "0.3.0",
  "models_loaded": {
    "drain3_parser": true,
    "tfidf_anomaly_detector": true,
    "granger_causal": true,
    "nlg_generator": true,
    "remediation_engine": true,
    "unified_scorer": true,
    "logbert": true,
    "lstm_ae": true,
    "temporal_transformer": true,
    "khbn": true
  },
  "log_sources": {
    "application": {"path": "...", "accessible": true, "size_bytes": 12345},
    "database": {"path": "...", "accessible": true, "size_bytes": 67890}
  },
  "active_incidents": 0,
  "total_incidents": 15,
  "parse_failures": 0
}
```

---

## 6. Frontend Architecture

### 6.1 Dashboard Component (`Dashboard.jsx`)

**State Management:**
```javascript
// Core state
const [incidents, setIncidents] = useState([]);
const [selectedIncidentId, setSelectedIncidentId] = useState(null);
const [report, setReport] = useState(null);
const [graphData, setGraphData] = useState(null);

// Derived state
const [logs, setLogs] = useState([]);
const [metrics, setMetrics] = useState(null);
const [remediationPlan, setRemediationPlan] = useState(null);
const [confidenceGate, setConfidenceGate] = useState(null);
```

**Data Loading Strategy:**
1. On mount: fetch incidents list
2. On analyze: POST /analyze → poll /report/{id}
3. On incident select: parallel fetch (graph, logs, remediation, metrics)

**Tab Navigation:**
```javascript
const TABS = [
  { key: 'timeline', label: 'Timeline' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'graph', label: 'Causal Graph' },
  { key: 'report', label: 'RCA Report' },
  { key: 'remediation', label: 'Remediation' },
  { key: 'audit', label: 'Audit Log' },
];
```

### 6.2 Causal Graph Component (`CausalGraph.jsx`)

**D3.js Force Simulation:**
- Force link: `d3.forceLink()` with distance 140
- Force charge: `d3.forceManyBody()` with strength -400
- Force center: `d3.forceCenter(width/2, height/2)`
- Force collision: `d3.forceCollide()` for node spacing

**Visual Encoding:**
- **Node Shape:** Square = root cause, Circle = affected
- **Node Color:** Green (< 0.3), Amber (0.3-0.7), Red (> 0.7)
- **Node Size:** 10-40px based on confidence
- **Edge Width:** 1-8px based on weight
- **Root Glow:** Gold stroke with Gaussian blur filter

**Interactions:**
- Drag nodes: fix position, release to float
- Zoom/pan: scale 0.2x to 4x
- Click node: trigger `onNodeClick` callback

### 6.3 RCA Report Panel (`RCAReportPanel.jsx`)

**Custom Markdown Parser:**
- Headers (h1-h3)
- Bold/italic
- Lists (ul/ol)
- Blockquotes
- Inline code
- Horizontal rules

**Causal Chain Breadcrumb:**
- Walks backward through edges from root cause
- Builds path: `root_cause ← intermediate ← ... ← symptom`
- Max depth: 10 levels

**Cause Cards:**
- Rank badge (gold/silver/bronze)
- Confidence bar with color coding
- Expandable evidence section

### 6.4 Other Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `Header` | Top navigation | Analyze button, title |
| `IncidentsList` | Sidebar | Sorted list, selected highlight |
| `AnomalyTimeline` | Time-series chart | Recharts-based |
| `AnomalyHeatmap` | Grid visualization | Log + metric anomaly grid |
| `RemediationPanel` | Action UI | Tier display, execute button |
| `AuditLog` | Action history | Immutable action trail |

---

## 7. Testing Strategy

### 7.1 Test Coverage Summary

| Suite | Tests | Description |
|-------|-------|-------------|
| Unit | 199 | Individual module tests |
| Integration | 27 | API + pipeline orchestration |
| System/E2E | 43 | Full scenarios, accuracy validation |
| **Total** | **269** | **All passing** |

### 7.2 Unit Test Files

```
tests/unit/
├── test_anomaly_detector.py      # TF-IDF + Z-score
├── test_logbert.py               # DistilBERT MLM
├── test_temporal_transformer.py   # Transformer forecasting
├── test_khbn.py                  # Bayesian network
├── test_causal_inference.py      # Granger + PC + FDR
├── test_enhanced_causal.py       # Enhanced causal engine
├── test_rca_scoring.py          # 5-factor scoring
├── test_unified_scorer.py       # Ensemble scoring
├── test_nlg_generator.py         # NLG templates
├── test_remediation_engine.py    # 3-tier safety
├── test_log_parser.py           # Drain3 extraction
├── test_metric_preprocessor.py  # Gap fill + normalize
├── test_config.py               # YAML config loading
├── test_log_ingest.py           # Log parsing
└── test_metric_ingest.py         # Metric collection
```

### 7.3 System Test Coverage

**10 Failure Scenarios × 4 Test Cases = 40+ E2E tests:**
1. DB Migration detection
2. Memory Leak detection
3. Network Partition detection
4. Thread Pool Exhaustion
5. DNS TTL misconfiguration
6. CPU Runaway Process
7. Connection Pool Leak
8. Cache Stampede
9. Log Rotation Disabled
10. Consumer Service Crash

**Verification:**
- Top-1 accuracy > 70% (PRD target)
- Top-3 accuracy > 88%
- Remediation plan completeness
- NLG narrative generation

### 7.4 Accuracy Validation

**PRD Targets:**
| Metric | Target | Status |
|--------|--------|--------|
| Log parse rate | 100% | ✅ Pass |
| Drain3 template recall | > 95% | ✅ Pass |
| Log anomaly detection P/R | > 85% / > 75% | ✅ Pass |
| Cross-file causality (p<0.05) | >= 80% | ✅ Pass |
| End-to-end Top-1 accuracy | > 70% | ✅ Pass |
| End-to-end Top-3 accuracy | > 88% | ✅ Pass |

---

## 8. Deployment & Infrastructure

### 8.1 Docker Compose Services

```yaml
services:
  generate:           # Init: generate 200 failure scenarios
  metric-generator:   # Init: generate metrics
  timescaledb:        # PostgreSQL 15 + TimescaleDB
  redis:              # Redis 7 (Streams)
  prometheus:         # Metric scraping
  backend:            # FastAPI + Uvicorn
  frontend:           # React + Nginx
```

### 8.2 Backend Dockerfile

**Base:** `python:3.11-slim`
**Key Optimizations:**
- CPU-only PyTorch (`requirements-docker.txt`)
- Removed unused deps (mlflow, dvc, matplotlib, seaborn)
- spaCy model download at build time
- Final image: ~2.3GB (was 9.86GB with GPU torch)

### 8.3 Frontend Dockerfile

**Base:** `node:20-alpine`
**Build:** `npm install && npm run build`
**Serve:** Nginx Alpine with SPA routing
**Final image:** 93.6MB

### 8.4 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RCA_DB_HOST` | timescaledb | Database host |
| `RCA_DB_PORT` | 5432 | Database port |
| `RCA_DB_USER` | postgres | Database user |
| `RCA_DB_PASSWORD` | password | Database password |
| `RCA_DB_NAME` | rca_system | Database name |
| `RCA_REDIS_HOST` | redis | Redis host |
| `RCA_REDIS_PORT` | 6379 | Redis port |
| `RCA_PROMETHEUS_URL` | http://prometheus:9090 | Prometheus URL |
| `PYTHONPATH` | /app | Python module path |
| `LOG_LEVEL` | info | Logging verbosity |

### 8.5 Volume Mounts

| Volume | Service | Purpose |
|--------|---------|---------|
| `app-data` | generate, backend | Logs + metrics + labels |
| `ts-data` | timescaledb | Persistent database |
| `redis-data` | redis | Redis persistence |
| `prom-data` | prometheus | Prometheus data |

### 8.6 Health Checks

| Service | Check | Interval | Timeout |
|---------|-------|----------|---------|
| timescaledb | `pg_isready` | 10s | 5s |
| redis | `redis-cli ping` | 10s | 5s |
| backend | HTTP `/health` | 30s | 10s |
| frontend | Nginx health | - | - |

---

## 9. Configuration Reference

### 9.1 config.yaml

```yaml
# Log sources (M1A)
log_sources:
  - label: application
    path: ./data/logs/app.log
    format: plaintext  # plaintext | json | syslog
  - label: database
    path: ./data/logs/db.log
    format: plaintext

# Metric sources (M1B)
metric_sources:
  prometheus_url: http://localhost:9090
  scrape_interval_seconds: 60

# Database (M2B)
database:
  host: localhost
  port: 5432
  user: postgres
  password: password
  dbname: rca_system

# Anomaly Detection (M3)
anomaly_detection:
  window_size: 60          # Timesteps per window
  threshold_percentile: 99   # P99 threshold for anomaly
  alpha: 0.80             # Model score weight

# Causal Inference (M4)
causal_inference:
  lags: 6                  # Max Granger lag
  fdr_alpha: 0.05         # Benjamini-Hochberg α
  correlation_window_minutes: 2

# Redis (M1B)
redis:
  host: localhost
  port: 6379
  stream_name: rca_metrics
  consumer_group: rca_pipeline
```

### 9.2 remediation_rules.yaml

**Structure per scenario:**
```yaml
scenario_key:
  description: "Human-readable description"
  tier1_auto:
    - action: action_name
      description: "..."
      command: "kubectl rollout restart ..."
      platform: kubernetes
      est_seconds: 180
      reversible: true
  tier2_guided:
    - step: 1
      title: "Step title"
      command: "kubectl rollout undo ..."
      expected_output: "deployment rolled back"
      verification: "kubectl rollout status ..."
      rollback_step: "kubectl rollout undo ..."
      est_minutes: 2
      safety_note: "..."
  tier3_advisory:
    - recommendation: "..."
      horizon: short_term
      owner: Backend Dev
      priority: 0.90
      effort: "2-3 days"
  prevention:
    immediate: [...]
    short_term: [...]
    long_term: [...]
```

### 9.3 safety_rules.yaml

**Tier 1 Patterns:**
```yaml
tier1_auto_execute:
  action_patterns:
    - pattern: "flush_*_cache"
      reversible: true
      blast_radius: single_service
      data_risk: none
    - pattern: "restart_*"
      reversible: true
      blast_radius: single_service
      data_risk: none
```

**Confidence Gate:**
```yaml
confidence_gate:
  min_threshold: 0.70
  low_confidence_behavior: advisory_only
```

---

## 10. Security Analysis

### 10.1 Security Strengths

✅ **No hardcoded secrets** - All credentials via environment variables
✅ **Input validation** - Pydantic models for all API inputs
✅ **Parameterized queries** - SQLAlchemy/psycopg2 with parameter binding
✅ **YAML safe loading** - `yaml.safe_load()` throughout
✅ **Audit logging** - Immutable append-only remediation audit trail
✅ **Safety tiers** - Deterministic safety classification (no ML for safety)
✅ **Confidence gate** - Auto-execute only when confidence > 0.70
✅ **CORS restrictions** - Whitelist of allowed origins

### 10.2 Security Recommendations

#### High Priority

1. **Authentication/Authorization**
   - No authentication on any endpoint
   - No authorization checks
   - **Recommendation:** Add OAuth2/JWT for production

2. **Rate Limiting**
   - No rate limiting on `/analyze` or `/events/deploy`
   - **Recommendation:** Add FastAPI middleware for rate limiting

3. **Webhook Validation**
   - `/events/deploy` and `/events/config` accept arbitrary data
   - **Recommendation:** Validate webhook signatures (GitHub, GitLab, etc.)

#### Medium Priority

4. **SQL Injection**
   - Most queries use ORM, but check for `.raw()` or string interpolation
   - **Status:** Low risk with current implementation

5. **Log Injection**
   - User-controlled log data could be logged verbatim
   - **Recommendation:** Sanitize log messages, limit length

6. **Command Injection**
   - Remediation engine renders Jinja2 templates with context
   - **Recommendation:** Validate all template variables, use allowlists

7. **Audit Log Tampering**
   - In-memory audit log (lost on restart)
   - **Recommendation:** Persist to TimescaleDB or file

### 10.3 Security Configuration

```python
# Current CORS (server.py:56-62)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recommendation for production:
allow_origins=["https://dashboard.example.com"]
```

---

## 11. Performance Analysis

### 11.1 Performance Characteristics

| Component | Data Size | Typical Latency |
|-----------|-----------|-----------------|
| Log parsing (Drain3) | 1K-100K lines | < 1s |
| TF-IDF training | 1K-10K messages | < 5s |
| LogBERT scoring | 2K messages | ~30s (CPU) |
| LSTM AE training | 2K timesteps | ~60s (CPU) |
| Granger causality | 8 signals × 1000 pts | < 10s |
| PC Algorithm | 8 signals × 1000 pts | < 30s |
| KHBN inference | 8 nodes | < 5s |
| NLG generation | Full report | < 1s |

### 11.2 Bottlenecks

1. **LogBERT Transformer** - Most expensive operation
   - Mitigation: 2000 message sampling cap
   - Further optimization: quantization, ONNX export

2. **Granger Causality** - O(n² × lag) pairwise tests
   - Mitigation: FDR reduces redundant tests
   - Further optimization: GPU acceleration

3. **LSTM AE Training** - Full epoch training
   - Mitigation: Only 3 epochs for inference
   - Further optimization: Pre-trained weights

### 11.3 Memory Usage

| Service | Memory | Notes |
|---------|--------|-------|
| Backend (no GPU) | ~2-4 GB | PyTorch + transformers |
| TimescaleDB | ~512 MB | PostgreSQL + extension |
| Redis | ~128 MB | Streams + pub/sub |
| Prometheus | ~256 MB | Time-series storage |
| Frontend (Nginx) | ~50 MB | Static files |

### 11.4 Optimization Opportunities

1. **Model Quantization**
   - LogBERT: INT8 quantization for 2-4x speedup
   - LSTM: FP16 inference

2. **Caching**
   - Cache trained models (Drain3, TF-IDF, LSTM)
   - Cache causal graphs for repeated analyses

3. **Async Processing**
   - Move heavy ML to separate worker processes
   - Use Celery or RQ for background tasks

4. **Database Indexing**
   - Index TimescaleDB hypertable on time + service
   - Partition logs by date

---

## 12. Areas for Improvement

### 12.1 Critical (High Priority)

#### 1. **Persistent Audit Log**
- **Issue:** In-memory audit log lost on restart
- **Fix:** Persist to TimescaleDB or file with append-only semantics
- **Files affected:** `remediation_engine.py`, `infra/init-db.sql`

#### 2. **API Authentication**
- **Issue:** No auth on any endpoint
- **Fix:** Add OAuth2/JWT middleware
- **Files affected:** `server.py`, add `auth/` module

#### 3. **Metric Data Persistence**
- **Issue:** Metrics only from CSV in local mode
- **Fix:** Implement full TimescaleDB writer
- **Files affected:** `timescaledb_store.py`

#### 4. **Real Log Tailing**
- **Issue:** Reads full files, not real-time
- **Fix:** Implement watchdog-based tailing
- **Files affected:** `log_ingest.py`

### 12.2 Important (Medium Priority)

#### 5. **Webhook Signature Validation**
- **Issue:** `/events/deploy` and `/events/config` accept arbitrary data
- **Fix:** Validate HMAC signatures from GitHub/GitLab
- **Files affected:** `server.py`

#### 6. **Model Persistence**
- **Issue:** Retrains models on every startup
- **Fix:** Save/load trained checkpoints
- **Files affected:** `logbert.py`, `deep_learning.py`, `log_parser.py`

#### 7. **Error Handling Consistency**
- **Issue:** Inconsistent error handling across modules
- **Fix:** Centralized error handling middleware
- **Files affected:** `server.py`, `pipeline.py`

#### 8. **Configuration Validation**
- **Issue:** No schema validation for config.yaml
- **Fix:** Add Pydantic model for config validation
- **Files affected:** `config.py`

### 12.3 Nice to Have (Low Priority)

#### 9. **MLflow Integration**
- **Issue:** MLflow service commented out in docker-compose
- **Fix:** Uncomment and integrate logging
- **Files affected:** `docker-compose.yml`, `models/*.py`

#### 10. **Grafana Dashboard**
- **Issue:** No visualization for system metrics
- **Fix:** Add Grafana with Prometheus data source
- **Files affected:** `docker-compose.yml`, add `grafana/` config

#### 11. **Alerting Integration**
- **Issue:** No integration with Alertmanager/PagerDuty
- **Fix:** Add webhook for alerting systems
- **Files affected:** `server.py`, add `/alerts` endpoint

#### 12. **Multi-tenancy**
- **Issue:** Single-tenant only
- **Fix:** Add tenant_id to all data models
- **Files affected:** All data stores

#### 13. **Distributed Deployment**
- **Issue:** All-in-one Docker Compose
- **Fix:** Kubernetes manifests for production
- **Files affected:** Add `k8s/` directory

#### 14. **Model A/B Testing**
- **Issue:** No way to compare model versions
- **Fix:** Add model versioning and shadow mode
- **Files affected:** `unified_scorer.py`

---

## 13. Technical Debt

### 13.1 Code Quality Issues

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Deeply nested conditionals | Medium | `pipeline.py:809-1265` | Extract to helper functions |
| Inconsistent error handling | Medium | Multiple files | Centralized error middleware |
| Magic numbers | Low | `dashboard.jsx`, `pipeline.py` | Named constants |
| No type hints | Medium | `generate_failures.py` | Add gradually |
| Long functions | Medium | Several > 200 lines | Extract to smaller functions |

### 13.2 Testing Gaps

| Gap | Coverage | Recommendation |
|-----|----------|----------------|
| No integration tests for TimescaleDB | Low | Add integration tests |
| No E2E tests for remediation execution | Medium | Add execute API tests |
| No load testing | Medium | Add k6 or Locust tests |
| No security tests | Low | Add OWASP ZAP scan |

### 13.3 Documentation Gaps

| Gap | Status | Recommendation |
|-----|--------|----------------|
| No API documentation site | OpenAPI/Swagger | Add `fastapi-docs` or Redoc |
| No architecture diagrams | Needed | Add C4 model diagrams |
| No runbook for deployment | Needed | Add ops runbook |
| PRD.md is authoritative | ✓ Good | Keep as single source of truth |

### 13.4 Infrastructure Gaps

| Gap | Status | Recommendation |
|-----|--------|----------------|
| No TLS/HTTPS | Security risk | Add reverse proxy with TLS |
| No backup strategy | Risk | Add TimescaleDB backup |
| No monitoring/alerting | Operations risk | Add Prometheus + Grafana |
| No log aggregation | Operations risk | Add ELK/Loki stack |

---

## 14. Future Roadmap

### 14.1 Phase 9: Production Hardening

1. **Authentication & Authorization**
   - OAuth2/JWT integration
   - Role-based access control (RBAC)
   - API key management

2. **High Availability**
   - Horizontal scaling for backend
   - TimescaleDB replication
   - Redis Sentinel/Cluster

3. **Observability**
   - Prometheus metrics for all services
   - Grafana dashboards
   - Structured logging (JSON)
   - Distributed tracing (Jaeger)

### 14.2 Phase 10: ML Enhancements

1. **Model Improvements**
   - Pre-trained model weights (save/load)
   - Online learning for concept drift
   - Ensemble model selection

2. **Advanced Causal Inference**
   - Deconfounded causal discovery
   - Counterfactual reasoning
   - Temporal causal graphs

3. **MLflow Integration**
   - Experiment tracking
   - Model versioning
   - A/B testing framework

### 14.3 Phase 11: Platform Features

1. **Multi-tenancy**
   - Tenant isolation
   - Shared infrastructure
   - Billing/usage tracking

2. **Integrations**
   - PagerDuty webhook
   - Slack notifications
   - JIRA ticket creation
   - ServiceNow CMDB

3. **Self-Healing**
   - Automated Tier 1 execution
   - Rollback orchestration
   - Chaos engineering integration

### 14.4 Phase 12: Research & Publication

1. **Academic Contributions**
   - Novel KHBN architecture
   - Cross-dataset transfer learning
   - Conference paper draft

2. **Benchmark**
   - Public dataset release
   - Leaderboard
   - Reproducibility study

---

## Appendix A: File Index

| Path | Lines | Purpose |
|------|-------|---------|
| `src/pipeline.py` | 1265+ | 13-stage orchestrator |
| `src/api/server.py` | 804 | FastAPI server (17 endpoints) |
| `src/models/causal_inference.py` | 522 | Granger + PC + FDR |
| `src/reporting/nlg_generator.py` | 533 | Jinja2 + spaCy NLG |
| `src/remediation/remediation_engine.py` | 502 | 3-tier safety engine |
| `src/models/khbn.py` | 442 | Bayesian network |
| `src/models/logbert.py` | 399 | DistilBERT MLM |
| `src/models/temporal_transformer.py` | 325 | Transformer forecaster |
| `frontend/src/Dashboard.jsx` | 288 | React orchestrator |
| `frontend/src/components/RCAReportPanel.jsx` | 756 | Report + markdown |
| `frontend/src/components/CausalGraph.jsx` | 322 | D3.js visualization |
| `scripts/generate_failures.py` | 1101 | 10 scenario generator |
| `config/remediation_rules.yaml` | 794 | 10 remediation plans |
| `PRD.md` | 1185+ | Full requirements |

---

## Appendix B: Dependency Tree

```
fastapi
├── uvicorn (ASGI server)
├── pydantic (validation)
└── starlette (framework)

transformers
├── torch (CPU inference)
├── tokenizers
└── huggingface_hub

pgmpy
├── networkx
└── scipy

causal-learn
└── scipy

statsmodels
├── numpy
└── pandas

scikit-learn
├── numpy
└── scipy

drain3
├── tqdm
└── pygtrie

jinja2
└── markupsafe

frontend (React 18)
├── vite
├── axios
├── d3
├── recharts
└── lucide-react
```

---

## Appendix C: Environment Setup

### Development
```bash
# Clone and setup
git clone https://github.com/your-repo/testingrca.git
cd testingrca

# Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Frontend
cd frontend
npm install

# Generate test data
python scripts/generate_failures.py --scenario all --count 20
python scripts/generate_metrics.py

# Run backend
uvicorn src.api.server:app --reload

# Run frontend (separate terminal)
cd frontend && npm run dev
```

### Docker Production
```bash
# Build and run
docker compose up --build

# Scale backend
docker compose up --scale backend=3

# View logs
docker compose logs -f backend

# Run tests
docker compose exec backend pytest

# Stop
docker compose down -v
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Authors:** Yashash Mathur (with Claude Code analysis)
