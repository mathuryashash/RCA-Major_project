**PRODUCT REQUIREMENTS DOCUMENT - v2.0**

**Automated Root Cause Analysis (RCA) System**

**for Production Failures**

_DISTINCTION-LEVEL BUILD · Log Analysis · Deep Learning · Causal AI · Intelligent Remediation · AIOps_

| **Aditya Prakash**       | **Shresth Modi**          | **Utsav Upadhyay**   | **Yashash Mathur**     |
| ------------------------ | ------------------------- | -------------------- | ---------------------- |
| _ML & Anomaly Detection_ | _Causal Inference Engine_ | _Data Layer & Infra_ | _Frontend & API Layer_ |

Final Year B.Tech / M.Tech - Computer Science & Engineering

March 2026 · Version 2.0 · Distinction-Level Scope

**Table of Contents**

# **1\. Executive Summary**

Modern distributed systems - spanning cloud microservices, data pipelines, financial trading platforms, and industrial IoT - fail in complex, non-obvious ways. A single misconfigured database index can silently degrade query performance for hours before triggering a user-facing outage, exactly as documented in the e-commerce case study where a 2 PM schema migration caused a \$50,000 revenue loss unfolding across 7 hours. Traditional monitoring tools detect that something failed; they cannot explain why.

This document specifies the requirements for an Automated Root Cause Analysis (RCA) System - a production-quality, AI-powered pipeline that ingests multivariate time-series metrics, event logs, and system dependency graphs to automatically identify, rank, and narrate the root causes of production failures. The system targets the Distinction Level of success, incorporating a Knowledge-Informed Hierarchical Bayesian Network (KHBN), PersonalisedPageRank root cause scoring, Prometheus/CloudWatch real integration, Natural Language Generation (NLG) report output, and full MLflow experiment tracking.

**Feasibility:** Technical feasibility assessed at 85% (HIGH). Primary constraints - labelled ground-truth data and GPU availability - are mitigated through synthetic failure generation (≥200 labelled scenarios) and free-tier cloud GPU (Google Colab). All core libraries are open-source and production-ready.

The system is validated against five real-world case studies (E-Commerce DB Migration, SaaS Memory Leak, Financial Split-Brain, IoT Thread Pool Exhaustion, DNS Regional Outage) and is capable of identifying root causes within 5 minutes of failure onset across 50+ distinct anomaly types.

| **Dimension**             | **Specification**                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Target Level**          | Distinction-Level (Production-Quality System)                                                                                                    |
| **Team**                  | Aditya Prakash · Shresth Modi · Utsav Upadhyay · Yashash Mathur                                                                                  |
| **Architecture**          | 8-Module AI Pipeline (Log Ingest → Metric Ingest → Preprocess & Parse → Detect → Infer → Rank → Serve → Remediate)                               |
| **Primary Input Sources** | Local log files (.log, .json, .txt), Prometheus/CloudWatch metrics, CI/CD event webhooks                                                         |
| **Anomaly Coverage**      | 50+ types across CPU, Memory, Storage/IO, Network, Application layers                                                                            |
| **Core Models**           | LSTM Autoencoder + Temporal Transformer (metrics) · LogBERT / TF-IDF + Drain3 (log text) · KHBN + PageRank (causal)                              |
| **Causal Methods**        | Granger Causality + PC Algorithm + Knowledge-Informed Bayesian Network                                                                           |
| **Real Integrations**     | Prometheus, CloudWatch, CI/CD webhooks, MLflow                                                                                                   |
| **Output**                | Ranked root causes with confidence scores, causal chains, NLG narratives, auto-fix commands, remediation walkthroughs, and prevention checklists |
| **Performance Targets**   | \>85% detection precision · >75% recall · >70% Top-1 RCA accuracy · <5 min latency                                                               |
| **Hardware Requirement**  | Inference: 4 GB RAM, 2 CPU cores (no GPU); Training: Colab free tier GPU                                                                         |

# **2\. Team, Roles & Module Ownership**

This is a 4-member final-year project. Each member owns one primary system layer and acts as reviewer for an adjacent layer. The following table specifies ownership, deliverables, and collaboration interfaces.

| **Member**         | **Primary Role**                | **Owned Modules**                                                                                                                                               | **Key Deliverables**                                                                                                            | **Reviewer For**                  |
| ------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| **Aditya Prakash** | ML Engineer - Anomaly Detection | Module 3: LSTM AE + Temporal Transformer (metrics); LogBERT / TF-IDF sequential log model (text); Model training pipeline                                       | Trained model weights (.pt), anomaly scoring API (metrics + logs), MLflow runs, ablation study                                  | Causal Inference Engine (Shresth) |
| **Shresth Modi**   | AI Engineer - Causal Inference  | Module 4: Granger causality, PC algorithm, KHBN, PageRank ranker                                                                                                | Causal DAG construction, root cause JSON output, KHBN configuration                                                             | Anomaly Detection (Aditya)        |
| **Utsav Upadhyay** | Data & Infrastructure Engineer  | Module 1A: Log File Ingestion (tail, parse, template extract); Module 1B: Metric Ingestion (Prometheus/CloudWatch); Module 2: Preprocessing; Docker/TimescaleDB | Log collector daemon, Drain3 log parser, synthetic data generator, real metric collector, HDF5 datasets, docker-compose.yml     | API & Dashboard (Yashash)         |
| **Yashash Mathur** | Full-Stack & API Engineer       | Module 5: NLG Report Generator; Module 6: FastAPI backend, React dashboard; Module 7: Remediation Engine                                                        | REST API, D3.js causal graph view, RCA report panel, remediation walkthroughs, auto-fix executor, runbook generator, demo video | Data & Infrastructure (Utsav)     |

## **2.1 Collaboration Model**

- Weekly sync meetings (Mondays, 1 hour) to review cross-module interfaces and unblock dependencies.
- Shared GitHub repository with branch-per-module; PRs require one cross-member review before merge.
- Shared MLflow tracking server (hosted on university VM or free-tier cloud) accessible to all members.
- Interface contract documents maintained in /docs/interfaces/ - updated whenever a module API changes.

# **3\. Problem Statement**

## **3.1 The Gap in Existing Solutions**

When a complex system fails, the evidence is buried across gigabytes of log files - application logs, system logs, database logs, container logs - spread across multiple files and services. Engineers must manually search through thousands of lines of unstructured text to reconstruct what happened. Current industry monitoring tools (Datadog, Prometheus Alertmanager, PagerDuty, CloudWatch Alarms) excel at detecting that a system has failed - they raise alerts when metrics cross static thresholds. However, they are fundamentally reactive and symptom-focused and cannot answer the questions that matter most during an incident:

| **Question**                                 | **Why Existing Tools Cannot Answer It**                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Why did this failure occur?                  | Threshold alerts only observe the moment of breach, not the causal chain leading to it           |
| Which component initiated the cascade?       | Tools report all correlated metrics simultaneously - they cannot distinguish root from symptom   |
| What upstream event triggered the failure?   | No temporal event-to-metric correlation; deployment logs and metrics live in separate silos      |
| How did a change 6 hours ago cause this?     | Delayed causal effects (hours to days) are invisible to threshold-based systems                  |
| What do the log files actually say happened? | Standard tools count log errors - they do not understand event sequences or cross-file causality |
| Which log line is the real smoking gun?      | grep and regex searches return hundreds of matches; nothing ranks them by causal importance      |
| What should we fix first?                    | No causal ranking; all alerts appear equally urgent                                              |

## **3.2 Technical Challenges**

Automated RCA is fundamentally hard because real production systems exhibit the following characteristics simultaneously:

- **Unstructured Log Text - logs are free-form strings; the same error appears in dozens of different phrasings across services.**
- Example: 'Connection refused', 'ECONNREFUSED 127.0.0.1:5432', 'could not connect to server' all mean the same thing.
- **Cross-File Causality - the root cause lives in one log file (db.log) while the symptom manifests in another (app.log).**
- Standard tools treat each log file in isolation; this system correlates events across all files by timestamp.
- **Partial Observability - not all signals can be instrumented; many causal factors are latent.**
- Example: A memory leak in a shared library may not surface as a direct metric for hours.
- **High Dimensionality - 50-100+ correlated metrics spike simultaneously during incidents.**
- Distinguishing causal metrics from symptomatic ones requires statistical causal inference, not correlation.
- **Temporal Delays - root causes and visible symptoms are often separated by hours.**
- Example: A DB schema migration at 14:00 causes user-facing failure at 21:00 (7-hour lag).
- **Noisy Data - sensor noise, missing values, and irregular sampling obscure true signals.**
- **Rare Failures - labelled failure events are sparse; supervised learning is impractical.**
- **Cascading Effects - a single root cause propagates through multiple intermediary systems.**
- Example: DB slow query → connection pool exhaustion → API timeout → retry storm → memory exhaustion → outage.
- **Correlation vs. Causation - most metrics are correlated during failures; only some are causal.**

# **4\. Goals, Non-Goals & Success Levels**

## **4.1 Core Goals (Distinction Level)**

- Read and aggregate log files from multiple configurable paths on the host machine (application logs, system logs, container logs, database logs) in real-time via a file-tailing daemon.
- Parse unstructured log text into structured event sequences using log template extraction (Drain3) and NLP embeddings (TF-IDF / LogBERT).
- Detect sequential log anomalies - patterns of log events that deviate from normal sequences learned during the healthy-operation training period.
- Correlate log anomaly signals with time-series metrics (CPU, memory, latency) to build a unified cross-signal causal picture.
- Ingest realistic multivariate time-series metrics from simulated and real production environments (Prometheus/CloudWatch).
- Detect 50+ distinct anomaly types in an unsupervised manner without requiring labelled failure examples.
- Apply Granger causality + PC algorithm to distinguish correlation from causation and build a causal DAG.
- Deploy a Knowledge-Informed Hierarchical Bayesian Network (KHBN) for topology-aware causal reasoning.
- Rank root cause candidates using Personalised PageRank over the causal graph for accuracy beyond heuristics.
- Generate human-readable NLG causal chain narratives automatically from ranked RCA output.
- Automatically classify every identified root cause fix by safety tier and execute safe remediation actions without human intervention.
- Generate prioritised step-by-step remediation walkthroughs with exact kubectl/systemctl/SQL commands for complex issues requiring human execution.
- Produce deployment, configuration, and DB migration rollback instructions tailored to the specific root cause and environment.
- Output a long-term prevention checklist covering architectural changes, alerting improvements, and process guardrails.
- Expose results via a FastAPI REST API and an interactive React dashboard with D3.js causal graph visualization.
- Deliver ≥200 labelled synthetic failure scenarios with ground-truth causal chains for evaluation.

## **4.2 Non-Goals**

- Automated chaos simulation is not the end product - the Chaos/Synthetic Failure Generator is an internal testing tool only, used to produce labelled log and metric datasets to train and evaluate the AI. The end product is the Log Analysis Agent that reads real log files.
- Automated remediation of code-level bugs - the system recommends code fixes but cannot patch application source code.
- Real-time streaming at production scale - the system operates on recent historical windows (default: 60-120 min).
- Replacing SREs - this is a decision-support and time-reduction tool, not a replacement for human judgment.
- Unstructured natural-language log parsing - the system handles structured JSON event logs, not raw log lines.
- Multi-cloud production deployment - the project targets a local or university-hosted demo environment.

## **4.3 Three-Level Success Framework**

| **Level**                        | **Key Criteria**                                                                                                                     | **Status in This PRD**               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| **Pass (MVP)**                   | 5-10 metrics, synthetic data only, LSTM AE anomaly detection, heuristic root cause ranking, basic dashboard                          | Included as Phase 1-3 baseline       |
| **Distinction (Target)**         | Full 50-anomaly catalog, KHBN + PageRank ranker, Prometheus real integration, NLG reports, MLflow tracking, 5 case studies validated | PRIMARY SCOPE of this PRD            |
| **Publication-Worthy (Stretch)** | Novel KHBN architecture contribution, cross-dataset transfer learning, human-in-the-loop feedback loop, conference paper draft       | Stretch goals marked in requirements |

# **5\. System Architecture**

The system is organized as a dual-stream, nine-module intelligence pipeline. Two parallel ingestion streams - a Log File Stream and a Metrics Stream - converge at the Anomaly Detection layer and flow through a unified Causal Inference, Ranking, Serving, and Remediation stack. The end product is the Log + Metric Analysis Agent. The Synthetic Failure Generator is a testing and training utility only - it is not shipped as part of the product.

**Core Design Principle:** Log files are the primary input. Configure the system by pointing it at file paths (e.g., /var/log/syslog, ./logs/app.log, /var/log/containers/\*.log) in config.yaml. The agent tails these files in real time, parses unstructured text into structured event sequences, and correlates them with system metrics by timestamp to reason across both streams for root cause identification.

| **Module** | **Name**                                | **Primary Responsibility**                                                                                                                                                                    | **Owner**         | **Key Technologies**                                                     |
| ---------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------ |
| **M1A**    | Log File Ingestion                      | Tail and read log files from configured paths in real time; parse timestamps, log levels, and raw messages; support .log/.json/.txt; buffer parsed events                                     | Utsav Upadhyay    | Python watchdog (inotify), regex, JSON parser, config.yaml path list     |
| **M1B**    | Metric Ingestion                        | Pull metrics from Prometheus/CloudWatch; receive CI/CD webhook events; write to Redis Streams buffer                                                                                          | Utsav Upadhyay    | Python asyncio, Prometheus API, boto3 CloudWatch, Redis Streams          |
| **M2A**    | Log Preprocessing & Template Extraction | Convert raw log text to structured event sequences via Drain3 log parser; extract log templates (e.g., 'User \* failed login'); assign integer event IDs; build per-file event count matrices | Utsav Upadhyay    | Drain3, pandas, NumPy, regex, TimescaleDB log_events table               |
| **M2B**    | Metric Preprocessing                    | Gap-fill, z-score normalise per rolling 7-day window, sliding window segmentation (60-step); output (batch, 60, n_metrics) HDF5 arrays                                                        | Utsav Upadhyay    | Pandas, NumPy, SciPy, TimescaleDB, DVC                                   |
| **M3**     | Deep Learning Anomaly Detection         | LSTM AE + Transformer for metric anomaly scoring; TF-IDF / LogBERT sequential model for log event sequence anomaly scoring; unified anomaly_score ∈ \[0,1\] per signal                        | Aditya Prakash    | PyTorch 2.x, HuggingFace Transformers, scikit-learn, MLflow              |
| **M4**     | Causal Inference Engine                 | Granger causality across both log event-count signals and metric signals; PC algorithm + KHBN; cross-file event correlation in ±2-min windows                                                 | Shresth Modi      | statsmodels, causal-learn, pgmpy, NetworkX                               |
| **M5**     | Root Cause Ranker + NLG                 | Personalised PageRank + rarity-weighted log event scoring; NLG narrative citing specific log lines and metric deviations as evidence                                                          | Shresth / Yashash | NetworkX (PageRank), Jinja2, spaCy                                       |
| **M6**     | API & Dashboard                         | FastAPI REST backend; React dashboard; D3.js causal graph; log event timeline with highlighted anomalous lines; metric heatmap; RCA panel                                                     | Yashash Mathur    | FastAPI, Uvicorn, React 18, D3.js, Recharts, Docker                      |
| **M7**     | Intelligent Remediation Engine          | Safety-classify each fix action (Tier 1/2/3); auto-execute safe reversible actions; generate step-by-step Tier 2 walkthroughs with exact commands; produce long-term prevention checklists    | Yashash Mathur    | Safety rule engine YAML, kubectl SDK, psycopg2, Jinja2 runbook templates |

## **5.1 Dual-Stream Data Flow Sequence**

- LOG STREAM: File tailing daemon (M1A) detects new log lines via inotify; reads and buffers raw lines with file path and wall-clock timestamp.
- LOG STREAM: M2A Drain3 parser extracts log templates from raw text; assigns event IDs; stores structured (timestamp, file, event_id, raw_line) records in TimescaleDB log_events table.
- LOG STREAM: M3 TF-IDF vectoriser or LogBERT encodes sequential log event windows; reconstruction error or sequence perplexity yields log anomaly score ∈ \[0,1\] per file per window.
- METRIC STREAM: M1B pulls metrics from Prometheus/CloudWatch and CI/CD webhooks; writes to Redis Streams.
- METRIC STREAM: M2B normalises and windows metric data into (batch, 60, n_metrics) HDF5 arrays.
- METRIC STREAM: M3 LSTM Autoencoder + Temporal Transformer produce metric anomaly score ∈ \[0,1\] per signal.
- CONVERGENCE: All signals (log anomaly scores + metric anomaly scores) with score > 0.5 enter M4; Granger F-tests run pairwise across the unified signal set; PC algorithm builds causal DAG.
- KHBN enriches the DAG with service topology prior; Personalised PageRank in M5 scores each node; NLG engine generates a narrative citing the specific anomalous log lines and metric deviations.
- M6 FastAPI serves results to React dashboard: log event timeline, causal graph, ranked RCA panel with supporting log line evidence.
- M7 Remediation Engine receives top-ranked root cause; classifies fix actions by safety tier; auto-executes Tier 1 actions; generates Tier 2 walkthroughs and Tier 3 prevention checklist.

## **5.2 Architectural Decision Log**

| **Decision**           | **Chosen Approach**                                                  | **Alternative Considered**                   | **Rationale**                                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Log parsing method     | Drain3 online log template extraction                                | Manual regex per log format, Spell, LenMa    | Drain3 is format-agnostic and online (no retraining when new log templates appear); produces stable event IDs for downstream ML                                                  |
| Log anomaly model      | TF-IDF sequential baseline (MVP) + LogBERT fine-tune (distinction)   | Pure regex error counting, DeepLog LSTM      | TF-IDF is fast and interpretable; LogBERT captures semantic meaning across log templates; DeepLog is a viable alternative but requires more labelled data                        |
| Metric anomaly model   | LSTM Autoencoder (primary) + Temporal Transformer (secondary)        | Isolation Forest, Prophet, STL decomposition | LSTM captures complex temporal dependencies; Transformer handles longer-range patterns; IF/Prophet kept as ablation baselines                                                    |
| Cross-stream causality | Granger causality on unified signal set (log event counts + metrics) | Treat log and metric streams separately      | Unified Granger allows the system to discover that a log event (e.g., 'DB migration applied') Granger-causes a metric spike (e.g., query latency) - the key cross-stream insight |
| Log file configuration | config.yaml path list (user-specified)                               | Auto-discovery via syslog socket             | Explicit path list gives users full control and avoids ingesting irrelevant system logs; auto-discovery available as opt-in stretch goal                                         |
| Root cause ranking     | Personalised PageRank + rarity weighting                             | Heuristic causal outflow-inflow scoring      | PageRank handles graph structure; rarity weighting ensures a rare log event (e.g., a config change) ranks above a common warning                                                 |

# **6\. Comprehensive Anomaly Detection Catalog (50 Types)**

The RCA system is designed to detect, score, and trace the root causes of 50 distinct anomaly types across five infrastructure and application layers. Each anomaly type has a documented detection signature, expected anomaly score profile, and known typical root causes. This catalog forms the ground truth for evaluating the system's detection coverage.

**Detection Method:** All 50 anomaly types are detected via the LSTM Autoencoder reconstruction error and/or Temporal Transformer forecast deviation. No per-type rules or thresholds are handcrafted - the model learns normal behaviour and flags deviations automatically.

## **6.1 CPU & Compute Anomalies (10 Types)**

| **#** | **Anomaly Type**              | **Detection Signature**                                   | **Anomaly Profile**                     | **Typical Root Cause**                                                    |
| ----- | ----------------------------- | --------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------- |
| 1     | CPU Saturation                | Sustained CPU >90% beyond normal traffic baseline         | Spike: sharp onset, sustained           | Runaway process, infinite loop, compute-heavy underpinned task            |
| 2     | CPU Oscillation               | Rapid alternation between high and low CPU usage          | Periodic: oscillatory pattern           | Misconfigured auto-scaler, task thrashing between cores                   |
| 3     | Single-Core Bottleneck        | One vCPU at 100% while others remain idle                 | Spatial asymmetry in per-core metrics   | Single-threaded application on multi-core system; poor parallelism        |
| 4     | CPU Steal Time Spike          | High steal% in virtualised environment (>20%)             | Spike coinciding with neighbor activity | Cloud noisy-neighbor resource contention                                  |
| 5     | Context Switch Storm          | OS context switches >10x baseline per second              | Step-change in cs/s metric              | Too many threads, poor thread scheduling, lock contention                 |
| 6     | CPU Affinity Misconfiguration | Process consistently pinned to performance-degraded cores | Persistent latency on specific cores    | Incorrect CPU binding / taskset configuration                             |
| 7     | Thermal Throttling            | CPU frequency reduction (P-state downgrade) during load   | CPU freq drops while load stays high    | Cooling failure, data-centre AC fault, faulty thermal sensor              |
| 8     | Nice Value Anomaly            | Critical process starved; run-queue depth increases       | Latency spike with normal CPU%          | Process priority misconfiguration; batch job consuming nice-0 slots       |
| 9     | Interrupt Storm               | Hardware interrupt rate spikes >100k/sec                  | High /proc/interrupts rate, softirq CPU | Faulty NIC, disk controller malfunction, misconfigured interrupt affinity |
| 10    | CPU Cache Miss Rate Spike     | L2/L3 cache miss ratio increases >3x baseline             | Latency increase without CPU saturation | Poor memory access locality; recent code change affecting data structures |

## **6.2 Memory Anomalies (10 Types)**

| **#** | **Anomaly Type**            | **Detection Signature**                           | **Anomaly Profile**                              | **Typical Root Cause**                                              |
| ----- | --------------------------- | ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| 11    | Memory Leak                 | Monotonic upward memory trend without GC recovery | Slow ramp: trend anomaly over hours              | Application bug, unclosed file handles, orphaned objects            |
| 12    | Out of Memory (OOM)         | Sudden process termination; memory hits 100%      | Sharp spike to ceiling, then drop                | Memory leak reaching limit, traffic surge, undersized instance      |
| 13    | Swap Thrashing              | Excessive swap I/O; memory oscillating near limit | Sawtooth pattern: rss + swap                     | Insufficient RAM, working set exceeds physical memory               |
| 14    | Page Fault Storm            | High major page fault rate                        | Spike in pgmajfault/s                            | Working set exceeds RAM; application doing large random I/O         |
| 15    | Memory Fragmentation        | Free memory exists but large allocations fail     | Allocation failure despite free% >10%            | Long-running process with frequent alloc/free cycles                |
| 16    | Shared Memory Leak          | /dev/shm exhausted; new processes fail to launch  | Monotonic growth in /dev/shm usage               | Orphaned shared memory segments from crashed processes              |
| 17    | Buffer/Cache Explosion      | Kernel page cache consuming >80% RAM              | Ramp in cached memory, swap pressure builds      | Filesystem-heavy workload without memory pressure signals           |
| 18    | Huge Pages Misconfiguration | Periodic latency spikes every 30-120 seconds      | Latency sawtooth correlated with THP defrag      | Transparent Huge Pages defragmentation on latency-sensitive service |
| 19    | Memory Bandwidth Saturation | Memory bus saturated; CPU stalls on memory access | Latency spike without CPU saturation or disk I/O | Excessive concurrent in-memory data structures; NUMA-unaware code   |
| 20    | NUMA Node Imbalance         | Uneven memory allocation across NUMA nodes        | Remote NUMA memory accesses spike                | Poor NUMA affinity settings; OS memory allocator not NUMA-aware     |

## **6.3 Storage & I/O Anomalies (10 Types)**

| **#** | **Anomaly Type**            | **Detection Signature**                                                | **Anomaly Profile**                         | **Typical Root Cause**                                                      |
| ----- | --------------------------- | ---------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| 21    | Disk Space Exhaustion       | Storage utilisation >95%; write operations begin failing               | Ramp to ceiling                             | Log accumulation without rotation, unchecked database growth                |
| 22    | I/O Wait Spike              | CPU iowait% >30%; processes blocked on disk                            | Spike coincident with latency increase      | Slow disk, RAID rebuild, storage network congestion                         |
| 23    | Disk Latency Increase       | Read/write latency >10x baseline (e.g., 5ms → 100ms)                   | Step-change in await metric                 | Disk degradation, controller fault, HBA queue depth saturation              |
| 24    | IOPS Saturation             | Disk operations rate at device maximum                                 | Flat ceiling in IOPS with latency spike     | Excessive small random reads/writes; missing index causing full table scans |
| 25    | I/O Pattern Shift           | Sequential → random I/O pattern change detected                        | Shift in read/write size distribution       | Query plan change after index addition/removal; data layout change          |
| 26    | Inode Exhaustion            | No free inodes despite disk space available                            | Inode count at maximum                      | Too many small files (log files, temp files, containerised layers)          |
| 27    | Write Amplification         | SSD physical writes >> application writes                              | High wAmp ratio                             | Poor write patterns (many small random writes); SSD wear levelling          |
| 28    | Read-Ahead Misconfiguration | Inefficient I/O prefetching; high read latency on sequential workloads | Read latency inconsistency                  | Incorrect blockdev readahead setting for workload type                      |
| 29    | RAID Degradation            | Array running in degraded mode; rebuild I/O overhead                   | I/O latency spike with disk failure event   | Disk failure in RAID array; undetected silent disk errors                   |
| 30    | Filesystem Corruption       | Journal errors in syslog; stat calls failing intermittently            | Sporadic I/O errors correlated with reboots | Improper shutdown, power failure, hardware fault on write path              |

## **6.4 Network Anomalies (10 Types)**

| **#** | **Anomaly Type**       | **Detection Signature**                                    | **Anomaly Profile**                                | **Typical Root Cause**                                         |
| ----- | ---------------------- | ---------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| 31    | Packet Loss            | Dropped packets >1% on any interface                       | Spike in dropped/error counters                    | Network congestion, faulty switch port, full NIC buffer        |
| 32    | Bandwidth Saturation   | NIC at >95% rated capacity; throughput ceiling             | Flat ceiling in bytes/sec                          | Traffic surge, DDoS, backup job consuming all bandwidth        |
| 33    | TCP Retransmissions    | Retransmission rate >5% of segments                        | Spike in retrans/s                                 | Network instability, switch congestion, asymmetric routing     |
| 34    | Connection Reset Storm | TCP RST packets >1000/min; connections abruptly closed     | Burst of RST events                                | Load balancer misconfiguration, firewall state table overflow  |
| 35    | DNS Resolution Failure | DNS lookup latency >1s or >5% failure rate                 | Spike in dns_query_duration                        | DNS server overload, network partition isolating resolver      |
| 36    | NAT Table Exhaustion   | New connections failing; NAT table full                    | Connection failure rate spike                      | Too many concurrent connections; short-lived connection leak   |
| 37    | MTU Mismatch           | Packet fragmentation; throughput drop on large payloads    | Throughput anomaly for specific payload sizes      | Path MTU discovery failure; VPN/tunnel MTU smaller than LAN    |
| 38    | ARP Cache Overflow     | ARP table exceeds kernel limit; new hosts unreachable      | ARP lookup failures in logs                        | Too many hosts in flat subnet; ARP scanning or broadcast storm |
| 39    | Interface Errors       | CRC errors, frame errors, carrier sense errors >0          | Error counter non-zero (should be zero)            | Faulty Ethernet cable, SFP module failure, NIC hardware fault  |
| 40    | Asymmetric Routing     | Request and reply packets take different paths; TCP issues | Intermittent connection drops with topology change | Routing misconfiguration after failover; ECMP hash imbalance   |

## **6.5 Application Layer Anomalies (10 Types)**

| **#** | **Anomaly Type**           | **Detection Signature**                                  | **Anomaly Profile**                        | **Typical Root Cause**                                                             |
| ----- | -------------------------- | -------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- |
| 41    | Thread Pool Exhaustion     | All worker threads busy; request queue growing           | Spike to thread_pool_size ceiling          | Slow downstream dependency causing thread blocking; synchronous code in async path |
| 42    | Connection Pool Saturation | DB connections at maximum; new queries failing           | Spike to pool_size ceiling                 | Connection leak, insufficient pool size, long-running transactions                 |
| 43    | Garbage Collection Pause   | GC pause time >500ms; latency spikes correlated with GC  | Sawtooth latency pattern                   | Oversized heap, memory pressure, wrong GC algorithm                                |
| 44    | Event Loop Blocking        | Node.js/async event loop response time >100ms            | Latency step-change                        | Synchronous blocking operation in async code (e.g., fs.readFileSync)               |
| 45    | Circuit Breaker Activation | Circuit breaker trips; dependency call failure rate >50% | Binary: open/closed state change           | Downstream service failure or high latency triggering threshold                    |
| 46    | Rate Limiter Throttling    | HTTP 429 responses >1% of requests                       | Spike in 429 rate                          | Traffic surge, DDoS, crawler bot, misconfigured rate limit                         |
| 47    | Cache Stampede             | Simultaneous cache misses spike; DB load spikes          | Periodic spike after TTL expiry            | Popular cache entry expiration; missing probabilistic early expiry                 |
| 48    | Deadlock                   | Thread/process wait time grows indefinitely              | Monotonic increase in blocked thread count | Lock acquisition order bug; two resources locked in opposite order                 |
| 49    | Semaphore Contention       | Concurrency limiter backpressure; queue depth growing    | Ramp in queue depth metric                 | Shared resource bottleneck; semaphore count too low for load                       |
| 50    | Message Queue Backlog      | Unconsumed messages accumulating; consumer lag growing   | Monotonic ramp in queue_depth              | Consumer service slow/failed; producer too fast; batch job                         |

# **7\. Functional Requirements**

## **7.1 Module 1A - Log File Ingestion \[Owner: Utsav Upadhyay\]**

**End Product Clarification:** Log file ingestion is the core product feature. The Synthetic Failure Generator (FR-04) is an internal development and testing tool only. It produces labelled log files that are fed into the agent to verify it finds the correct root cause - it is not shipped to end users.

### **FR-01: Configuration-Based File Path Specification**

The system shall accept a YAML configuration file (config.yaml) specifying an array of log file paths to monitor. The configuration shall support absolute paths, relative paths, glob patterns (e.g., /var/log/containers/\*.log), and named log source labels. Example:

\# config.yaml - Log File Ingestion Configuration

log_sources:

\- label: application

path: ./logs/app.log

format: plaintext # or: json, syslog

\- label: database

path: ./logs/db.log

format: plaintext

\- label: system

path: /var/log/syslog

format: syslog

\- label: containers

path: /var/log/containers/\*.log

format: json

metric_sources:

prometheus_url: <http://localhost:9090>

scrape_interval_seconds: 60

### **FR-02: Real-Time File Tailing**

The log collector daemon shall monitor all configured file paths for new content using OS-level inotify events (Linux) or polling (macOS/Windows fallback) via the Python watchdog library. New lines shall be read and buffered within 1 second of being written. File rotation (log file replaced by a new file at the same path) shall be handled automatically by re-opening the file descriptor.

### **FR-03: Multi-Format Log Parsing**

The system shall parse log lines into structured records (timestamp, level, source_file, raw_message) supporting the following formats:

| **Format**          | **Example Raw Line**                                                            | **Parsing Method**                          |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| Plaintext timestamp | 2026-03-01 10:05:32 ERROR Connection refused to db:5432                         | Regex: ^(\\S+ \\S+) (\\w+) (.+)\$           |
| Syslog RFC3164      | Mar 1 10:05:32 hostname app\[1234\]: Connection refused                         | Regex: syslog pattern + hostname extraction |
| JSON log            | { "ts": "2026-03-01T10:05:32Z", "level": "error", "msg": "Connection refused" } | json.loads() + field mapping from config    |
| ISO 8601 timestamp  | 2026-03-01T10:05:32.456Z \[ERROR\] Connection refused                           | Regex: ISO 8601 datetime prefix             |
| Unix timestamp      | 1740823532 ERROR Connection refused                                             | int() conversion → datetime.fromtimestamp() |

Lines that cannot be parsed shall be stored as raw_message with level=UNKNOWN and flagged in a parse_failures counter exposed via GET /health.

## **7.2 Module 1B - Metric Ingestion \[Owner: Utsav Upadhyay\]**

### **FR-05: Prometheus Integration**

The system shall connect to a Prometheus /metrics endpoint via HTTP pull at configurable scrape intervals (default: 60 seconds). Connection failures shall be retried with exponential backoff (max 3 attempts). This is a complementary signal source - the system functions with log files alone if no Prometheus endpoint is configured.

### **FR-06: CloudWatch Integration**

The system shall optionally query AWS CloudWatch GetMetricData API for configured namespaces (EC2, RDS, Lambda, ECS). Authentication via IAM credentials or instance profiles. If no CloudWatch credentials are configured, this module is disabled gracefully.

### **FR-07: CI/CD Event Ingestion**

The system shall receive deployment and configuration change events via HTTP webhook (POST /events/deploy, POST /events/config). Events shall be stored as structured log-like records in TimescaleDB and treated as high-rarity signals in the causal inference engine.

### **FR-04: Synthetic Failure Generator (Internal Testing Tool)**

**Scope Note:** This is NOT a shipped product feature. It is an internal tool used during development to produce labelled log files and metric traces with known root causes, enabling evaluation of the agent's detection accuracy.

The generator shall produce realistic log file output mimicking application, database, and system logs for the 10 failure scenarios below, with ground-truth causal chain labels stored as JSONL:

For development and evaluation, the system shall include a Python script (generate_failures.py) that simulates a configurable multi-service topology and injects the following failure modes with configurable severity and propagation delays:

| **Failure Scenario**             | **Services Affected**      | **Propagation Delay** | **Ground Truth Label**     |
| -------------------------------- | -------------------------- | --------------------- | -------------------------- |
| Database Schema Migration        | DB + all API callers       | 0-7 hours             | db_migration_applied       |
| Memory Leak (WebSocket handler)  | API server instances       | 4-48 hours            | memory_leak_code_deploy    |
| Network Partition (inter-region) | Cross-region Redis cluster | 0-5 minutes           | network_partition_bgp      |
| Thread Pool Exhaustion           | IoT API + InfluxDB         | 30-90 seconds         | thread_pool_background_job |
| DNS Propagation Delay            | All services using DNS LB  | 0-30 minutes          | dns_ttl_misconfiguration   |
| CPU Saturation                   | Any single service         | Immediate             | cpu_runaway_process        |
| Connection Pool Saturation       | DB-connected services      | 15-60 minutes         | connection_leak_bug        |
| Cache Stampede                   | Cache + DB                 | At TTL expiry         | cache_ttl_expiry           |
| Disk Space Exhaustion            | Logging service            | Gradual over hours    | log_rotation_disabled      |
| Message Queue Backlog            | Consumer service           | Minutes to hours      | consumer_service_crash     |

The generator shall produce ≥200 labelled failure scenarios (≥20 per failure type) with ground-truth causal chains stored as JSONL labels.

## **7.3 Module 2A - Log Preprocessing & Template Extraction \[Owner: Utsav Upadhyay\]**

### **FR-08: Log Template Extraction (Drain3)**

The system shall apply the Drain3 online log parser to convert raw log messages into log templates. Drain3 groups semantically similar messages into templates by replacing variable tokens (IDs, IPs, usernames, numbers) with wildcards. Example:

| **Raw Log Message**                      | **Extracted Template**        | **Event ID**            |
| ---------------------------------------- | ----------------------------- | ----------------------- |
| User 12345 failed login from 192.168.1.1 | User \* failed login from \*  | EVT_047                 |
| Connection refused to postgres:5432      | Connection refused to \*:\*   | EVT_012                 |
| Connection refused to mysql:3306         | Connection refused to \*:\*   | EVT_012 (same template) |
| Allocated 2048MB for request #98765      | Allocated \*MB for request \* | EVT_089                 |

Extracted templates and their integer Event IDs shall be stored in TimescaleDB (log_templates table). The drain3 state shall be persisted to disk so that previously learned templates survive service restart.

### **FR-09: Event Count Matrix Construction**

For each log source file, the system shall construct a fixed-length sliding window of event counts: a matrix of shape (window_size, n_templates) where each row is a timestep and each column is the count of a specific Event ID in that timestep. This matrix is the input to the log anomaly detection model.

### **FR-10: Log Level & Rarity Scoring**

Each parsed log line shall be assigned a base rarity score based on: (a) log level weight (DEBUG=0.1, INFO=0.2, WARN=0.5, ERROR=0.8, CRITICAL=1.0), and (b) inverse document frequency across the training corpus (rare templates score higher). This rarity score is used as a prior weight in the causal root cause ranker.

## **7.4 Module 2B - Metric Preprocessing \[Owner: Utsav Upadhyay\]**

### **FR-12: Missing Data Handling**

The pipeline shall handle missing values as follows: gaps &lt;5 minutes shall be forward-filled; gaps 5-30 minutes shall be linearly interpolated; gaps &gt;30 minutes shall be flagged with a missing_data sentinel value and excluded from model inference windows.

### **FR-13: Normalization**

All metrics shall be z-score normalized per metric using rolling 7-day mean and standard deviation stored in TimescaleDB. Normalization parameters shall be updated daily to adapt to gradual operational drift.

### **FR-14: Windowing**

Metric data shall be segmented into fixed-length sliding windows of 60 timesteps (representing 1 hour at 1-minute resolution) with 50% overlap. Windows shall be stored as NumPy arrays of shape (batch, 60, n_metrics) in HDF5 format.

## **7.5 Module 3 - Deep Learning Anomaly Detection Engine \[Owner: Aditya Prakash\]**

**Dual-Model Approach:** Module 3 runs two parallel sub-pipelines - a Metric Anomaly sub-pipeline (LSTM AE + Transformer on numeric time-series) and a Log Anomaly sub-pipeline (TF-IDF / LogBERT on sequential event matrices). Both produce anomaly scores in \[0,1\] on the same timeline, which are unified before passing to the Causal Inference Engine.

### **FR-15: Log Sequential Anomaly Detection - TF-IDF Baseline**

The system shall train a TF-IDF vectoriser on event count matrices from the healthy log corpus. At inference time, each window's TF-IDF representation is compared against the training distribution using cosine distance. Windows with distance > 99th-percentile threshold are flagged as anomalous. This is the MVP log model - fast, interpretable, and requires no GPU.

### **FR-16: Log Sequential Anomaly Detection - LogBERT (Distinction Level)**

For the distinction-level build, a LogBERT model (BERT fine-tuned on log event sequences) shall be trained on healthy log event sequences. Anomaly detection uses masked log event prediction: events that the model assigns low probability to are flagged as anomalous. LogBERT captures semantic relationships between log templates that TF-IDF cannot. The model shall be fine-tuned on the project's synthetic log corpus using HuggingFace Transformers on Google Colab GPU.

### **FR-17: LSTM Autoencoder (Metric Stream)**

The system shall train an LSTM Autoencoder on 30-90 days of healthy metric data. Architecture: 2-layer encoder (128→64 units) with symmetric decoder; dropout 0.2; Adam optimizer; reconstruction loss = MSE per timestep per metric. Anomaly score per metric per window = sigmoid(reconstruction_error / 99th-percentile threshold).

### **FR-18: Temporal Transformer (Metric Stream)**

A multi-head Temporal Transformer (4 heads, d_model=128, 2 encoder layers) shall perform multi-step metric forecasting (horizon = 10 timesteps). Forecast deviation score = normalised MAE between forecast and actuals, providing a secondary anomaly signal sensitive to sudden step-changes that autoencoders may miss.

### **FR-19: Unified Anomaly Score**

The final unified anomaly score for each signal (log file or metric) shall be computed as: score = α × model_score + (1−α) × rarity_prior, where the rarity_prior for log signals is the log level and inverse-frequency weight from FR-10, and α = 0.80 (configurable). Signals with unified score > 0.5 are classified as anomalous and passed to the Causal Inference Engine.

### **FR-20: Anomaly Onset Detection**

The system shall identify the precise timestamp, source file or metric name, and the specific log line(s) or metric value(s) that first triggered the anomaly threshold. For log anomalies, the specific raw log line(s) contributing most to the anomaly score shall be extracted and stored as supporting_evidence for the NLG report.

### **FR-21: MLflow Experiment Tracking**

All training runs for both log and metric models shall log to MLflow: config.yaml snapshot, training data hash (SHA-256), per-epoch loss curves, validation precision/recall/F1 for both streams, and model artifacts. Separate MLflow experiments shall track log model runs and metric model runs for independent ablation.

## **7.6 Module 4 - Causal Inference Engine \[Owner: Shresth Modi\]**

### **FR-22: Unified Signal Set for Causal Analysis**

The causal inference engine shall treat log event-count time series (one per log template per file) and metric time series as a unified set of signals. This enables cross-stream causal discovery - for example, identifying that a log event 'DB migration applied' (from db.log) Granger-causes a metric spike in 'query_latency_p95' (from Prometheus). The cross-file and cross-stream event correlation window is ±2 minutes for log-to-log and ±30 minutes for event-to-metric.

### **FR-23: Granger Causality Testing**

The system shall apply pairwise Granger causality F-tests (statsmodels) across all anomalous signals (score >0.5) using lags 1-6. FDR correction (Benjamini-Hochberg) shall be applied. Significant edges (q < 0.05) populate the causal adjacency matrix.

### **FR-24: PC Algorithm Causal Structure Learning**

The PC algorithm (causal-learn library) shall orient edges of the causal DAG. PC result shall be combined with the Granger adjacency matrix via majority-vote edge weighting. Cycles shall be broken by removing the weakest edge.

### **FR-25: Knowledge-Informed Hierarchical Bayesian Network (KHBN)**

The service dependency topology (from CMDB or config YAML) shall serve as a structural prior for the Bayesian Network fitted via pgmpy. The KHBN posterior re-ranks root cause candidates, reducing spurious causal edges. CloudRCA (Alibaba) reported >20% reduction in SRE troubleshooting time; hybrid KG+CBN tools cut spurious correlations by 60.6% in industrial deployments.

### **FR-26: CI/CD Event and Log Event Correlation**

Deployment and configuration events (from FR-07) shall be treated as high-rarity log-like signals in the causal graph. An event correlated within ±30 minutes of anomaly onset receives a deployment_event_bonus of 0.30 in the root cause score. Specific log lines from the time of the event shall be extracted as supporting evidence.

## **7.7 Module 5 - Root Cause Ranker & NLG Report Generator \[Owner: Yashash Mathur\]**

### **FR-27: Personalised PageRank Root Cause Ranking**

The system shall compute Personalised PageRank (damping = 0.85) over the causal DAG, bias vector proportional to anomaly_score. Final RCA score:

_rca_score = 0.35 × pagerank_score + 0.25 × temporal_priority + 0.20 × anomaly_score + 0.10 × rarity_prior + 0.10 × event_bonus_

Top-K (default: 5) candidates returned with normalised confidence scores summing to 1.0.

### **FR-28: Causal Chain Serialization**

For each top-K candidate the system shall produce a JSON causal chain trace: root_cause node, chain of (from, to, p_value, lag_minutes, supporting_log_lines\[\]) tuples, confidence score, first_seen timestamp, and raw log line evidence extracted from FR-20.

### **FR-29: NLG Narrative Generation**

A Jinja2 template engine shall generate a human-readable narrative including: one-sentence summary, step-by-step causal chain with timestamps, cited log lines as evidence (e.g., 'At 18:30, db.log recorded: Connection pool at capacity - 100/100'), remediation recommendations, and prevention suggestions. Output: Markdown + plain text.

## **7.8 Module 6 - API & Dashboard \[Owner: Yashash Mathur\]**

### **FR-30: REST API Endpoints**

The FastAPI backend shall expose the following endpoints:

| **Endpoint**                             | **Method** | **Description**                                                                       | **Response Code** |
| ---------------------------------------- | ---------- | ------------------------------------------------------------------------------------- | ----------------- |
| POST /analyze                            | POST       | Trigger RCA on a time window; returns incident_id immediately                         | 202 Accepted      |
| GET /report/{incident_id}                | GET        | Full RCA: anomalies, causal chain JSON, ranked causes, NLG narrative, cited log lines | 200 OK            |
| GET /logs/{incident_id}                  | GET        | Return annotated log lines around the incident window, sorted by anomaly contribution | 200 OK            |
| GET /metrics/{service}                   | GET        | Last 60-minute anomaly score time series for a service                                | 200 OK            |
| GET /graph/{incident_id}                 | GET        | Causal DAG as node-link JSON for D3.js                                                | 200 OK            |
| POST /remediate/{incident_id}            | POST       | Trigger Remediation Engine; returns safety-classified action plan                     | 202 Accepted      |
| POST /remediate/{incident_id}/execute    | POST       | Confirm and execute pending Tier 1 auto-fix actions                                   | 200 OK            |
| GET /remediate/{incident_id}/walkthrough | GET        | Full step-by-step Tier 2 walkthrough with commands                                    | 200 OK            |
| GET /remediate/{incident_id}/prevention  | GET        | Long-term prevention checklist (Markdown + JSON)                                      | 200 OK            |
| GET /audit/{incident_id}                 | GET        | Immutable audit log of all auto-executed remediation actions                          | 200 OK            |
| GET /health                              | GET        | System health: models loaded, DB connected, log files accessible, data fresh          | 200 OK            |

### **FR-31: Log Event Timeline View**

The dashboard shall display a unified log + metric timeline. Log lines shall appear as event markers on the timeline, colour-coded by log level (blue=INFO, amber=WARN, red=ERROR/CRITICAL). Anomalous log lines identified by FR-20 shall be highlighted with a pulsing border. Clicking any log marker shall expand the raw log line and its anomaly contribution score.

### **FR-32: Anomaly Heatmap**

A multi-signal heatmap shall show anomaly scores across all monitored log files and metrics, colour-coded green/amber/red, with zoomable time windows and signal filtering by file label or service name.

### **FR-33: Interactive Causal Graph Viewer**

A D3.js force-directed DAG shall display log event nodes (square) and metric nodes (circle) together in the causal graph. Node colour indicates anomaly severity; node size is proportional to PageRank score; edge thickness is proportional to Granger F-statistic; edge labels show lag time. Root cause node is highlighted distinctly. Clicking a log node expands the supporting raw log lines.

### **FR-34: RCA Report Panel**

The ranked top-5 root cause panel shall display: rank badge, confidence score bar, causal chain path, NLG narrative excerpt, onset timestamp, and an expandable evidence section showing the exact log lines and metric values that constitute the supporting evidence.

### **FR-35: Incident Replay Mode**

Users shall be able to step through a failure scenario chronologically, watching log events and metric anomalies appear in sequence. Each step advances time by one configurable interval (default: 1 minute). The causal graph updates incrementally as new anomalies are detected.

## **7.7 Module 7 - Intelligent Remediation Engine \[Owner: Yashash Mathur\]**

**Design Philosophy:** The system never takes a destructive or irreversible action autonomously. Safe = reversible, low-blast-radius, well-understood. Complex = irreversible, multi-system, or requires human judgment. When in doubt, the engine defaults to guided walkthrough.

### **FR-25: Safety Classification Engine**

Every potential remediation action associated with a root cause shall be classified into one of three safety tiers before execution. Classification is based on a deterministic rule engine (YAML-configured) - no ML model is used for safety decisions to ensure auditability.

| **Safety Tier** | **Label**              | **Definition**                                                                     | **System Behaviour**                                                                                                     | **Examples**                                                                                        |
| --------------- | ---------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| **Tier 1**      | SAFE - Auto-Execute    | Reversible, single-service scope, no data loss risk, pre-validated for environment | System executes immediately, logs action with timestamp and before/after state, sends confirmation receipt               | Restart a crashed pod, flush a cache, kill a runaway process, reload a config without downtime      |
| **Tier 2**      | GUIDED - Walkthrough   | Irreversible OR multi-service scope OR requires environment-specific knowledge     | System generates numbered step-by-step walkthrough with exact commands but does NOT execute; engineer confirms each step | Rollback a deployment, increase connection pool size, modify DB index, update DNS TTL               |
| **Tier 3**      | ADVISORY - Inform Only | Requires architectural change, code fix, or organisation-level process change      | System explains the issue, provides recommendation and reasoning, links to relevant runbook or documentation             | Fix a memory leak in application code, redesign Redis quorum strategy, implement canary deployments |

### **FR-26: Immediate Command Generation (Tier 1 & 2)**

For each root cause in the top-K ranked list, the Remediation Engine shall generate platform-specific immediate commands covering the following command types:

| **Command Category**   | **Platform**          | **Example Generated Command**                                                     | **Tier** |
| ---------------------- | --------------------- | --------------------------------------------------------------------------------- | -------- |
| Pod/container restart  | Kubernetes            | kubectl rollout restart deployment/&lt;service-name&gt; -n &lt;namespace&gt;      | Tier 1   |
| Service restart        | systemd (Linux)       | sudo systemctl restart &lt;service-name&gt;                                       | Tier 1   |
| Process termination    | Linux                 | sudo kill -9 \$(pgrep -f &lt;process-pattern&gt;)                                 | Tier 1   |
| Cache flush            | Redis                 | redis-cli -h &lt;host&gt; FLUSHDB (scoped to anomalous service cache)             | Tier 1   |
| Deployment rollback    | Kubernetes            | kubectl rollout undo deployment/&lt;service-name&gt; --to-revision=&lt;N&gt;      | Tier 2   |
| Deployment rollback    | Docker Compose        | docker-compose up --scale &lt;service&gt;=0 && docker-compose up -d --no-recreate | Tier 2   |
| DB index query hint    | PostgreSQL            | SET enable_seqscan = off; -- force index: &lt;index_name&gt;                      | Tier 2   |
| Connection pool resize | PostgreSQL pg_bouncer | UPDATE pgbouncer.config SET pool_size=&lt;N&gt; WHERE db='&lt;dbname&gt;'; RELOAD | Tier 2   |
| DB migration rollback  | Alembic / Flyway      | alembic downgrade -1 OR flyway undo -target=&lt;version&gt;                       | Tier 2   |
| Config change rollback | Kubernetes ConfigMap  | kubectl apply -f configmap-&lt;prev-version&gt;.yaml                              | Tier 2   |
| DNS TTL update         | Route53 / generic     | aws route53 change-resource-record-sets ... TTL=60 (emergency low TTL)            | Tier 2   |
| Thread pool resize     | Java JVM flag         | Add -Dserver.tomcat.max-threads=&lt;N&gt; to service startup config               | Tier 2   |

Commands shall be generated using a Jinja2 template engine populated with environment-specific variables (namespace, service name, host, version) extracted from the M1 ingestion context and system configuration YAML. Commands shall never be hardcoded.

### **FR-27: Rollback Instruction Generator**

For root causes involving a recent deployment, configuration change, or database migration (identified via CI/CD event correlation in FR-16), the engine shall generate a complete rollback procedure covering three rollback dimensions:

- **Deployment Rollback - identifies the last stable version tag from the deployment event log; generates platform-appropriate rollback command (kubectl rollout undo, docker-compose service pin, or Helm rollback); includes health check verification steps post-rollback.**
- **Configuration Rollback - retrieves the previous configuration values from the event log (stored at ingestion time); generates a diff of changed keys; produces a configuration restore command or patch manifest.**
- **Database Migration Rollback - identifies the migration version from the deployment event; generates the down-migration command for the detected migration framework (Alembic, Flyway, Liquibase, or raw SQL); flags if a down-migration is irreversible (e.g., DROP TABLE) and escalates to Tier 3.**

### **FR-28: Guided Walkthrough Generator**

For Tier 2 (GUIDED) remediation actions, the engine shall generate a structured, numbered step-by-step walkthrough. Each step shall contain:

| **Field**             | **Description**                                   | **Example**                                                                          |
| --------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Step number           | Ordered sequence position                         | Step 3 of 7                                                                          |
| Title                 | One-line action summary                           | Roll back deployment to v2.14.2                                                      |
| Command               | Exact shell/SQL command to run                    | kubectl rollout undo deployment/product-api --to-revision=14                         |
| Expected output       | What the engineer should see if the step succeeds | deployment.apps/product-api rolled back                                              |
| Verification check    | How to confirm the step worked                    | kubectl rollout status deployment/product-api (should show: successfully rolled out) |
| Rollback of this step | How to undo THIS step if something goes wrong     | kubectl rollout undo deployment/product-api (re-apply the problematic version)       |
| Estimated time        | Time this step typically takes                    | ~2 minutes                                                                           |
| Safety note           | Any risk or precaution for this step              | Ensure no active transactions on the order service before proceeding                 |

### **FR-29: Long-Term Prevention Checklist**

Following every RCA report, the engine shall generate a long-term prevention checklist tailored to the identified root cause category. The checklist shall be divided into three horizons:

| **Horizon** | **Timeframe** | **Focus**                                                   | **Example Items**                                                                                                                                        |
| ----------- | ------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Immediate   | 0-24 hours    | Stop the bleeding, prevent recurrence of this exact failure | Add index usage monitoring alert; increase connection pool to 200; add DB migration canary check to CI/CD pipeline                                       |
| Short-Term  | 1-4 weeks     | Architectural hardening and process guardrails              | Implement pre-migration query performance benchmarks; enforce schema change review board; add load testing stage to deployment pipeline                  |
| Long-Term   | 1-3 months    | Systemic resilience improvements                            | Implement chaos engineering programme; adopt blue/green deployments for all DB-touching services; build automated rollback triggers on error rate breach |

The prevention checklist shall be exportable as a Markdown file and shall include estimated effort, suggested owner (SRE, Dev, DBA, Platform), and priority score based on incident severity and recurrence probability.

### **FR-30: Remediation Audit Log**

Every auto-executed Tier 1 action shall be written to an immutable audit log (append-only TimescaleDB table: remediation_actions) containing: incident_id, action_type, command_executed, executor (system/human), before_state snapshot, after_state snapshot, timestamp, and outcome (success/failure/timeout). This log is exposed via GET /audit/{incident_id} and displayed in the dashboard Remediation History tab.

### **FR-31: Remediation Confidence Gate**

The Remediation Engine shall only propose or execute remediation actions when the top-ranked root cause confidence score exceeds a configurable threshold (default: 0.70). Below this threshold, the system shall surface a low-confidence warning and present the remediation as advisory only, requiring explicit human confirmation before any Tier 1 auto-execution.

### **FR-32: Human Confirmation Interface**

The React dashboard shall include a Remediation Action Panel that displays all pending Tier 1 auto-executions with a 30-second countdown timer and a Cancel button. Tier 2 walkthroughs shall be presented as an interactive step checklist where engineers mark each step complete before unlocking the next. Tier 3 advisories shall be displayed in a read-only recommendation card.

# **8\. Non-Functional Requirements**

| **ID**     | **Category**    | **Requirement**                                              | **Target**                                        | **Owner** |
| ---------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------- | --------- |
| **NFR-01** | Performance     | Anomaly detection inference latency (60-min window)          | < 5 seconds on CPU                                | Aditya    |
| **NFR-02** | Performance     | Causal graph construction (20 signals, lag 6)                | < 30 seconds                                      | Shresth   |
| **NFR-03** | Performance     | Full RCA pipeline (detect → rank → NLG)                      | < 5 minutes post-failure                          | All       |
| **NFR-04** | Performance     | LSTM AE training time on healthy dataset (30 days)           | < 20 min CPU / < 3 min GPU                        | Aditya    |
| **NFR-05** | Performance     | Dashboard initial load                                       | < 3 seconds                                       | Yashash   |
| **NFR-06** | Accuracy        | Anomaly detection precision on injected failures             | \> 85%                                            | Aditya    |
| **NFR-07** | Accuracy        | Anomaly detection recall on injected failures                | \> 75%                                            | Aditya    |
| **NFR-08** | Accuracy        | Root cause Top-1 identification accuracy                     | \> 70% (distinction target)                       | Shresth   |
| **NFR-09** | Accuracy        | Root cause Top-3 identification accuracy                     | \> 88%                                            | Shresth   |
| **NFR-10** | Accuracy        | False positive rate on normal operation periods              | < 5%                                              | Aditya    |
| **NFR-11** | Scalability     | Maximum supported signals without OOM                        | 50 signals, 10,000 timesteps                      | Utsav     |
| **NFR-12** | Scalability     | Maximum services in causal graph                             | 20 services                                       | Shresth   |
| **NFR-13** | Reliability     | Inference-mode hardware requirement                          | 4 GB RAM, 2 CPU cores (no GPU)                    | Utsav     |
| **NFR-14** | Reliability     | GPU required only for                                        | Offline training (Colab free tier)                | Aditya    |
| **NFR-15** | Portability     | Target OS                                                    | Linux / macOS via Docker Compose                  | Utsav     |
| **NFR-16** | Reproducibility | All experiments reproducible with fixed random seed          | Seed in config.yaml                               | Aditya    |
| **NFR-17** | Reproducibility | Dataset versioned with DVC; hash in every MLflow run         | SHA-256 data hash logged                          | Utsav     |
| **NFR-18** | Observability   | All API requests logged with structured JSON logging         | ELK or stdout JSON                                | Yashash   |
| **NFR-19** | Remediation     | Tier 1 auto-execution latency from trigger to completion     | < 10 seconds                                      | Yashash   |
| **NFR-20** | Remediation     | Tier 2 walkthrough generation latency                        | < 3 seconds                                       | Yashash   |
| **NFR-21** | Remediation     | Prevention checklist generation latency                      | < 5 seconds                                       | Yashash   |
| **NFR-22** | Remediation     | Safety classification accuracy (Tier assignment correctness) | 100% - safety rules are deterministic and audited | Yashash   |
| **NFR-23** | Remediation     | Remediation only triggered when root cause confidence        | ≥ 0.70 threshold (configurable)                   | Yashash   |
| **NFR-24** | Remediation     | All Tier 1 auto-actions written to immutable audit log       | Append-only, no delete API                        | Yashash   |

# **9\. Technical Stack**

| **Category**                  | **Technology**                        | **Version**         | **Purpose**                                                            | **Owner**         |
| ----------------------------- | ------------------------------------- | ------------------- | ---------------------------------------------------------------------- | ----------------- |
| **Log File Tailing**          | Python watchdog                       | 4.x                 | inotify-based real-time file monitoring for log ingestion              | Utsav             |
| **Log Parsing**               | Drain3                                | 0.9+                | Online log template extraction - format-agnostic, no retraining needed | Utsav             |
| **Log NLP (MVP)**             | scikit-learn TF-IDF                   | 1.4+                | Fast, interpretable log sequential anomaly detection baseline          | Aditya            |
| **Log NLP (Distinction)**     | HuggingFace Transformers (LogBERT)    | 4.x                 | Semantic log anomaly detection via masked event prediction fine-tuning | Aditya            |
| **Language**                  | Python                                | 3.11+               | Core system, ML, data processing, CLI scripts                          | All               |
| **Deep Learning**             | PyTorch                               | 2.x                 | LSTM Autoencoder, Temporal Transformer training & inference            | Aditya            |
| **Data Processing**           | Pandas + NumPy                        | 2.x / 1.26          | Windowing, normalization, statistical computation                      | Utsav             |
| **Statistical Inference**     | statsmodels                           | 0.14+               | Granger causality F-tests, ADF stationarity tests                      | Shresth           |
| **Causal Structure Learning** | causal-learn                          | 0.1.3+              | PC algorithm for causal DAG orientation                                | Shresth           |
| **Bayesian Networks**         | pgmpy                                 | 0.1.25+             | KHBN structure fitting and inference                                   | Shresth           |
| **Graph Processing**          | NetworkX                              | 3.x                 | Causal DAG construction, Personalised PageRank                         | Shresth / Yashash |
| **Time-Series Database**      | TimescaleDB                           | 2.x (PostgreSQL 15) | Metric storage with hypertables and retention policies                 | Utsav             |
| **Data Versioning**           | DVC                                   | 3.x                 | Dataset versioning and reproducibility                                 | Utsav             |
| **Experiment Tracking**       | MLflow                                | 2.x                 | Model versioning, metric logging, artifact storage                     | Aditya            |
| **Metric Collection**         | Prometheus Python Client              | 0.19+               | Exposing system metrics and scraping targets                           | Utsav             |
| **Cloud Metrics**             | boto3 (AWS SDK)                       | 1.34+               | CloudWatch GetMetricData API integration                               | Utsav             |
| **API Backend**               | FastAPI + Uvicorn                     | 0.110+ / 0.29+      | REST API serving RCA results                                           | Yashash           |
| **NLG Engine**                | Jinja2 + spaCy                        | 3.x / 3.7+          | Template-based narrative generation                                    | Yashash           |
| **Frontend Framework**        | React                                 | 18.x                | Interactive dashboard SPA                                              | Yashash           |
| **Graph Visualization**       | D3.js                                 | 7.x                 | Force-directed causal DAG rendering                                    | Yashash           |
| **Chart Library**             | Recharts                              | 2.x                 | Anomaly timeline and metric charts                                     | Yashash           |
| **Containerization**          | Docker + Docker Compose               | 26.x / 2.x          | Reproducible deployment, one-command startup                           | Utsav             |
| **Testing**                   | pytest + hypothesis                   | 8.x / 6.x           | Unit, integration, and property-based tests                            | All               |
| **Remediation Engine**        | Safety Rule Engine (YAML)             | -                   | Deterministic Tier 1/2/3 safety classification                         | Yashash           |
| **Remediation Execution**     | kubernetes Python client / subprocess | 29.0+ / stdlib      | kubectl command execution for Tier 1 auto-fix actions                  | Yashash           |
| **Runbook Templating**        | Jinja2                                | 3.x                 | Walkthrough and prevention checklist generation                        | Yashash           |
| **CI/CD**                     | GitHub Actions                        | -                   | Automated test runs on PR merge                                        | All               |

# **10\. Validation Case Studies**

The system shall be validated against five real-world-inspired failure scenarios. Each case study serves both as a functional specification for the expected system output and as an evaluation benchmark with known ground-truth root causes. These case studies shall be reproduced using the synthetic failure generator and documented in the final project report.

## **Case Study 1 - E-Commerce Platform: Database Migration Induced Failure**

**Business Impact:** \$50,000 revenue loss over a 45-minute outage affecting customer checkout.

| **Field**                | **Details**                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| System                   | Large e-commerce platform - 10,000 requests/minute peak                                                |
| Architecture             | Frontend API, Product Service, Order Service, PostgreSQL, Redis Cache                                  |
| Root Cause               | DB schema migration (composite index on 50M-row table) at 14:00 → query planner used inefficient index |
| Failure Onset            | 21:00 (7-hour delayed effect from root cause event)                                                    |
| MTTD (Traditional)       | 45 minutes (manual SRE investigation)                                                                  |
| MTTD (RCA System Target) | < 5 minutes (automated detection at 21:05)                                                             |

### **Expected System Output**

| **Metric**            | **Expected Value** | **Actual at Failure** | **Anomaly Score** |
| --------------------- | ------------------ | --------------------- | ----------------- |
| api_latency_p95       | 180ms              | 5,000ms               | 0.96              |
| db_connections_active | 65/100             | 100/100               | 0.89              |
| product_query_latency | 60ms               | 500ms                 | 0.91              |
| error_rate            | 0.01%              | 15.0%                 | 0.98              |

Expected Ranked RCA Output: (1) db_migration_applied @ 14:00 - 87% confidence; (2) connection_pool_size_insufficient - 23%; (3) traffic_volume_increase - 12%.

### **Expected Remediation Engine Output - Case Study 1**

| **Tier**               | **Action**                                                             | **Command / Instruction**                                                           | **Est. Time** |
| ---------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------- |
| Tier 1 - AUTO          | Flush query plan cache                                                 | SELECT pg_stat_reset(); -- resets query planner statistics                          | < 5 sec       |
| Tier 1 - AUTO          | Restart connection pool manager                                        | sudo systemctl restart pgbouncer                                                    | < 10 sec      |
| Tier 2 - GUIDED Step 1 | Verify current migration version                                       | alembic current → expected: head (v2.14.3)                                          | 1 min         |
| Tier 2 - GUIDED Step 2 | Roll back DB migration                                                 | alembic downgrade -1 (restores original index)                                      | 8 min         |
| Tier 2 - GUIDED Step 3 | Force PostgreSQL to use original index                                 | SET enable_seqscan=off; EXPLAIN ANALYZE SELECT ...                                  | 2 min         |
| Tier 2 - GUIDED Step 4 | Verify query latency returned to baseline                              | SELECT mean_exec_time FROM pg_stat_statements WHERE query LIKE '%product_variants%' | 1 min         |
| Tier 3 - ADVISORY      | Require EXPLAIN ANALYZE benchmarks before all future schema migrations | Process change - assign to DBA team                                                 | Long-term     |

## **Case Study 2 - SaaS Platform: Memory Leak Causing Cascading Failure**

**Challenge:** Slow-onset failure over 48 hours; symptoms appeared long after root cause was introduced.

| **Field**                  | **Details**                                                                                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| System                     | Multi-tenant SaaS analytics platform - 1M events/hour                                                                                     |
| Architecture               | Node.js API (20 instances), Kafka, ElasticSearch, Redis                                                                                   |
| Root Cause                 | Code deployment v2.14.3 introduced WebSocket connection leak (2MB/leaked connection)                                                      |
| Failure Pattern            | Monotonic memory ramp: 1.5 GB → 3.8 GB over 36 hours                                                                                      |
| Anomaly Score at Detection | 0.94 (trend anomaly - distinct from spike anomaly profile)                                                                                |
| Expected Causal Chain      | Code deploy → Leaked WebSocket connections → Memory growth → GC pressure → CPU spikes → Event processing delays → Queue backlog → Cascade |

Expected NLG Narrative: 'A code deployment of v2.14.3 on Monday at 10:00 AM introduced a memory leak in the WebSocket connection handler. Over 36 hours, approximately 1,800 connections accumulated 3.6 GB of leaked memory, causing GC pressure and cascading into a complete service failure.'

### **Expected Remediation Engine Output - Case Study 2**

| **Tier**               | **Action**                                                                    | **Command / Instruction**                                 | **Est. Time** |
| ---------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------- | ------------- |
| Tier 1 - AUTO          | Restart all API server instances (rolling)                                    | kubectl rollout restart deployment/node-api -n production | ~3 min        |
| Tier 1 - AUTO          | Force garbage collection on running instances                                 | kill -SIGUSR1 \$(pgrep node) -- triggers manual GC        | < 5 sec       |
| Tier 2 - GUIDED Step 1 | Roll back deployment to v2.14.2                                               | kubectl rollout undo deployment/node-api --to-revision=13 | 2 min         |
| Tier 2 - GUIDED Step 2 | Verify rollback completed                                                     | kubectl rollout status deployment/node-api                | 1 min         |
| Tier 2 - GUIDED Step 3 | Monitor memory trend for 30 minutes post-rollback                             | watch -n 30 'kubectl top pods -l app=node-api'            | 30 min        |
| Tier 3 - ADVISORY      | Add connection.close() in WebSocket disconnect handler in v2.14.3 source code | Code fix - assign to owning dev team                      | Long-term     |
| Tier 3 - ADVISORY      | Add memory leak detection test to CI/CD pipeline                              | Process change - assign to DevOps                         | Long-term     |

## **Case Study 3 - Financial Services: Network Partition Causing Split-Brain**

**Business Impact:** \$2.3M in erroneous trades executed before detection due to Redis split-brain.

| **Field**                      | **Details**                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| System                         | High-frequency trading platform - 100,000 transactions/second                            |
| Architecture                   | Multi-region (US-East/West), Redis cluster (6 nodes), Kafka                              |
| Root Cause                     | AWS inter-region BGP route leak → network partition → Redis cluster split-brain          |
| System Vulnerability           | Redis cluster lacked strict quorum; both partitions accepted writes independently        |
| Expected Root Cause Confidence | 95% - network_partition_bgp_route_leak                                                   |
| Key Anomaly                    | cross_region_latency: 45ms → 5,000ms (timeout); redis_cluster_nodes: 6 → 2 clusters of 3 |

### **Expected Remediation Engine Output - Case Study 3**

| **Tier**               | **Action**                                                                                 | **Command / Instruction**                                           | **Est. Time** |
| ---------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------- |
| Tier 1 - AUTO          | Activate circuit breaker on cross-region traffic                                           | curl -X POST <http://circuit-breaker-api/trip?service=cross-region> | < 5 sec       |
| Tier 2 - GUIDED Step 1 | Halt all trading (risk management circuit breaker)                                         | ./trading-halt.sh --reason=split-brain --severity=critical          | 2 min         |
| Tier 2 - GUIDED Step 2 | Force Redis cluster quorum to US-East (primary region)                                     | redis-cli --cluster fix &lt;us-east-redis-host&gt;:6379             | 5 min         |
| Tier 2 - GUIDED Step 3 | Reconcile transaction state between regions                                                | python reconcile_transactions.py --start=&lt;partition_time&gt;     | 30-60 min     |
| Tier 3 - ADVISORY      | Implement Redis Sentinel with strict majority quorum (minimum 3 nodes required for writes) | Architecture change - assign to Platform team                       | Long-term     |

## **Case Study 4 - Industrial IoT: Thread Pool Exhaustion in Manufacturing System**

**Detection Speed:** RCA system identifies root cause in 3 minutes vs. 15-30 minutes for traditional monitoring.

| **Field**                      | **Details**                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| System                         | Industrial IoT - monitoring 10,000 factory sensors                                                           |
| Architecture                   | Java Spring Boot API, MQTT broker, InfluxDB                                                                  |
| Root Cause                     | Daily sensor data ingestion job executing synchronous InfluxDB queries on shared API thread pool             |
| Failure Timeline               | 15:42:00 job starts → 15:43:30 all 100 API threads blocked → complete API failure                            |
| Expected Granger Relationships | sensor_ingestion_job → influxdb_query_latency (p<0.001) → thread_pool_util (p<0.001) → api_timeout (p<0.001) |
| Expected Confidence            | 93% - thread_pool_background_job_contention                                                                  |

### **Expected Remediation Engine Output - Case Study 4**

| **Tier**               | **Action**                                                       | **Command / Instruction**                                                             | **Est. Time** |
| ---------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------- |
| Tier 1 - AUTO          | Kill ingestion job process immediately                           | sudo kill -9 \$(pgrep -f sensor_ingestion_job)                                        | < 5 sec       |
| Tier 1 - AUTO          | Restart Spring Boot API to release blocked threads               | sudo systemctl restart iot-api                                                        | ~45 sec       |
| Tier 2 - GUIDED Step 1 | Verify all threads released                                      | curl <http://localhost:8080/actuator/metrics/executor.active> \| jq .measurements     | 1 min         |
| Tier 2 - GUIDED Step 2 | Re-schedule ingestion job to dedicated thread pool               | Edit application.yaml: ingestion.thread-pool.size=10, isolation=true; restart service | 5 min         |
| Tier 3 - ADVISORY      | Refactor ingestion job to use CompletableFuture async processing | Code change - assign to backend dev team                                              | Long-term     |

## **Case Study 5 - Cloud Storage: DNS Propagation Delay Causing Regional Outage**

**Challenge:** Intermittent failure pattern - different users experiencing different behaviour based on DNS cache state.

| **Field**           | **Details**                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| System              | Cloud object storage service serving 1 PB of data                                                              |
| Architecture        | Multi-region with DNS-based load balancing                                                                     |
| Root Cause          | DNS record update with TTL=300s during failover; stale DNS entries directed US-West traffic to failed endpoint |
| Failure Pattern     | Partial failure: US-West 47.3% error rate; US-East and EU-Central normal                                       |
| Key Challenge       | Intermittent pattern confuses threshold-based monitors; LSTM detects partial failure profile                   |
| Expected Root Cause | dns_ttl_misconfiguration with region-partitioned error pattern - 89% confidence                                |

### **Expected Remediation Engine Output - Case Study 5**

| **Tier**               | **Action**                                                                                               | **Command / Instruction**                                                                              | **Est. Time** |
| ---------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------- |
| Tier 1 - AUTO          | Set emergency low DNS TTL to force rapid cache expiry                                                    | aws route53 change-resource-record-sets --hosted-zone-id &lt;ID&gt; --change-batch file://ttl-60s.json | < 10 sec      |
| Tier 1 - AUTO          | Flush DNS cache on all US-West API nodes                                                                 | ansible us-west-api -m command -a 'systemd-resolve --flush-caches'                                     | ~30 sec       |
| Tier 2 - GUIDED Step 1 | Verify DNS resolution for affected endpoint in US-West                                                   | dig @8.8.8.8 api.storage.example.com -- confirm new IP returned                                        | 1 min         |
| Tier 2 - GUIDED Step 2 | Update load balancer health check to bypass stale DNS                                                    | aws elbv2 modify-target-group --target-group-arn &lt;ARN&gt; --health-check-path /health               | 2 min         |
| Tier 2 - GUIDED Step 3 | Monitor US-West error rate until below 1%                                                                | watch -n 10 'curl -s <https://metrics/api/error_rate?region=us-west>'                                  | 5-10 min      |
| Tier 3 - ADVISORY      | Implement DNS health check monitoring with automated failover (Route53 health checks + failover routing) | Architecture change - assign to Platform/SRE team                                                      | Long-term     |
| Tier 3 - ADVISORY      | Add DNS propagation verification step to all failover runbooks                                           | Process change - assign to SRE team                                                                    | Long-term     |

# **11\. API Specification**

## **11.1 POST /analyze - Trigger RCA Analysis**

Request: POST /analyze Content-Type: application/json

{

"start_time": "2026-02-24T18:00:00Z",

"end_time": "2026-02-24T21:30:00Z",

"services": \["product-api", "order-api", "postgres"\],

"priority": "high"

}

Response 202 Accepted:

{ "incident_id": "INC-20260224-001",

"status": "processing",

"estimated_completion_seconds": 180,

"poll_url": "/report/INC-20260224-001" }

## **11.2 GET /report/{incident_id} - Retrieve Full RCA Report**

Response 200 OK:

{

"incident_id": "INC-20260224-001",

"status": "complete",

"detected_at": "2026-02-24T21:05:00Z",

"anomalous_metrics": \[

{"name":"db_query_latency_p95","score":0.91,"first_seen":"2026-02-24T18:30:00Z"},

{"name":"api_latency_p95","score":0.96,"first_seen":"2026-02-24T21:00:00Z"}

\],

"causal_chain": \[

{"from":"db_migration_event","to":"db_query_latency","p_value":0.002,"lag_minutes":30},

{"from":"db_query_latency","to":"db_connections","p_value":0.001,"lag_minutes":135}

\],

"ranked_causes": \[

{"rank":1,"cause":"db_migration_applied_14:00","confidence":0.87,"type":"deployment_event"},

{"rank":2,"cause":"connection_pool_insufficient","confidence":0.23,"type":"configuration"}

\],

"narrative": "A database schema migration deployed at 14:00 caused PostgreSQL to use an

inefficient index, slowing product queries from 50ms to 500ms over 4 hours..."

}

## **11.3 POST /remediate/{incident_id} - Get Remediation Plan**

Response 202 Accepted:

{

"incident_id": "INC-20260224-001",

"root_cause": "db_migration_applied_14:00",

"confidence": 0.87,

"remediation_plan": {

"tier1_auto_actions": \[

{ "action": "flush_query_cache",

"command": "redis-cli -h db-cache-01 FLUSHDB",

"auto_execute_in_seconds": 30,

"cancel_url": "/remediate/INC-001/cancel/flush_query_cache" }

\],

"tier2_walkthrough": {

"total_steps": 7,

"steps": \[

{ "step": 1, "title": "Verify migration version",

"command": "alembic current",

"expected_output": "head (schema v2.14.3)",

"verification": "alembic history | head -3",

"est_minutes": 1 },

{ "step": 2, "title": "Roll back DB migration",

"command": "alembic downgrade -1",

"safety_note": "Down-migration is reversible - original index restored",

"est_minutes": 8 }

\]

},

"tier3_advisory": \[

{ "recommendation": "Require EXPLAIN ANALYZE benchmarks before all schema changes",

"horizon": "short_term", "owner": "DBA", "priority": 0.91 }

\]

}

}

## **11.4 TimescaleDB Schema**

CREATE TABLE metrics (

time TIMESTAMPTZ NOT NULL,

host TEXT NOT NULL,

service TEXT NOT NULL,

metric_name TEXT NOT NULL,

value DOUBLE PRECISION,

unit TEXT

);

SELECT create_hypertable('metrics', 'time');

# **12\. Project Phases & Timeline**

The project is organised into 6 phases across 12 weeks, targeting the Distinction Level. GPU training (Phase 2) shall use Google Colab free tier. Each phase has a designated lead and cross-member review checkpoint.

| **Phase**   | **Name**                                 | **Duration** | **Lead** | **Key Deliverables**                                                                                                                                                                                                                                                                                        | **Review Milestone**                                                                                                                                 |
| ----------- | ---------------------------------------- | ------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Foundation & Data Layer                  | Weeks 1-2    | Utsav    | GitHub repo setup, synthetic failure generator (≥200 scenarios), Prometheus + CloudWatch collectors, TimescaleDB schema, EDA notebooks, docker-compose.yml baseline                                                                                                                                         | Week 2: Data review - all team members verify generator output covers all 10 failure types                                                           |
| **Phase 2** | Anomaly Detection Engine                 | Weeks 3-5    | Aditya   | LSTM Autoencoder trained on 30-day synthetic healthy data, Temporal Transformer, anomaly scoring pipeline, MLflow experiment tracking, evaluation vs. Isolation Forest baseline                                                                                                                             | Week 5: Model review - Aditya presents F1 scores, Shresth reviews output schema for causal engine compatibility                                      |
| **Phase 3** | Causal Inference Engine                  | Weeks 5-7    | Shresth  | Granger causality module, PC algorithm DAG, KHBN configuration, Personalised PageRank ranker, causal chain JSON serialization, evaluation on case study ground truth                                                                                                                                        | Week 7: Causal accuracy review - all members validate Top-1 accuracy on 5 case studies                                                               |
| **Phase 4** | NLG, API, Dashboard & Remediation Engine | Weeks 7-9    | Yashash  | FastAPI REST backend, Jinja2 NLG engine, React dashboard, D3.js causal graph viewer, anomaly timeline, RCA panel, incident replay, Remediation Engine (safety classifier, Tier 1 executor, Tier 2 walkthrough generator, Tier 3 advisory, prevention checklist, audit log), all 6 remediation API endpoints | Week 9: Full integration demo - all modules connected end-to-end on Case Study 1 with live Tier 1 auto-fix + Tier 2 walkthrough visible in dashboard |
| **Phase 5** | Integration, Evaluation & Ablation       | Weeks 9-11   | All      | End-to-end pipeline test on all 5 case studies, ablation study (KHBN vs no-KHBN, PageRank vs heuristic), performance benchmarking, accuracy table, Docker Compose final build                                                                                                                               | Week 11: Performance gate - all NFRs validated; results table complete                                                                               |
| **Phase 6** | Documentation, Demo & Report             | Week 12      | All      | Final report (30-60 pages), architecture diagrams, model card, dataset card, demo video (3-5 min MP4), presentation slides (15-20), README, deliverables checklist sign-off                                                                                                                                 | Week 12: Final review - supervisor demo + peer evaluation                                                                                            |

# **13\. Experiment Tracking & Ablation Study Plan**

All training experiments shall be logged in MLflow. The following experiment table template defines the ablation study, comparing the full Distinction-Level system against simpler baselines to quantify the contribution of each component.

| **Run ID** | **Hypothesis**       | **Model**            | **Key Hyperparams**           | **Anomaly F1** | **Top-1 Acc** | **Top-3 Acc** | **Detection Latency** | **Notes**             |
| ---------- | -------------------- | -------------------- | ----------------------------- | -------------- | ------------- | ------------- | --------------------- | --------------------- |
| EXP-001    | LSTM baseline        | LSTM AE              | layers=2, hidden=128, lr=1e-3 | 0.78           | 0.61          | 0.82          | 4.2 min               | Threshold=99th pct    |
| EXP-002    | Larger LSTM          | LSTM AE              | layers=3, hidden=256, lr=1e-3 | 0.81           | 0.64          | 0.85          | 4.5 min               | Overfits <30d data    |
| EXP-003    | IF baseline          | IForest              | n_estimators=100              | 0.70           | 0.48          | 0.71          | 2.1 min               | Comparison baseline   |
| EXP-004    | Add Transformer      | LSTM+Transformer     | α=0.65 ensemble               | 0.84           | 0.67          | 0.87          | 4.8 min               | Best detect result    |
| EXP-005    | No KHBN              | LSTM+PR (no KHBN)    | PageRank d=0.85               | 0.84           | 0.63          | 0.85          | 4.7 min               | Ablation: KHBN value  |
| EXP-006    | Full Distinction     | LSTM+Trans+KHBN+PR   | All components                | 0.84           | 0.72          | 0.91          | 4.9 min               | Target result         |
| EXP-007    | No event correlation | EXP-006 minus events | -                             | 0.84           | 0.55          | 0.80          | 4.6 min               | CI/CD events matter   |
| EXP-008    | Heuristic ranker     | LSTM+KHBN (no PR)    | Causal outflow score          | 0.84           | 0.65          | 0.87          | 4.6 min               | PageRank vs heuristic |

The ablation study shall demonstrate the incremental accuracy gain of: (1) LSTM over Isolation Forest, (2) Transformer ensemble over LSTM alone, (3) KHBN over raw Granger DAG, (4) PageRank over heuristic scoring, and (5) CI/CD event correlation over metrics-only analysis.

# **14\. Deliverables Checklist**

| **#**  | **Artefact**                   | **Description**                                                                                         | **Format**                | **Owner**      | **Status** |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------- | -------------- | ---------- |
| **1**  | Source Code                    | All 6 modules: ingestion, preprocessing, LSTM, causal engine, ranker, NLG, API, dashboard               | GitHub repo (Python, JS)  | All            | -          |
| **2**  | Trained Model Weights          | LSTM AE weights + normalization params + Transformer checkpoint                                         | PyTorch .pt + config.yaml | Aditya         | -          |
| **3**  | Synthetic Dataset              | ≥200 labelled failure scenarios with ground-truth causal chains (10 failure types × ≥20 each)           | HDF5 + JSONL labels       | Utsav          | -          |
| **4**  | Dataset Card                   | Data card: sources, schema, splits, known biases, licence, SHA-256 hash                                 | Markdown / PDF            | Utsav          | -          |
| **5**  | README                         | Installation, usage, quickstart, architecture overview, citation, environment setup                     | README.md in repo root    | Yashash        | -          |
| **6**  | MLflow Experiment Table        | All training runs with hyperparameters, metrics, model artifacts; ablation study results                | MLflow UI + exported CSV  | Aditya         | -          |
| **7**  | Reproducibility Scripts        | train.py, evaluate.py, generate_failures.py with full CLI flags and seed arguments                      | Python scripts            | Aditya / Utsav | -          |
| **8**  | Final Report                   | 30-60 page technical report: architecture, implementation, evaluation, case studies, discussion         | PDF (LaTeX or Word)       | All            | -          |
| **9**  | Presentation Slides            | 15-20 slides: problem, architecture, results, demo screenshots                                          | PowerPoint / PDF          | All            | -          |
| **10** | Demo Video                     | 3-5 minute end-to-end walkthrough of Case Study 1 on the working system                                 | MP4                       | Yashash        | -          |
| **11** | Model Card                     | Intended use, limitations, evaluation results, ethical considerations                                   | Markdown                  | Aditya         | -          |
| **12** | Docker Compose File            | One-command deployment of full stack: API, dashboard, TimescaleDB, Prometheus, MLflow                   | docker-compose.yml        | Utsav          | -          |
| **13** | Remediation Safety Rule Config | YAML file defining all Tier 1/2/3 safety classifications for every supported root cause and action type | safety_rules.yaml         | Yashash        | -          |
| **14** | Runbook Template Library       | Jinja2 templates for all supported remediation walkthroughs and prevention checklists                   | /templates/runbooks/      | Yashash        | -          |
| **15** | Remediation Audit Log Schema   | TimescaleDB schema + sample audit log entries for all 5 case studies                                    | SQL + JSONL               | Yashash        | -          |

# **15\. Demo Script (3-5 Minutes)**

The following demo script shall be used for the final evaluation presentation and recorded demo video. It demonstrates the full end-to-end system on Case Study 1 (E-Commerce DB Migration Failure).

| **Timestamp** | **Action**               | **What to Show**                                                                  | **Expected System Response**                                                                                                                                                 |
| ------------- | ------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 00:00-00:30   | Introduction             | Open dashboard - system monitoring live demo environment                          | Anomaly timeline showing baseline green (normal) scores across all services                                                                                                  |
| 00:30-01:15   | Failure injection        | Trigger generate_failures.py for db_migration scenario                            | Anomaly scores begin rising on db_query_latency (T+18:30 simulated); timeline turns amber                                                                                    |
| 01:15-02:00   | Anomaly detection        | Navigate to Anomaly Timeline tab                                                  | db_query_latency score 0.91, api_latency 0.96, error_rate 0.98 highlighted red; onset timestamp marked                                                                       |
| 02:00-02:45   | Causal graph             | Navigate to Causal Graph tab                                                      | D3.js DAG shows: db_migration_event → db_query_latency → db_connections → api_latency → error_rate; root cause node highlighted                                              |
| 02:45-03:15   | RCA report               | Navigate to RCA Report panel                                                      | Ranked list: (1) db_migration_applied_14:00 - 87% confidence; (2) connection_pool_insufficient - 23%. NLG narrative displayed.                                               |
| 03:15-03:45   | API demo                 | Run curl in terminal: GET /report/INC-001                                         | Full JSON response displayed: anomalous_metrics, causal_chain, ranked_causes, narrative                                                                                      |
| 03:45-04:15   | Second scenario          | Trigger memory_leak scenario; switch to Case Study 2                              | System detects monotonic trend anomaly; correctly ranks memory_leak_code_deploy as #1 root cause (91%)                                                                       |
| 04:15-04:45   | MLflow                   | Open MLflow UI; show EXP-006 vs EXP-003                                           | LSTM+KHBN+PageRank F1=0.84 vs IsolationForest F1=0.70; ablation table visible                                                                                                |
| 04:45-05:15   | Remediation - Tier 1     | Navigate to Remediation Panel - show Tier 1 auto-actions with 30-second countdown | Two Tier 1 actions queued: flush_query_cache and restart pgbouncer - countdown timer visible; Cancel button present                                                          |
| 05:15-05:45   | Remediation - Tier 2     | Click 'View Full Walkthrough' - show 7-step guided procedure                      | Step-by-step checklist appears; Step 1 shows alembic current command with expected output and verification check                                                             |
| 05:45-06:15   | Remediation - Prevention | Click 'Prevention Checklist' tab                                                  | 3-horizon checklist: Immediate (add query monitoring alert), Short-Term (CI/CD benchmark gate), Long-Term (chaos engineering programme) - each with owner and priority score |
| 06:15-06:30   | Audit Log                | Show GET /audit/INC-001 in terminal                                               | JSON audit log: Tier 1 auto-actions executed, timestamps, before/after state snapshots                                                                                       |
| 06:30-06:45   | Conclusion               | Return to dashboard overview                                                      | 'Diagnosis in 3 minutes. Safe fixes auto-applied in 10 seconds. Complex rollback walkthrough generated instantly. Zero manual log-trawling.'                                 |

# **16\. Risk Register**

| **ID**   | **Risk**                                                                                        | **Likelihood** | **Impact** | **Mitigation Strategy**                                                                                                                                     | **Owner** |
| -------- | ----------------------------------------------------------------------------------------------- | -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| **R-01** | Spurious Granger causal edges producing wrong root causes                                       | High           | High       | Apply FDR correction (Benjamini-Hochberg); combine with PC algorithm; KHBN prior reduces false edges by topology knowledge                                  | Shresth   |
| **R-02** | LSTM training instability (gradient vanishing/exploding)                                        | Medium         | High       | Use gradient clipping (max_norm=1.0), layer normalization, learning rate scheduling (ReduceLROnPlateau)                                                     | Aditya    |
| **R-03** | Synthetic data not capturing real-world failure complexity (synthetic-to-real gap)              | Medium         | High       | Validate on ≥2 public benchmarks (NASA anomaly, AIOps challenge datasets) in addition to synthetic; document gap explicitly                                 | Utsav     |
| **R-04** | KHBN fitting fails with insufficient anomaly events (sparse data problem)                       | Medium         | Medium     | Fall back to PC-Granger DAG only; use structure prior from topology YAML to constrain fitting; minimum 50 events required                                   | Shresth   |
| **R-05** | Causal DAG contains cycles (violating DAG constraint)                                           | Medium         | High       | Break cycles by removing weakest-weighted Granger edge; apply FCI algorithm as alternative if PC fails                                                      | Shresth   |
| **R-06** | Dashboard performance degrades with >30-node causal graphs                                      | Low            | Medium     | Implement node clustering for >20 nodes; lazy render; virtualise D3.js force simulation                                                                     | Yashash   |
| **R-07** | Prometheus/CloudWatch API rate limits or auth failures                                          | Medium         | Low        | MVP falls back to synthetic data; integration is distinction-level; mock collector available for demo                                                       | Utsav     |
| **R-08** | Team coordination delays on cross-module interface changes                                      | Medium         | Medium     | Interface contracts documented in /docs/interfaces/; weekly syncs; interface changes require 48-hour advance notice                                         | All       |
| **R-09** | Tier 1 auto-execution applied to wrong service due to incorrect environment variable resolution | Low            | Critical   | Safety rule engine validates service name and namespace against allow-list before execution; dry-run mode available; confidence gate ≥0.70 required         | Yashash   |
| **R-10** | Rollback command fails mid-way leaving system in partial state                                  | Low            | High       | Each walkthrough step includes a 'rollback of this step' instruction; system monitors step outcome and escalates to Tier 3 advisory on failure              | Yashash   |
| **R-11** | Generated kubectl/SQL commands incompatible with target environment version                     | Medium         | Medium     | Commands generated from version-aware Jinja2 templates; environment version stored in system config YAML; fallback to generic commands with version warning | Yashash   |

# **17\. Evaluation Criteria & Success Metrics**

## **17.1 Quantitative Metrics**

| **Metric**                            | **Pass Level Target** | **Distinction Level Target** | **Measurement Method**                                               |
| ------------------------------------- | --------------------- | ---------------------------- | -------------------------------------------------------------------- |
| Anomaly Detection Precision           | \>75%                 | \>85%                        | Precision on injected failure scenarios (synthetic labelled dataset) |
| Anomaly Detection Recall              | \>65%                 | \>75%                        | Recall on injected failure scenarios                                 |
| Anomaly Detection F1                  | \>70%                 | \>80%                        | Harmonic mean of precision and recall                                |
| Top-1 Root Cause Accuracy             | \>60%                 | \>70%                        | Fraction of incidents where correct root cause is ranked #1          |
| Top-3 Root Cause Accuracy             | \>75%                 | \>88%                        | Fraction of incidents where correct root cause is in top 3           |
| False Positive Rate on Normal Data    | <10%                  | <5%                          | FPR measured on 30 days of healthy synthetic data                    |
| Detection Latency (from onset)        | <10 minutes           | <5 minutes                   | Wall-clock time from anomaly_onset_timestamp to ranked_causes output |
| Causal Edge Accuracy vs. Ground Truth | \>60%                 | \>75%                        | Precision of causal DAG edges vs. known dependency graph             |

## **17.2 Qualitative Evaluation**

- Interpretability: Can a non-expert engineer understand the NLG causal chain narrative without reading code or dashboards?
- Remediation Safety: Is every auto-executed Tier 1 action correctly classified as reversible by an independent reviewer?
- Walkthrough Completeness: Does each Tier 2 walkthrough contain sufficient detail for an engineer unfamiliar with the system to execute it safely?
- Prevention Quality: Does the long-term prevention checklist address the structural root cause, not just the immediate symptom?
- Generalizability: Does the system identify novel failure patterns introduced in held-out test scenarios not seen during training?
- Robustness: Evaluate performance under increasing Gaussian noise (σ = 0.1, 0.3, 0.5) and missing data fractions (10%, 25%, 40%).
- Ablation study completeness: Every architectural decision in Section 5.2 Decision Log must be validated with a corresponding EXP- row.
- Demo convincingness: Can the demo video be understood by an industry SRE with no prior knowledge of the system?

## **17.3 Comparison Baselines**

| **Baseline**                    | **Method**                                      | **Expected F1** | **Expected Top-1 Acc** | **Purpose**                                       |
| ------------------------------- | ----------------------------------------------- | --------------- | ---------------------- | ------------------------------------------------- |
| B-1: Static Threshold           | Alert when metric > mean + 3σ                   | ~0.55           | ~0.30                  | Industry standard - what the system improves upon |
| B-2: Isolation Forest           | Unsupervised anomaly detection, no causal layer | ~0.70           | ~0.48                  | Classical ML baseline                             |
| B-3: LSTM AE only (no causal)   | Full anomaly detection, heuristic ranking       | ~0.78           | ~0.61                  | Ablation: value of causal inference layer         |
| B-4: LSTM + Granger (no KHBN)   | Causal DAG without topology prior               | ~0.82           | ~0.65                  | Ablation: value of KHBN                           |
| TARGET: Full Distinction System | LSTM + Transformer + KHBN + PageRank            | ~0.84           | \>0.70                 | Primary system                                    |

# **18\. Academic & Industry Positioning**

## **18.1 Research Foundations**

| **Technique**                                           | **Source / Validation**                                                             | **Implementation in This Project** |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
| LSTM Autoencoder for time-series anomaly detection      | Validated in production: Microsoft FluxInfer, Google Vertex Anomaly Detection       | Module 3 primary model             |
| Granger Causality for temporal causal inference         | Standard econometric technique; validated for system metrics in academic literature | Module 4 Stage 1                   |
| PC Algorithm for causal structure learning              | Spirtes et al. (2000); open implementation in causal-learn library                  | Module 4 Stage 2                   |
| Knowledge-Informed Hierarchical Bayesian Network (KHBN) | CloudRCA (Alibaba Cloud) - reported >20% reduction in SRE troubleshooting time      | Module 4 KHBN layer                |
| Personalised PageRank for root cause ranking            | RUN framework - academic validation on AIOps datasets                               | Module 5 ranker                    |
| Hybrid KG+CBN reduces spurious correlations             | EV manufacturing deployment - 60.6% reduction in false causal edges                 | KHBN topology prior                |

## **18.2 Why This Exceeds Typical Final-Year Projects**

- Unsupervised learning - no labelled data required for anomaly detection, unlike 90% of student ML projects.
- Causal reasoning - goes beyond classification/regression into the frontier of causal AI (Judea Pearl's framework).
- End-to-end production architecture - data simulation, ML training, causal inference, API, and interactive UI.
- Realistic constraints - partial observability, noisy data, rare failures, delayed effects (not toy datasets).
- Industry-validated techniques - KHBN from Alibaba CloudRCA, PageRank from academic RUN framework.
- Comprehensive evaluation - 5 case studies, 8 ablation experiments, 4 baselines, quantitative + qualitative metrics.
- Reproducibility - MLflow + DVC + Docker Compose + fixed seeds; fully reproducible by any evaluator.

## **18.3 Industry Relevance**

The problem this system addresses is actively solved by dedicated engineering teams at:

| **Company**   | **Internal System**                                     | **Overlap with This Project**                                |
| ------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| Alibaba Cloud | CloudRCA                                                | KHBN architecture directly adopted from CloudRCA paper       |
| Google        | Vertex AI Anomaly Detection, SRE postmortem automation  | LSTM-based temporal anomaly detection; NLG report generation |
| Microsoft     | FluxInfer, Azure Chaos Engineering                      | LSTM Autoencoder architecture; multi-signal correlation      |
| Netflix       | ATLAS anomaly detection, distributed tracing RCA        | Causal graph construction; microservice dependency analysis  |
| AWS           | CloudWatch Anomaly Detection, Systems Manager OpsCenter | Prometheus/CloudWatch integration; structured causal output  |

# **19\. Testing Strategy**

**Important Distinction:** The Synthetic Failure Generator and the Google Microservices Demo are internal development tools used to create test log files and metric traces. They are not the end product. The end product is the AutoRCA agent that reads any log file from any system and finds the root cause. The chaos tools simply give us labelled test data to prove the agent works correctly.

## **19.1 How We Test the Log Analysis Agent**

Because real production failures are rare and unlabelled, we need a controlled way to produce log files with known root causes so we can verify the agent identifies them correctly. The testing strategy has three layers:

| **Layer**                                      | **Method**                                                                                                                                                                                      | **What It Produces**                                                                  | **How It Is Used**                                                                                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Layer 1 - Synthetic Log Generator              | Python script (generate_failures.py) writes realistic log file content mimicking application, database, system, and container logs for each of the 10 failure scenarios                         | Labelled .log files + JSONL ground-truth causal chains (≥200 scenarios)               | Feed log files into the agent via config.yaml; verify agent's ranked root causes match the JSONL ground truth labels                       |
| Layer 2 - Google Microservices Demo (Internal) | Run the open-source Google Online Boutique microservices demo locally or on a university VM; inject failures via Chaos Mesh or manual fault injection; collect the real container logs produced | Real container log files from a running distributed system with known injected faults | Point the agent's config.yaml at the container log directory; verify agent detects the injected fault - this bridges synthetic-to-real gap |
| Layer 3 - Public Benchmark Datasets            | NASA SMAP/MSL anomaly dataset (metrics), Loghub public log datasets (HDFS, BGL, OpenStack logs from academic research)                                                                          | External labelled log and metric datasets with known anomaly timestamps               | Evaluate agent's log model (LogBERT/TF-IDF) precision/recall against published benchmark results                                           |

## **19.2 Synthetic Log File Formats**

The generate_failures.py script shall produce realistic log files in standard formats for each failure scenario. Example outputs for Case Study 1 (DB Migration):

\# db.log - generated by synthetic failure generator

2026-02-24 14:00:01 INFO Applying migration: add_composite_index_product_variants

2026-02-24 14:02:15 INFO Migration complete. Rows affected: 50241872

2026-02-24 18:31:02 WARN Slow query detected: SELECT \* FROM product_variants (487ms)

2026-02-24 19:30:44 WARN Slow query detected: SELECT \* FROM product_variants (1203ms)

2026-02-24 20:45:11 ERROR Connection pool at capacity: 100/100 connections active

\# app.log

2026-02-24 21:00:03 ERROR Request timeout: waiting for DB connection (5000ms)

2026-02-24 21:00:04 ERROR Request timeout: waiting for DB connection (5000ms)

2026-02-24 21:00:05 CRITICAL Service health check failed: error_rate=15.3%

The agent, when pointed at db.log and app.log via config.yaml, is expected to: (1) identify the INFO 'Applying migration' log line at 14:00 as the anomaly-correlated event, (2) detect WARN slow query lines as the first anomaly onset at 18:31, (3) trace the causal chain to the CRITICAL error at 21:00, and (4) rank the DB migration as the #1 root cause with >80% confidence.

## **19.3 Evaluation Protocol**

| **Test Type**                      | **Input**                                                              | **Expected Output**                                                  | **Pass Criterion**                                  |
| ---------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------- |
| Unit: Log Parser                   | Single raw log line in each of 5 formats                               | Structured (timestamp, level, message) record                        | 100% parse rate on test fixtures                    |
| Unit: Drain3 Template Extraction   | 100 log lines with 10 unique templates in varied phrasings             | 10 stable template IDs, correct wildcard placement                   | \>95% template recall vs. manual labels             |
| Integration: Log Anomaly Detection | 30 days healthy logs + 1 injected failure scenario                     | Anomaly score >0.5 on failure window; <0.3 on healthy windows        | Precision >85%, Recall >75%                         |
| Integration: Cross-File Causality  | db.log (root cause) + app.log (symptom)                                | Causal edge from db event to app error in DAG                        | Edge present with p<0.05 in ≥80% of test cases      |
| System: End-to-End RCA             | config.yaml pointing at generated log files for each of 5 case studies | Correct root cause ranked #1 with >70% confidence                    | Top-1 accuracy >70%, Top-3 >88%                     |
| System: Remediation                | RCA report for Case Study 1                                            | Tier 1 cache flush + Tier 2 migration rollback walkthrough (7 steps) | All steps present, commands correct for environment |

# **20\. Glossary**

| **Term**                | **Full Name / Definition**                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| RCA                     | Root Cause Analysis - identifying the primary upstream cause of a system failure                                                                    |
| LSTM                    | Long Short-Term Memory - a recurrent neural network architecture for sequential/temporal data modelling                                             |
| Autoencoder             | Neural network trained to reconstruct its input; elevated reconstruction error indicates anomaly                                                    |
| Transformer             | Attention-based neural architecture for sequence modelling; used here for multi-step metric forecasting                                             |
| Granger Causality       | Statistical test: X Granger-causes Y if past values of X improve prediction of Y beyond Y's own past                                                |
| PC Algorithm            | Peter-Clark algorithm for causal structure learning; infers DAG from conditional independence tests                                                 |
| KHBN                    | Knowledge-Informed Hierarchical Bayesian Network - Bayesian Network enriched with service topology prior                                            |
| Personalised PageRank   | PageRank variant with a bias vector; here biased toward anomalous nodes to rank root causes                                                         |
| DAG                     | Directed Acyclic Graph - directed graph with no cycles; used to model causal relationships                                                          |
| NLG                     | Natural Language Generation - automatic generation of human-readable text from structured data                                                      |
| MLflow                  | Open-source platform for ML experiment tracking, model versioning, and artifact management                                                          |
| DVC                     | Data Version Control - Git-like versioning for datasets and ML models                                                                               |
| MTTR                    | Mean Time to Resolution - average time from incident detection to full service restoration                                                          |
| MTTD                    | Mean Time to Detection - average time from failure onset to first alert or detection                                                                |
| MTTRC                   | Mean Time to Root Cause - average time from failure onset to correct root cause identification                                                      |
| AIOps                   | Artificial Intelligence for IT Operations - applying ML to observability and incident management                                                    |
| SRE                     | Site Reliability Engineering - discipline for managing large-scale systems reliability at companies like Google                                     |
| FDR                     | False Discovery Rate - fraction of false positives among all discoveries; controlled via Benjamini-Hochberg                                         |
| TimescaleDB             | PostgreSQL extension for time-series data with hypertables and automatic partitioning                                                               |
| Log Template            | A generalised pattern extracted from raw log messages by replacing variable tokens with wildcards (e.g., 'User \* failed login from \*')            |
| Drain3                  | An online log parsing algorithm that clusters log messages into templates without requiring predefined formats or retraining on new patterns        |
| LogBERT                 | A BERT-based language model fine-tuned on log event sequences for semantic anomaly detection via masked event prediction                            |
| Event Count Matrix      | A (window_size × n_templates) matrix representing how many times each log template appeared per timestep - the input to the log anomaly model       |
| Rarity Prior            | A score assigned to each log event based on log level weight and inverse document frequency; used to up-weight rare events in root cause ranking    |
| File Tailing            | Continuously reading new lines appended to a log file, analogous to the Unix 'tail -f' command; implemented via OS inotify events                   |
| Causal Chain            | Ordered sequence of cause-effect relationships from root cause to observed failure symptom                                                          |
| Safety Tier             | Classification of a remediation action as Tier 1 (auto-execute), Tier 2 (guided walkthrough), or Tier 3 (advisory only)                             |
| Remediation Walkthrough | Step-by-step guided procedure with exact commands, expected outputs, verification checks, and rollback instructions for Tier 2 actions              |
| Prevention Checklist    | Tiered set of architectural, process, and monitoring improvements to prevent recurrence, organised into Immediate / Short-Term / Long-Term horizons |
| Audit Log               | Immutable append-only record of every auto-executed remediation action including before/after state snapshots                                       |
| Confidence Gate         | Minimum root cause confidence score (default: 0.70) required before the Remediation Engine may propose or execute any action                        |
