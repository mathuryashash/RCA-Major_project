# AI-Powered Root Cause Analysis (RCA) System
## End-to-End Implementation & Deployment Journey

This document serves as a comprehensive overview of the AI-Powered Root Cause Analysis (RCA) system built for your Major Project. It explains the journey from initial planning to cloud deployment, detailing every module, technology choice, and integration step. You can use this to easily present your project to evaluators.

---

### Phase 1: Planning and Architecture Design
**Goal:** Design a system capable of diagnosing the root cause of production incidents (like a database crash or CPU spike) by analyzing system metrics.

**What we did:**
1. **Defined the Core Framework:** We established a multi-step pipeline consisting of Data Preprocessing, Anomaly Detection, Causal Inference, and Root Cause Ranking.
2. **Key Technologies:** Selected `Python` for data processing, `PyTorch` for deep learning (LSTMs), `NetworkX` for graph algorithms, `statsmodels` for statistical causality, and `Streamlit` for the interactive dashboard.
3. **Drafted the PRD:** Maintained an agile Product Requirements Document (`PRD.md`) outlining six primary phases of development.

---

### Phase 2: Data Generation and Preprocessing
**Goal:** The system needs realistic telemetry data to train on. We needed to simulate both "normal" operational behavior and multiple specific "failure" scenarios.

**What we did:**
1. **Synthetic Data Generator:** Built `SyntheticMetricsGenerator` to simulate realistic microservice metrics (CPU, Memory, API Latency, Database connections) using sinusoidal waves for diurnal traffic and Gaussian noise.
2. **Failure Injections:** Implemented injection logic for multiple failure types:
   - `database_slow_query`: Spikes DB latency, then API latency, then causes an error rate spike.
   - `memory_leak`: Slowly ramps up memory usage until an Out-Of-Memory application crash.
   - `cpu_spike`: Creates sudden spikes in utilization.
3. **Preprocessing Pipeline:** Built a robust scaling pipeline utilizing `MinMaxScaler` to normalize metrics between 0-1, as Neural Networks perform best on normalized bounds.

---

### Phase 3: AI Anomaly Detection (The "What went wrong?" phase)
**Goal:** Detect when metrics deviate from their normal boundaries before an alert would typically trigger.

**What we did:**
1. **LSTM Autoencoder:** Implemented a Deep Learning sequence model using `PyTorch`. 
   - *How it works:* The Autoencoder tries to reconstruct the normal telemetry data. If a sequence of metrics enters the model during an incident, the model will fail to reconstruct it accurately. High "Reconstruction Error" flags the metric as anomalous.
2. **Ensemble Detection:** Relying only on Deep Learning can lead to false positives. We fortified our detection by creating an `EnsembleAnomalyDetector`. It combines three methods:
   - **LSTM Deviation** (Deep Learning)
   - **Moving Average Convergence/Divergence (MACD)** (Statistical)
   - **Temporal Pattern Breaks**
   - *Result*: A metric is only marked anomalous if multiple estimators flag it simultaneously, dramatically reducing false alarms.

---

### Phase 4: Causal Inference (The "Why did it go wrong?" phase)
**Goal:** Once anomalies are detected, determine the sequence of events. Which anomaly caused the others?

**What we did:**
1. **Granger Causality:** Implemented a `GrangerAnalyzer` utilizing statistical hypothesis testing. It checks if the past values of Metric A can predict the future values of Metric B better than Metric B's own past values. If yes, A "Granger-causes" B.
2. **Causal Graph Building:** Used `NetworkX` to turn the Granger causality matrix into a Directed Graph (nodes are metrics, edges indicate causation strength and time-lag).
3. **Dynamic Topology (Jaeger Integration):** Statistical causality is sometimes wrong. We built a `DynamicGraphGenerator` that queries a live Distributed Tracing backend (like Jaeger/OpenTelemetry) to prune mathematically impossible connections. E.g., if the App doesn't talk to the Cache, the App cannot cause the Cache to crash.

---

### Phase 5: Root Cause Ranking & Reporting
**Goal:** Process the massive correlation graph down into a single human-readable "Top 3 Suspects" list.

**What we did:**
1. **Multi-Factor Scoring Engine:** Built a `RootCauseRanker` that scores candidates based on:
   - *Temporal Priority:* Who broke first?
   - *Causal Outflow:* Who caused the most downstream chaos?
   - *PageRank Centrality:* Using Google's famous web-page algorithm on the causal graph to find the "center of gravity" of the failure.
2. **Report Generation:** Built a Markdown/JSON generator that formats the technical rankings into an Executive Summary.

---

### Phase 6: Interactive Dashboard & Command Line Pipeline
**Goal:** Make the complex AI engine accessible and easy to test.

**What we did:**
1. **Main Entrypoint (`train_and_run.py`):** Consolidated all the disparate modules into a single, clean `.py` execution script that automatically executes the entire pipeline end-to-end and outputs ASCII reports directly in the terminal.
   - Fixed Windows CP1252 local encoding bugs.
2. **Streamlit UI (`dashboard.py`):** Constructed a full web application featuring:
   - Sidebars for data tuning
   - Plotly interactive charts for Anomaly Time-series
   - Visualized causal graphs using NetworkX coordinates.
   - Interactive Ground-Truth comparison to evaluate model accuracy.

---

### Phase 7: Production Containerization and Deployment
**Goal:** Host the application on the cloud so external users can access it.

**What we did:**
1. **Git Initialization:** Created a local git repository to track all codebase changes.
2. **Dockerization:** Wrote a `Dockerfile` and `.dockerignore`. Packaging the system into a Docker container ensures that "it just works" on any cloud machine, circumventing Python environment mismatch issues.
3. **Cloud Push (Render.com/GitHub):**
   - Pushed the entire containerized architecture to GitHub (`mathuryashash/RCA-Major_project`).
   - Connected the Git pipeline to Render.
   - Render automatically compiled the Docker image and deployed the Streamlit dashboard to public HTTPS at `rca-major-project.onrender.com`.

---

### Final Thoughts for your Presentation
When presenting this project, highlight the transition from **Synthetic Rules** to **AI & Statistics**. Many standard RCA tools rely on hardcoded "If-This-Then-That" logic. Your project stands out because it utilizes structural Causal Inference combined with Deep Learning (LSTMs) to mathematically *derive* the failure cascade, making it agnostic to whatever infrastructure it runs on.
