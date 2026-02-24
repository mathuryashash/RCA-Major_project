

<think>The user is a college student working on a major project: an Automated Root Cause Analysis (RCA) system for production failures. They have two documents:

1. A literature review on RCA architectures, frameworks, and industrial case studies
2. A project documentation/proposal for building an AI-powered RCA system

They want me to analyze both and help them build this system with detailed steps, architecture, and feasibility guidance. Let me provide a comprehensive, practical guide.

Let me think about this carefully - they're a college student, so I need to be realistic about what's achievable while being thorough. The project doc says 16-24 weeks, final year CS/Engineering project.


I'll organize my response to cover the project scope and feasibility, break down the architecture into manageable components, walk through implementation step-by-step, dive into the technical details of each piece, and give them practical advice for staying on track.</think># Building Your Automated RCA System: A Complete Implementation Guide

I've gone through both documents carefully. Your literature review provides strong theoretical grounding, and your project doc outlines an ambitious but achievable system. Let me break down exactly how to build this, what's realistic, and where to focus your energy.

---

## 1. Honest Feasibility Assessment

Your project doc rates feasibility at 85%, which is reasonable but needs nuance. The four-stage pipeline (learn → detect → infer causality → rank) is well-established in research, and the individual components all have mature open-source implementations. The real challenge is integration and evaluation.

Here's where things stand practically for a college-level project on a 16–24 week timeline:

**What's very achievable:** Anomaly detection on time-series metrics using LSTM autoencoders, basic Granger causality between metric pairs, a working end-to-end pipeline on synthetic data, a Streamlit dashboard showing results, and detection of 10–15 anomaly types from your catalog.

**What's achievable with effort:** Causal graph construction using the PC algorithm, integration with Prometheus for real metric ingestion, root cause ranking with confidence scores, and evaluation on public benchmark datasets like those from Azure or NASA.

**What's stretch territory:** Multi-modal fusion of logs and metrics (as MULAN does), transfer learning across platforms (as CloudRCA does), real-time production deployment, and achieving publication-quality novelty.

My recommendation is to target the "Distinction Level" from your project doc but plan your architecture so the stretch features can be added incrementally. Start narrow and deep, then expand.

---

## 2. System Architecture — What to Actually Build

Based on your literature review's insights and your project doc's pipeline, here's the concrete architecture I'd recommend:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  Prometheus/Synthetic → Metrics (CPU, mem, latency...)   │
│  CI/CD Events         → Deployments, config changes      │
│  (Optional) Logs      → Application log streams          │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│              DATA INGESTION & PREPROCESSING              │
│  • Time-series alignment & resampling                    │
│  • Missing value imputation (forward-fill, interpolation)│
│  • Normalization (z-score per metric)                    │
│  • Sliding window creation                               │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│           NORMAL BEHAVIOR LEARNING (Stage 1)             │
│  • LSTM Autoencoder trained on healthy data               │
│  • Learns reconstruction of normal metric patterns        │
│  • Captures inter-metric correlations                     │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│            ANOMALY DETECTION (Stage 2)                   │
│  • Reconstruction error scoring                           │
│  • Dynamic thresholding (POT / adaptive percentile)      │
│  • Anomaly type classification                            │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│            CAUSAL INFERENCE (Stage 3)                    │
│  • Pairwise Granger causality tests                      │
│  • PC algorithm for causal graph structure                │
│  • Event correlation (deployments, config changes)        │
│  • Temporal precedence analysis                           │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│          ROOT CAUSE RANKING & REPORTING (Stage 4)        │
│  • PageRank on causal graph                               │
│  • Composite scoring (causal outflow + temporal priority) │
│  • Natural language report generation                     │
│  • Streamlit dashboard with causal graph visualization    │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture Works for Your Project

Your literature review highlights that the most successful industrial RCA systems are **hybrid** — they combine data-driven learning with structural knowledge. Your project doc's pipeline aligns with this. The architecture above maps directly to CloudRCA's two-stage pipeline (feature engineering → probabilistic inference) but simplifies the probabilistic model to something you can build in the available time. Instead of a full Knowledge-informed Hierarchical Bayesian Network, you use Granger causality + PC algorithm + PageRank, which achieves similar goals with much less implementation complexity.

---

## 3. Step-by-Step Implementation Plan

### Phase 1: Foundation (Weeks 1–3)

**Goal:** Development environment, synthetic data generator, data pipeline.

**3.1 — Project Setup**

```
your-rca-project/
├── data/
│   ├── synthetic/          # Generated failure scenarios
│   ├── benchmarks/         # NASA, Azure public datasets
│   └── raw/                # Real Prometheus data (later)
├── src/
│   ├── data_generation/    # Synthetic data generator
│   ├── preprocessing/      # Normalization, windowing
│   ├── models/             # LSTM autoencoder, etc.
│   ├── anomaly_detection/  # Scoring, thresholding
│   ├── causal_inference/   # Granger, PC algorithm
│   ├── ranking/            # Root cause scoring
│   └── visualization/      # Streamlit dashboard
├── notebooks/              # Exploration & experiments
├── tests/                  # Unit & integration tests
├── configs/                # Hyperparameters, system configs
├── docker/                 # Dockerfile, docker-compose
└── docs/                   # Project documentation
```

**Environment:**

```bash
# Create conda environment
conda create -n rca python=3.10
conda activate rca

# Core dependencies
pip install torch torchvision  # or tensorflow
pip install numpy pandas scikit-learn
pip install statsmodels         # Granger causality
pip install causal-learn        # PC algorithm (CMU's causal-learn)
pip install networkx            # Graph operations
pip install streamlit plotly    # Dashboard
pip install prometheus-api-client  # Prometheus integration
```

I'd recommend `causal-learn` (from CMU) over `CausalNex` or `DoWhy` for the PC algorithm because it's more actively maintained, has a cleaner API for constraint-based causal discovery, and maps directly to what the literature review discusses.

**3.2 — Synthetic Data Generator (This Is Critical)**

This is the single most important component to get right early. Your literature review notes that "performance on controlled synthetic datasets frequently does not translate to real-world systems" — but for a college project, synthetic data with known ground truth is essential for validating your pipeline. You need to build a generator that can simulate realistic metric behavior and inject known failure scenarios.

```python
import numpy as np
import pandas as pd

class SystemMetricsGenerator:
    """
    Generates synthetic time-series metrics for a microservice system
    with realistic correlations and injectable failure scenarios.
    """
    
    def __init__(self, num_services=5, sampling_interval_sec=60, seed=42):
        np.random.seed(seed)
        self.num_services = num_services
        self.interval = sampling_interval_sec
        self.services = [f"service_{i}" for i in range(num_services)]
        
        # Define inter-service dependencies (adjacency matrix)
        # service_0 -> service_1 -> service_2 (linear chain)
        # service_0 -> service_3 (branch)
        # service_3 -> service_4
        self.dependency_matrix = np.zeros((num_services, num_services))
        self.dependency_matrix[0, 1] = 1  # 0 causes 1
        self.dependency_matrix[1, 2] = 1  # 1 causes 2
        self.dependency_matrix[0, 3] = 1  # 0 causes 3
        self.dependency_matrix[3, 4] = 1  # 3 causes 4
    
    def generate_normal_metrics(self, duration_hours=168):
        """Generate one week of normal operational metrics."""
        num_points = int(duration_hours * 3600 / self.interval)
        timestamps = pd.date_range(
            start='2025-01-01', periods=num_points, freq=f'{self.interval}s'
        )
        
        data = {}
        for svc in self.services:
            # Base patterns: daily seasonality + weekly trend + noise
            t = np.arange(num_points)
            daily_cycle = 20 * np.sin(2 * np.pi * t / (1440))  # 24h cycle
            weekly_trend = 5 * np.sin(2 * np.pi * t / (10080))
            
            data[f'{svc}_cpu'] = np.clip(
                40 + daily_cycle + weekly_trend + np.random.normal(0, 3, num_points),
                0, 100
            )
            data[f'{svc}_memory_mb'] = np.clip(
                1500 + daily_cycle * 25 + np.random.normal(0, 50, num_points),
                0, 4096
            )
            data[f'{svc}_latency_ms'] = np.clip(
                100 + daily_cycle * 2 + np.random.normal(0, 15, num_points),
                1, 10000
            )
            data[f'{svc}_error_rate'] = np.clip(
                0.001 + np.random.exponential(0.0005, num_points),
                0, 1
            )
            data[f'{svc}_request_rate'] = np.clip(
                500 + daily_cycle * 15 + np.random.normal(0, 20, num_points),
                0, 5000
            )
        
        return pd.DataFrame(data, index=timestamps)
    
    def inject_failure(self, df, failure_type, root_service_idx=0, 
                       start_time=None, propagation_delay_minutes=30):
        """
        Inject a failure scenario with known root cause and propagation.
        Returns modified dataframe and ground truth metadata.
        """
        df = df.copy()
        svc = self.services[root_service_idx]
        
        if start_time is None:
            # Pick a random point in the second half
            start_idx = len(df) // 2 + np.random.randint(0, len(df) // 4)
        else:
            start_idx = df.index.get_loc(start_time, method='nearest')
        
        ground_truth = {
            'root_cause_service': svc,
            'root_cause_type': failure_type,
            'root_cause_time': df.index[start_idx],
            'affected_services': [],
            'causal_chain': []
        }
        
        if failure_type == 'memory_leak':
            # Gradual memory increase in root service
            leak_duration = min(1440, len(df) - start_idx)  # up to 24 hours
            leak_ramp = np.linspace(0, 2500, leak_duration)
            df.loc[df.index[start_idx:start_idx+leak_duration], 
                   f'{svc}_memory_mb'] += leak_ramp
            
            # After memory hits threshold, CPU spikes (GC pressure)
            gc_start = start_idx + int(leak_duration * 0.7)
            gc_duration = leak_duration - int(leak_duration * 0.7)
            df.loc[df.index[gc_start:gc_start+gc_duration], 
                   f'{svc}_cpu'] += np.random.uniform(20, 40, gc_duration)
            
            # Propagate latency increase to downstream services
            self._propagate_failure(
                df, root_service_idx, start_idx + int(leak_duration * 0.5),
                'latency_ms', magnitude=300, ground_truth=ground_truth,
                propagation_delay_minutes=propagation_delay_minutes
            )
        
        elif failure_type == 'cpu_saturation':
            duration = min(360, len(df) - start_idx)
            df.loc[df.index[start_idx:start_idx+duration], 
                   f'{svc}_cpu'] = np.clip(
                df.loc[df.index[start_idx:start_idx+duration], 
                       f'{svc}_cpu'] + 45,
                0, 100
            )
            self._propagate_failure(
                df, root_service_idx, start_idx,
                'latency_ms', magnitude=500, ground_truth=ground_truth,
                propagation_delay_minutes=propagation_delay_minutes
            )
        
        elif failure_type == 'connection_pool_exhaustion':
            duration = min(180, len(df) - start_idx)
            # Latency creep -> connection saturation -> error spike
            ramp = np.linspace(0, 400, duration)
            df.loc[df.index[start_idx:start_idx+duration], 
                   f'{svc}_latency_ms'] += ramp
            
            error_start = start_idx + int(duration * 0.6)
            error_dur = duration - int(duration * 0.6)
            df.loc[df.index[error_start:error_start+error_dur], 
                   f'{svc}_error_rate'] += np.linspace(0, 0.15, error_dur)
            
            self._propagate_failure(
                df, root_service_idx, start_idx + int(duration * 0.3),
                'error_rate', magnitude=0.1, ground_truth=ground_truth,
                propagation_delay_minutes=propagation_delay_minutes
            )
        
        return df, ground_truth
    
    def _propagate_failure(self, df, source_idx, start_idx, 
                           metric_suffix, magnitude, ground_truth,
                           propagation_delay_minutes):
        """Propagate failure effects through dependency graph with delays."""
        delay_steps = propagation_delay_minutes  # assuming 1-min intervals
        
        # BFS through dependency graph
        visited = {source_idx}
        queue = [(source_idx, 0)]  # (service_idx, depth)
        
        while queue:
            current, depth = queue.pop(0)
            for downstream in range(self.num_services):
                if (self.dependency_matrix[current, downstream] == 1 
                    and downstream not in visited):
                    visited.add(downstream)
                    
                    prop_start = start_idx + delay_steps * (depth + 1)
                    if prop_start < len(df):
                        prop_duration = min(300, len(df) - prop_start)
                        svc_name = self.services[downstream]
                        col = f'{svc_name}_{metric_suffix}'
                        
                        # Attenuated effect at each hop
                        attenuation = 0.7 ** (depth + 1)
                        effect = magnitude * attenuation
                        df.loc[df.index[prop_start:prop_start+prop_duration], 
                               col] += effect
                        
                        ground_truth['affected_services'].append(svc_name)
                        ground_truth['causal_chain'].append({
                            'from': self.services[current],
                            'to': svc_name,
                            'delay_minutes': delay_steps * (depth + 1),
                            'metric': metric_suffix
                        })
                    
                    queue.append((downstream, depth + 1))
```

This generator gives you controlled experiments with known ground truth — you know exactly which service is the root cause, what the failure type is, and what the causal chain looks like. This is how you'll validate every component of your pipeline.

---

### Phase 2: Anomaly Detection (Weeks 4–7)

**Goal:** Train LSTM autoencoder, implement anomaly scoring.

**4.1 — LSTM Autoencoder**

This is the core of your Stage 1 and Stage 2. The model learns to reconstruct normal metric behavior. During inference, high reconstruction error signals anomalies.

```python
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for multivariate time-series
    anomaly detection. Trained on normal data only.
    """
    
    def __init__(self, n_features, hidden_size=64, n_layers=2, 
                 latent_size=32, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        # Encoder: compress sequence to latent representation
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.encoder_to_latent = nn.Linear(hidden_size, latent_size)
        
        # Decoder: reconstruct sequence from latent
        self.latent_to_decoder = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_size, n_features)
    
    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # Use last hidden state
        latent = self.encoder_to_latent(hidden[-1])  # (batch, latent_size)
        
        # Decode
        decoder_input = self.latent_to_decoder(latent)  # (batch, hidden_size)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        decoder_out, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction, latent
    
    def get_anomaly_scores(self, x):
        """
        Compute per-metric anomaly scores based on reconstruction error.
        Returns scores in [0, 1] range.
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            # MSE per metric per timestep
            errors = (x - reconstruction) ** 2
            # Average over time dimension -> per-metric score
            metric_scores = errors.mean(dim=1)  # (batch, n_features)
        return metric_scores


class AnomalyDetector:
    """
    Wraps the LSTM autoencoder with training, threshold calibration,
    and anomaly detection logic.
    """
    
    def __init__(self, n_features, window_size=60, device='cpu'):
        self.window_size = window_size
        self.device = device
        self.model = LSTMAutoencoder(n_features).to(device)
        self.threshold_per_metric = None  # Calibrated from validation data
    
    def create_windows(self, df, stride=1):
        """Create sliding windows from dataframe."""
        values = df.values.astype(np.float32)
        windows = []
        for i in range(0, len(values) - self.window_size + 1, stride):
            windows.append(values[i:i + self.window_size])
        return torch.tensor(np.array(windows))
    
    def train(self, normal_df, epochs=50, lr=1e-3, val_split=0.2):
        """Train on normal data only."""
        windows = self.create_windows(normal_df, stride=5)
        
        # Split into train and validation
        split = int(len(windows) * (1 - val_split))
        train_data = windows[:split]
        val_data = windows[split:]
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                reconstruction, _ = self.model(batch)
                loss = criterion(reconstruction, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_data, criterion)
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {total_loss/len(train_loader):.6f} | "
                      f"Val Loss: {val_loss:.6f}")
        
        # Calibrate thresholds on validation data
        self._calibrate_thresholds(val_data)
    
    def _validate(self, val_data, criterion):
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            recon, _ = self.model(val_data)
            return criterion(recon, val_data).item()
    
    def _calibrate_thresholds(self, val_data, percentile=99):
        """Set anomaly thresholds at the Nth percentile of validation errors."""
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            scores = self.model.get_anomaly_scores(val_data)
            scores = scores.cpu().numpy()
            self.threshold_per_metric = np.percentile(
                scores, percentile, axis=0
            )
    
    def detect(self, df):
        """
        Run anomaly detection on new data.
        Returns DataFrame with anomaly scores and boolean flags.
        """
        windows = self.create_windows(df, stride=1)
        windows = windows.to(self.device)
        
        scores = self.model.get_anomaly_scores(windows).cpu().numpy()
        
        # Normalize scores to [0, 1] using calibrated thresholds
        normalized = scores / (self.threshold_per_metric + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        
        # Build results dataframe
        # Align with original timestamps (use last timestamp of each window)
        result_index = df.index[self.window_size - 1:]
        result_df = pd.DataFrame(
            normalized[:len(result_index)],
            index=result_index,
            columns=[f"{col}_anomaly_score" for col in df.columns]
        )
        
        # Boolean anomaly flags
        for i, col in enumerate(df.columns):
            result_df[f"{col}_is_anomaly"] = scores[:len(result_index), i] > \
                                              self.threshold_per_metric[i]
        
        return result_df
```

**Key design decisions here:**

The window size of 60 (at 1-minute intervals = 1 hour lookback) captures enough temporal context for the model to learn daily patterns. You'll want to experiment with this. The 99th percentile threshold calibration means roughly 1% of normal data will be flagged as anomalous, which is a good starting false positive rate you can tune. Training on normal data only means this is an unsupervised approach — you don't need labeled failures, which addresses the data scarcity challenge your literature review emphasizes.

---

### Phase 3: Causal Inference (Weeks 8–12)

This is the intellectually hardest part and what differentiates your project from a basic anomaly detector. Your literature review discusses several approaches: Granger causality (from the RUN framework), the PC algorithm (from CloudRCA), and knowledge-informed methods. For your implementation, I recommend a layered approach.

**5.1 — Granger Causality Analysis**

```python
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import networkx as nx

class CausalInferenceEngine:
    """
    Implements causal inference using Granger causality tests
    and the PC algorithm for causal graph construction.
    """
    
    def __init__(self, max_lag=10, significance_level=0.05):
        self.max_lag = max_lag
        self.significance = significance_level
    
    def granger_causality_matrix(self, df, anomalous_metrics):
        """
        Compute pairwise Granger causality between anomalous metrics.
        Only tests metrics that were flagged as anomalous to limit
        computational cost (addresses the O(n²) scaling concern).
        
        Returns: dict of {(cause, effect): {'p_value': ..., 'lag': ...}}
        """
        results = {}
        
        for cause_metric in anomalous_metrics:
            for effect_metric in anomalous_metrics:
                if cause_metric == effect_metric:
                    continue
                
                try:
                    test_data = df[[effect_metric, cause_metric]].dropna()
                    if len(test_data) < self.max_lag * 3:
                        continue
                    
                    gc_result = grangercausalitytests(
                        test_data, maxlag=self.max_lag, verbose=False
                    )
                    
                    # Find the best lag (lowest p-value)
                    best_lag = None
                    best_p = 1.0
                    for lag, test_dict in gc_result.items():
                        # Use the F-test p-value
                        p_val = test_dict[0]['ssr_ftest'][1]
                        if p_val < best_p:
                            best_p = p_val
                            best_lag = lag
                    
                    if best_p < self.significance:
                        results[(cause_metric, effect_metric)] = {
                            'p_value': best_p,
                            'optimal_lag': best_lag,
                            'strength': 1 - best_p  # Higher = stronger
                        }
                
                except Exception as e:
                    continue  # Skip pairs with insufficient data
        
        return results
    
    def build_causal_graph(self, granger_results, anomaly_scores):
        """
        Construct a directed causal graph from Granger results.
        Nodes are metrics, edges are causal relationships.
        """
        G = nx.DiGraph()
        
        # Add nodes with anomaly scores as attributes
        for metric, score in anomaly_scores.items():
            G.add_node(metric, anomaly_score=score)
        
        # Add edges from Granger causality results
        for (cause, effect), info in granger_results.items():
            G.add_edge(
                cause, effect,
                p_value=info['p_value'],
                lag=info['optimal_lag'],
                strength=info['strength']
            )
        
        # Prune cycles (causal graphs should be DAGs)
        # Use temporal precedence to break cycles
        G = self._break_cycles_by_temporal_precedence(G)
        
        return G
    
    def _break_cycles_by_temporal_precedence(self, G):
        """Remove edges that violate temporal ordering."""
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            # Find the weakest edge in the cycle and remove it
            weakest_edge = None
            weakest_strength = float('inf')
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    strength = G[u][v].get('strength', 0)
                    if strength < weakest_strength:
                        weakest_strength = strength
                        weakest_edge = (u, v)
            if weakest_edge:
                G.remove_edge(*weakest_edge)
        return G
    
    def correlate_events(self, anomaly_timestamps, events_df, 
                         window_hours=24):
        """
        Correlate detected anomalies with system events
        (deployments, config changes, etc.)
        """
        correlations = []
        
        for _, event in events_df.iterrows():
            event_time = event['timestamp']
            
            for metric, anomaly_time in anomaly_timestamps.items():
                time_diff = (anomaly_time - event_time).total_seconds() / 3600
                
                # Event must precede anomaly (cause before effect)
                if 0 < time_diff < window_hours:
                    correlations.append({
                        'event': event['description'],
                        'event_time': event_time,
                        'event_type': event.get('type', 'unknown'),
                        'affected_metric': metric,
                        'anomaly_time': anomaly_time,
                        'time_delta_hours': time_diff,
                        'correlation_score': 1.0 / (1.0 + time_diff)
                    })
        
        return sorted(correlations, key=lambda x: x['correlation_score'], 
                      reverse=True)
```

**5.2 — PC Algorithm Integration (for richer causal structure)**

The PC algorithm, which your literature review mentions is used in CloudRCA for structure learning, gives you a more principled causal graph than pairwise Granger tests alone. Use it as a complement:

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

def learn_causal_structure_pc(df, anomalous_metrics, alpha=0.05):
    """
    Use the PC algorithm to learn causal graph structure
    from observed metric data.
    """
    data = df[anomalous_metrics].dropna().values
    
    # Run PC algorithm with Fisher's Z conditional independence test
    cg = pc(data, alpha=alpha, indep_test=fisherz)
    
    # Convert to NetworkX graph
    G = nx.DiGraph()
    for i, metric in enumerate(anomalous_metrics):
        G.add_node(metric)
    
    # cg.G.graph contains the adjacency matrix
    adj_matrix = cg.G.graph
    for i in range(len(anomalous_metrics)):
        for j in range(len(anomalous_metrics)):
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                # Directed edge from j to i
                G.add_edge(anomalous_metrics[j], anomalous_metrics[i])
            elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                # Undirected edge - use Granger to orient
                G.add_edge(anomalous_metrics[i], anomalous_metrics[j],
                          undirected=True)
    
    return G
```

---

### Phase 4: Root Cause Ranking (Weeks 11–13)

**6.1 — Composite Root Cause Scoring**

This implements Stage 4 from your project doc. The scoring combines multiple signals, similar to how CloudRCA uses probabilistic inference and the RUN framework uses personalized PageRank:

```python
class RootCauseRanker:
    """
    Ranks candidate root causes using a composite score
    based on causal graph properties, temporal precedence,
    and anomaly severity.
    """
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'causal_outflow': 0.30,
            'causal_inflow_penalty': 0.15,
            'temporal_priority': 0.25,
            'anomaly_severity': 0.15,
            'event_correlation': 0.15
        }
    
    def rank(self, causal_graph, anomaly_scores, 
             anomaly_first_detected, event_correlations=None):
        """
        Score and rank each anomalous metric as a potential root cause.
        
        Returns: sorted list of (metric, score, explanation) tuples
        """
        candidates = []
        
        # Compute PageRank on reversed graph (high rank = influential cause)
        reversed_graph = causal_graph.reverse()
        try:
            pagerank_scores = nx.pagerank(reversed_graph, weight='strength')
        except:
            pagerank_scores = {n: 1.0/len(causal_graph) 
                              for n in causal_graph.nodes}
        
        all_anomaly_times = list(anomaly_first_detected.values())
        if all_anomaly_times:
            earliest = min(all_anomaly_times)
            latest = max(all_anomaly_times)
            time_range = (latest - earliest).total_seconds() or 1.0
        
        for metric in causal_graph.nodes:
            score_components = {}
            
            # 1. Causal outflow: how many things does this cause?
            out_degree = causal_graph.out_degree(metric)
            max_out = max(causal_graph.out_degree(n) 
                        for n in causal_graph.nodes) or 1
            score_components['causal_outflow'] = out_degree / max_out
            
            # 2. Causal inflow penalty: root causes have few incoming edges
            in_degree = causal_graph.in_degree(metric)
            max_in = max(causal_graph.in_degree(n) 
                        for n in causal_graph.nodes) or 1
            score_components['causal_inflow_penalty'] = 1 - (in_degree / max_in)
            
            # 3. Temporal priority: earlier anomalies more likely root causes
            if metric in anomaly_first_detected and all_anomaly_times:
                time_offset = (anomaly_first_detected[metric] - earliest
                              ).total_seconds()
                score_components['temporal_priority'] = 1 - (
                    time_offset / time_range
                )
            else:
                score_components['temporal_priority'] = 0.0
            
            # 4. Anomaly severity
            if metric in anomaly_scores:
                score_components['anomaly_severity'] = anomaly_scores[metric]
            else:
                score_components['anomaly_severity'] = 0.0
            
            # 5. Event correlation (deployment/config change nearby)
            if event_correlations:
                relevant = [ec for ec in event_correlations 
                           if ec['affected_metric'] == metric]
                if relevant:
                    score_components['event_correlation'] = \
                        relevant[0]['correlation_score']
                else:
                    score_components['event_correlation'] = 0.0
            else:
                score_components['event_correlation'] = 0.0
            
            # Composite score
            total_score = sum(
                self.weights[k] * score_components[k] 
                for k in self.weights
            )
            
            # Incorporate PageRank
            total_score = 0.7 * total_score + 0.3 * pagerank_scores.get(
                metric, 0
            )
            
            explanation = self._generate_explanation(
                metric, score_components, causal_graph, 
                anomaly_first_detected
            )
            
            candidates.append({
                'metric': metric,
                'confidence': round(total_score * 100, 1),
                'components': score_components,
                'explanation': explanation
            })
        
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates
    
    def _generate_explanation(self, metric, components, graph, timestamps):
        """Generate human-readable explanation for why this is a root cause."""
        parts = []
        
        if components['causal_outflow'] > 0.5:
            downstream = list(graph.successors(metric))
            parts.append(
                f"This metric causally influences {len(downstream)} "
                f"downstream metrics: {', '.join(downstream[:3])}"
            )
        
        if components['temporal_priority'] > 0.7:
            parts.append(
                f"This anomaly was among the earliest detected "
                f"(at {timestamps.get(metric, 'unknown')})"
            )
        
        if components['causal_inflow_penalty'] > 0.8:
            parts.append(
                "This metric has very few upstream causes, "
                "suggesting it is a source rather than a symptom"
            )
        
        if components['event_correlation'] > 0.5:
            parts.append(
                "A system event (deployment/config change) was detected "
                "shortly before this anomaly"
            )
        
        return "; ".join(parts) if parts else "Identified through composite scoring"
```

---

### Phase 5: Dashboard & Reporting (Weeks 13–16)

**7.1 — Streamlit Dashboard**

```python
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_dashboard():
    st.set_page_config(page_title="RCA System", layout="wide")
    st.title("🔍 Automated Root Cause Analysis Dashboard")
    
    # Sidebar: scenario selection
    st.sidebar.header("Configuration")
    scenario = st.sidebar.selectbox(
        "Select Failure Scenario",
        ["E-Commerce DB Migration", "SaaS Memory Leak", 
         "Network Partition", "Thread Pool Exhaustion",
         "Custom / Live Data"]
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Metric Timeline with Anomalies")
        # Plot metrics with anomaly scores overlaid
        fig = create_metric_timeline(metrics_df, anomaly_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔗 Causal Graph")
        # Interactive causal graph visualization
        fig_graph = create_causal_graph_viz(causal_graph)
        st.plotly_chart(fig_graph, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Root Cause Rankings")
        for i, candidate in enumerate(ranked_causes[:5]):
            with st.expander(
                f"#{i+1}: {candidate['metric']} "
                f"({candidate['confidence']}%)", 
                expanded=(i == 0)
            ):
                st.progress(candidate['confidence'] / 100)
                st.write(candidate['explanation'])
                
                # Show component breakdown
                st.json(candidate['components'])
        
        st.subheader("📋 Causal Chain")
        st.write(generate_causal_chain_text(ranked_causes[0], causal_graph))


def create_causal_graph_viz(G):
    """Create interactive Plotly visualization of causal graph."""
    pos = nx.spring_layout(G, k=2, seed=42)
    
    # Create edges
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=2 * data.get('strength', 0.5), color='#888'),
            hoverinfo='text',
            text=f"{u} → {v}<br>p={data.get('p_value', 'N/A'):.4f}"
        ))
    
    # Create nodes
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        score = G.nodes[node].get('anomaly_score', 0)
        node_text.append(f"{node}<br>Anomaly: {score:.2f}")
        node_color.append(score)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=20, color=node_color, 
                   colorscale='RdYlGn_r', showscale=True),
        text=[n.split('_')[-1] for n in G.nodes],
        textposition="top center",
        hovertext=node_text
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(showlegend=False, height=500)
    return fig
```

---

## 4. Connecting to Research: How Your System Maps to the Literature

Your literature review describes several state-of-the-art frameworks. Here's how your implementation relates to each and what elements you can cite in your report:

**CloudRCA (Alibaba):** Your system mirrors CloudRCA's two-stage pipeline — feature engineering followed by inference. CloudRCA uses a KHBN; you use Granger causality + PC algorithm + PageRank. In your writeup, position this as a simplification that trades some accuracy for implementation feasibility while preserving the core insight of separating feature extraction from causal reasoning.

**RUN Framework (Neural Granger):** Your Granger causality module directly implements this paper's core idea. The RUN framework adds a contrastive pre-training step and uses neural networks for the Granger test rather than classical statistical tests. If you want to reach "publication-worthy" level, implementing neural Granger causality (using your LSTM as the forecasting model) would be a meaningful extension.

**OOBN (Aviation):** The modular architecture concept from OOBN translates to how you structure your synthetic data generator — services are modular objects with encapsulated behavior. You aren't implementing full OOBN, but the principle of reducing spurious correlations through structural knowledge (your service dependency graph) is the same.

**MULAN (Multi-modal):** This is your stretch goal. If you add log analysis alongside metrics, you're moving toward multi-modal RCA. A simpler version: treat deployment events and config changes as a second modality and correlate them with metric anomalies (which your `correlate_events` method already does).

---

## 5. Evaluation Strategy

This is where many student projects fall short. Plan your evaluation carefully:

**5.1 — Synthetic Data Evaluation (Primary)**

Generate 50–100 failure scenarios across your supported failure types. For each, your system should identify the injected root cause. Measure top-1 accuracy (did the highest-ranked cause match the true root cause?), top-3 accuracy (was it in the top 3?), mean reciprocal rank, and detection latency (time from failure injection to alert).

**5.2 — Public Benchmark Evaluation**

Use the Azure VM Traces dataset or the SMD (Server Machine Dataset) from OmniAnomaly. These give you real-world metric patterns with labeled anomalies. They won't have labeled root causes, but you can evaluate anomaly detection precision/recall.

**5.3 — Ablation Studies**

This is what elevates your project. Remove each component one at a time and measure the impact. Run the system without Granger causality (just anomaly ranking by severity) to show causal inference adds value. Run without event correlation to show deployment/config change correlation matters. Run with different LSTM architectures (shallow vs. deep, with/without attention) to justify your design choices.

---

## 6. Realistic Timeline

| Weeks | Phase | Deliverable |
|-------|-------|-------------|
| 1–3 | Foundation | Synthetic data generator, project structure, basic data pipeline |
| 4–5 | Model Training | LSTM autoencoder trained, basic anomaly detection working |
| 6–7 | Anomaly Detection | Thresholding, per-metric scoring, evaluation on synthetic data |
| 8–10 | Causal Inference | Granger causality, PC algorithm, causal graph construction |
| 11–12 | Root Cause Ranking | Composite scoring, PageRank integration, event correlation |
| 13–14 | Dashboard | Streamlit UI, causal graph visualization, report generation |
| 15–16 | Evaluation | Comprehensive evaluation, ablation studies, benchmark testing |
| 17–18 | Documentation | Final report, code cleanup, demo preparation |
| 19–24 | Extensions (if time) | Prometheus integration, log analysis, neural Granger causality |

---

## 7. Critical Advice

**Start with the synthetic data generator and never stop using it.** Every time you build a new component, test it against known ground truth. This is your safety net.

**Don't try to detect all 50 anomaly types at once.** Start with 5 — memory leak, CPU saturation, connection pool exhaustion, latency spike, and error rate spike. These cover the most interesting failure patterns and are sufficient for a strong project. You can always add more later.

**The causal inference is your differentiator.** Any student can build an anomaly detector. The causal graph construction and root cause ranking is what makes this project "advanced." Invest disproportionate time here and make sure your evaluation clearly shows it adds value over simple anomaly ranking.

**Keep a research journal.** Document every experiment, every hyperparameter choice, every failed approach. This becomes your project report and will make writing the final documentation dramatically easier.

Want me to dive deeper into any specific component — for example, the neural Granger causality extension, the PC algorithm implementation details, or how to set up Prometheus integration for real metrics?