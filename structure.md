# Comprehensive Analysis: Building an AI-Powered Root Cause Analysis (RCA) System

I'll provide a detailed roadmap for implementing the RCA system based on both documents. This is an ambitious but highly feasible college project with strong practical value.

---

## 1. PROJECT OVERVIEW & FEASIBILITY ASSESSMENT

### 1.1 Why This Project Works for College

**Strengths:**
- **Multi-disciplinary**: Combines ML, statistics, systems engineering, and software design
- **Measurable outcomes**: Clear metrics (precision, recall, F1-score)
- **Scalable scope**: Can be MVP in 12 weeks or publication-ready in 24 weeks
- **Real-world relevance**: Companies like Google, Netflix, Amazon actively solve this
- **Impressive portfolio piece**: Demonstrates skills for SRE/DevOps/ML roles

**Feasibility Rating: 8.5/10**
- ✅ Mature technologies available (PyTorch, statsmodels, DoWhy)
- ✅ Public datasets available (NASA, Azure)
- ✅ Free GPU resources (Google Colab, Kaggle)
- ⚠️ Moderate complexity in causal inference implementation
- ⚠️ Production integration is time-consuming

---

## 2. CORE ARCHITECTURE BREAKDOWN

### 2.1 System Pipeline (4-Stage Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    RCA SYSTEM PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STAGE 1: NORMAL BEHAVIOR LEARNING                          │
│  ├─ Data Ingestion: Collect healthy metrics (30-90 days)   │
│  ├─ Model: LSTM Autoencoder / Transformer                   │
│  └─ Output: Learned normal patterns                          │
│                                                              │
│  STAGE 2: ANOMALY DETECTION                                 │
│  ├─ Method 1: Reconstruction Error                          │
│  ├─ Method 2: Forecast Deviation                            │
│  └─ Output: Anomalies with confidence scores                │
│                                                              │
│  STAGE 3: CAUSAL INFERENCE                                  │
│  ├─ Granger Causality Tests                                 │
│  ├─ Temporal Precedence Analysis                            │
│  ├─ Causal Graph Construction (PC Algorithm)                │
│  └─ Output: Causal relationships between metrics            │
│                                                              │
│  STAGE 4: ROOT CAUSE RANKING                                │
│  ├─ Causal Outflow Score                                    │
│  ├─ Causal Inflow Score                                     │
│  ├─ Temporal Priority Score                                 │
│  └─ Output: Ranked root causes with confidence %            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. DETAILED IMPLEMENTATION ROADMAP

### 3.1 Phase 1: Foundation Setup (Weeks 1-2)

#### Step 1.1: Development Environment

```python
# requirements.txt
# Core Dependencies
python==3.10
torch==2.0.0
pytorch-lightning==2.0.0
transformers==4.30.0

# Time Series & Statistics
pandas==2.0.0
numpy==1.24.0
statsmodels==0.14.0
scikit-learn==1.3.0
scipy==1.11.0

# Causal Inference
dowhy==0.10.1
causalnex==0.13.0
tigramite==4.2.1

# Graph Processing
networkx==3.1
igraph==0.10.4

# Visualization
plotly==5.14.0
streamlit==1.25.0
matplotlib==3.7.0
seaborn==0.12.0

# Monitoring Integration
prometheus-client==0.17.0
boto3==1.28.0  # AWS CloudWatch

# Infrastructure
docker==6.1.0
pytest==7.4.0
black==23.7.0
```

#### Step 1.2: Project Structure

```
rca-system/
├── data/
│   ├── raw/                 # Raw metrics from monitoring
│   ├── synthetic/           # Generated failure scenarios
│   └── benchmarks/          # NASA, Azure datasets
├── src/
│   ├── data_ingestion/
│   │   ├── prometheus_client.py
│   │   ├── cloudwatch_client.py
│   │   └── data_normalizer.py
│   ├── preprocessing/
│   │   ├── time_series_utils.py
│   │   ├── feature_engineering.py
│   │   └── window_generator.py
│   ├── models/
│   │   ├── lstm_autoencoder.py
│   │   ├── transformer_model.py
│   │   └── train.py
│   ├── anomaly_detection/
│   │   ├── reconstruction_error.py
│   │   ├── forecast_deviation.py
│   │   └── anomaly_scorer.py
│   ├── causal_inference/
│   │   ├── granger_causality.py
│   │   ├── causal_graph.py
│   │   ├── pc_algorithm.py
│   │   └── temporal_analysis.py
│   ├── root_cause_ranking/
│   │   ├── scorer.py
│   │   └── rank_aggregator.py
│   └── reporting/
│       ├── report_generator.py
│       └── visualization.py
├── tests/
│   ├── test_models.py
│   ├── test_anomaly_detection.py
│   ├── test_causal_inference.py
│   └── test_integration.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_case_studies.ipynb
│   └── 04_evaluation.ipynb
├── config/
│   ├── config.yaml
│   └── logging.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

### 3.2 Phase 2: Data Pipeline (Weeks 3-4)

#### Step 2.1: Synthetic Data Generator

This is crucial for development without real production data.

```python
# src/data_ingestion/synthetic_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class SyntheticMetricsGenerator:
    """Generates realistic system metrics with known failure patterns"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_normal_behavior(
        self, 
        duration_days: int = 60,
        sampling_interval_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Generate 60 days of normal operational metrics
        
        Metrics:
        - CPU, Memory, Disk I/O
        - API latency (P50, P95, P99)
        - Error rate
        - Database connections
        - Cache hit rate
        """
        
        num_samples = (duration_days * 24 * 60) // sampling_interval_minutes
        timestamps = pd.date_range(
            start='2024-01-01', 
            periods=num_samples, 
            freq=f'{sampling_interval_minutes}min'
        )
        
        data = {
            'timestamp': timestamps,
            'cpu_utilization': self._generate_cpu_normal(),
            'memory_usage_percent': self._generate_memory_normal(),
            'api_latency_p50_ms': self._generate_latency_p50(),
            'api_latency_p95_ms': self._generate_latency_p95(),
            'api_latency_p99_ms': self._generate_latency_p99(),
            'error_rate_percent': self._generate_error_rate_normal(),
            'db_connections_active': self._generate_db_connections(),
            'cache_hit_rate': self._generate_cache_hit_rate(),
            'request_throughput': self._generate_throughput(num_samples),
            'disk_io_wait_percent': self._generate_disk_io(),
        }
        
        return pd.DataFrame(data)
    
    def _generate_cpu_normal(self, num_samples: int) -> np.ndarray:
        """
        CPU follows diurnal pattern:
        - Low during night (10-20%)
        - Medium during day (40-60%)
        - Peaks during business hours (50-70%)
        """
        time_of_day = np.arange(num_samples) % 288  # 24h / 5min
        
        # Diurnal pattern
        base = 40 + 20 * np.sin(2 * np.pi * time_of_day / 288)
        
        # Add random noise
        noise = np.random.normal(0, 3, num_samples)
        
        cpu = np.clip(base + noise, 5, 95)
        return cpu
    
    def _generate_memory_normal(self, num_samples: int) -> np.ndarray:
        """Memory with slow oscillations due to garbage collection"""
        # Slow upward trend broken by GC drops
        base = 60 + 0.01 * np.arange(num_samples)
        
        # Add GC drops every ~100 samples
        gc_pattern = np.sin(2 * np.pi * np.arange(num_samples) / 100) * 5
        
        noise = np.random.normal(0, 2, num_samples)
        memory = np.clip(base + gc_pattern + noise, 30, 95)
        
        return memory
    
    def _generate_latency_p50(self, num_samples: int) -> np.ndarray:
        """P50 latency baseline"""
        return np.clip(50 + np.random.normal(0, 5, num_samples), 20, 100)
    
    def _generate_latency_p95(self, num_samples: int) -> np.ndarray:
        """P95 usually 2-3x P50"""
        p50 = self._generate_latency_p50(num_samples)
        return p50 * 2.5 + np.random.normal(0, 10, num_samples)
    
    def _generate_latency_p99(self, num_samples: int) -> np.ndarray:
        """P99 usually 5-10x P50"""
        p50 = self._generate_latency_p50(num_samples)
        return p50 * 7 + np.random.normal(0, 20, num_samples)
    
    def _generate_error_rate_normal(self, num_samples: int) -> np.ndarray:
        """Very low error rate during normal operation"""
        return np.abs(np.random.normal(0.01, 0.005, num_samples))
    
    def _generate_db_connections(self, num_samples: int) -> np.ndarray:
        """Database connections with business hour peaks"""
        time_of_day = np.arange(num_samples) % 288
        
        # Peak during business hours (9 AM - 5 PM)
        business_hours = (time_of_day > 108) & (time_of_day < 240)
        base = np.where(business_hours, 65, 25)
        
        noise = np.random.normal(0, 5, num_samples)
        return np.clip(base + noise, 5, 100)
    
    def _generate_cache_hit_rate(self, num_samples: int) -> np.ndarray:
        """Cache hit rate between 85-95%"""
        return 90 + np.random.normal(0, 3, num_samples)
    
    def _generate_throughput(self, num_samples: int) -> np.ndarray:
        """Request throughput (requests/sec)"""
        time_of_day = np.arange(num_samples) % 288
        
        # Diurnal pattern
        base = 1000 + 500 * np.sin(2 * np.pi * time_of_day / 288)
        noise = np.random.normal(0, 50, num_samples)
        
        return np.clip(base + noise, 100, 2000)
    
    def _generate_disk_io(self, num_samples: int) -> np.ndarray:
        """Disk I/O wait percentage"""
        return np.clip(np.random.normal(5, 2, num_samples), 1, 15)
    
    def inject_failure_scenario(
        self, 
        df: pd.DataFrame, 
        failure_type: str,
        start_idx: int,
        duration_samples: int,
        severity: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject synthetic failure into normal data
        
        failure_type options:
        - 'database_slow_query': Gradually increasing query latency
        - 'memory_leak': Monotonic memory increase
        - 'network_partition': Spike in latency + error rate
        - 'thread_pool_exhaustion': API latency spike + DB connection spike
        - 'cpu_spike': Sudden CPU utilization increase
        """
        
        df_copy = df.copy()
        metadata = {
            'failure_type': failure_type,
            'start_time': df_copy.iloc[start_idx]['timestamp'],
            'start_idx': start_idx,
            'duration_samples': duration_samples,
            'severity': severity,
            'root_cause': None,
            'causal_chain': []
        }
        
        if failure_type == 'database_slow_query':
            # Gradually increase query latency
            for i in range(duration_samples):
                progress = i / duration_samples
                multiplier = 1 + (9 * progress * severity)  # Up to 10x slower
                
                idx = start_idx + i
                df_copy.loc[idx, 'api_latency_p50_ms'] *= multiplier
                df_copy.loc[idx, 'api_latency_p95_ms'] *= multiplier
                df_copy.loc[idx, 'api_latency_p99_ms'] *= multiplier
            
            # After latency increase, connections pool gets exhausted
            pool_exhaust_idx = start_idx + int(duration_samples * 0.3)
            for i in range(duration_samples - int(duration_samples * 0.3)):
                idx = pool_exhaust_idx + i
                df_copy.loc[idx, 'db_connections_active'] = 95 + np.random.normal(3, 1)
            
            # Finally, errors spike
            error_spike_idx = start_idx + int(duration_samples * 0.6)
            for i in range(error_spike_idx, start_idx + duration_samples):
                df_copy.loc[i, 'error_rate_percent'] = 10 + np.random.normal(0, 2)
            
            metadata['root_cause'] = 'Database schema migration causing inefficient index'
            metadata['causal_chain'] = [
                'Slow database queries',
                'Connection pool exhaustion',
                'API timeouts',
                'High error rate'
            ]
        
        elif failure_type == 'memory_leak':
            # Monotonic memory increase over time
            for i in range(duration_samples):
                idx = start_idx + i
                progress = i / duration_samples
                increase = 30 * progress * severity  # Increase by 30%
                df_copy.loc[idx, 'memory_usage_percent'] += increase
            
            metadata['root_cause'] = 'Unclosed file handles in new code deployment'
            metadata['causal_chain'] = [
                'Leaked connections accumulating',
                'Memory pressure increasing',
                'Garbage collection overhead',
                'Eventually OOM kill'
            ]
        
        elif failure_type == 'network_partition':
            # Sudden spike in latency and errors
            for i in range(duration_samples):
                idx = start_idx + i
                df_copy.loc[idx, 'api_latency_p95_ms'] = 5000 + np.random.normal(0, 500)
                df_copy.loc[idx, 'api_latency_p99_ms'] = 7000 + np.random.normal(0, 500)
                df_copy.loc[idx, 'error_rate_percent'] = 40 + np.random.normal(0, 5)
            
            metadata['root_cause'] = 'Inter-region network partition'
            metadata['causal_chain'] = [
                'Network partition detected',
                'Cross-region requests timeout',
                'Service errors spike'
            ]
        
        elif failure_type == 'thread_pool_exhaustion':
            # API latency spikes, DB connections max out
            for i in range(duration_samples):
                idx = start_idx + i
                df_copy.loc[idx, 'api_latency_p50_ms'] = 1000 + np.random.normal(0, 100)
                df_copy.loc[idx, 'db_connections_active'] = 99 + np.random.normal(1, 0.5)
                df_copy.loc[idx, 'error_rate_percent'] = 25 + np.random.normal(0, 3)
            
            metadata['root_cause'] = 'Background job blocking API thread pool'
            metadata['causal_chain'] = [
                'Ingestion job started',
                'Synchronous DB queries blocking',
                'Thread pool exhausted',
                'API requests timeout'
            ]
        
        elif failure_type == 'cpu_spike':
            # Sudden CPU increase
            for i in range(duration_samples):
                idx = start_idx + i
                df_copy.loc[idx, 'cpu_utilization'] = 85 + np.random.normal(0, 3)
                df_copy.loc[idx, 'api_latency_p95_ms'] *= 1.5
            
            metadata['root_cause'] = 'Runaway process consuming CPU'
            metadata['causal_chain'] = [
                'Infinite loop in new code',
                'CPU utilization spike',
                'Request latency increases'
            ]
        
        return df_copy, metadata


# Usage Example
if __name__ == "__main__":
    generator = SyntheticMetricsGenerator()
    
    # Generate 60 days of normal data
    normal_data = generator.generate_normal_behavior(duration_days=60)
    print(f"Normal data shape: {normal_data.shape}")
    
    # Inject a failure at day 50
    failure_start = 50 * 24 * 60 // 5  # Convert days to samples
    data_with_failure, metadata = generator.inject_failure_scenario(
        normal_data,
        failure_type='database_slow_query',
        start_idx=failure_start,
        duration_samples=200,  # ~17 hours
        severity=0.8
    )
    
    print(f"Failure metadata: {json.dumps(metadata, indent=2, default=str)}")
    print(f"Error rate before failure: {normal_data.iloc[failure_start-10]['error_rate_percent']:.2f}%")
    print(f"Error rate after failure: {data_with_failure.iloc[failure_start+200]['error_rate_percent']:.2f}%")
```

---

### 3.3 Phase 3: Deep Learning Models (Weeks 5-7)

#### Step 3.1: LSTM Autoencoder for Anomaly Detection

```python
# src/models/lstm_autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection
    
    Architecture:
    - Encoder: LSTM layers that compress sequence to latent representation
    - Decoder: LSTM layers that reconstruct original sequence from latent
    
    Anomaly Score = Reconstruction Error = ||original - reconstructed||
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Encoder: Compress sequence to latent vector
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Bottleneck: Project to latent dimension
        self.encoder_proj = nn.Linear(64, latent_dim)
        
        # Decoder: Expand latent vector back to sequence
        # First, repeat latent vector for each timestep
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=64,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection back to original dimension
        self.decoder_proj = nn.Linear(64, input_size)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            latent: (batch_size, latent_dim)
        """
        _, (h_n, _) = self.encoder_lstm(x)
        # Take last hidden state from last layer
        latent = self.encoder_proj(h_n[-1])  # (batch_size, latent_dim)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to sequence
        
        Args:
            latent: (batch_size, latent_dim)
        
        Returns:
            reconstructed: (batch_size, sequence_length, input_size)
        """
        # Repeat latent for each timestep
        latent_expanded = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # (batch_size, sequence_length, latent_dim)
        
        decoder_output, _ = self.decoder_lstm(latent_expanded)
        # (batch_size, sequence_length, 64)
        
        reconstructed = self.decoder_proj(decoder_output)
        # (batch_size, sequence_length, input_size)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode
        
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            reconstructed: (batch_size, sequence_length, input_size)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed
    
    def get_reconstruction_error(
        self, 
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reconstruction error for anomaly detection
        
        Args:
            x: (batch_size, sequence_length, input_size)
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            error: scalar or (batch_size,) depending on reduction
            reconstructed: reconstructed sequence
        """
        reconstructed = self.forward(x)
        
        # MSE per sample (mean across timesteps and features)
        mse = torch.mean((x - reconstructed) ** 2, dim=[1, 2])
        # (batch_size,)
        
        if reduction == 'mean':
            return torch.mean(mse), reconstructed
        elif reduction == 'sum':
            return torch.sum(mse), reconstructed
        else:  # 'none'
            return mse, reconstructed


class AnomalyDetectionTrainer:
    """Train LSTM autoencoder on normal data"""
    
    def __init__(
        self,
        model: LSTMAutoencoder,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x in train_loader:
            batch_x = batch_x[0].to(self.device)  # DataLoader returns tuple
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(batch_x)
            loss = self.criterion(batch_x, reconstructed)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x[0].to(self.device)
                reconstructed = self.model(batch_x)
                loss = self.criterion(batch_x, reconstructed)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10
    ):
        """Train with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    f'best_model.pt'
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def plot_history(self):
        """Visualize training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Model Training History')
        plt.grid(True)
        plt.show()


# Example usage
def prepare_sequences(
    data: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM
    
    Args:
        data: (num_samples, num_features)
        sequence_length: length of each sequence
    
    Returns:
        X: (num_sequences, sequence_length, num_features)
        indices: original indices for tracking
    """
    X = []
    indices = []
    
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
        indices.append(i)
    
    return np.array(X), np.array(indices)


if __name__ == "__main__":
    # Example: Train autoencoder
    from sklearn.preprocessing import StandardScaler
    
    # Generate synthetic normal data
    num_samples = 10000
    num_features = 10
    normal_data = np.random.randn(num_samples, num_features) * 10 + 50
    
    # Normalize
    scaler = StandardScaler()
    normal_data = scaler.fit_transform(normal_data)
    
    # Create sequences
    sequence_length = 60
    X, _ = prepare_sequences(normal_data, sequence_length)
    
    # Split train/val
    train_size = int(0.8 * len(X))
    X_train = torch.from_numpy(X[:train_size]).float()
    X_val = torch.from_numpy(X[train_size:]).float()
    
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train model
    model = LSTMAutoencoder(
        input_size=num_features,
        sequence_length=sequence_length,
        latent_dim=32
    )
    
    trainer = AnomalyDetectionTrainer(model)
    trainer.fit(train_loader, val_loader, epochs=50)
    trainer.plot_history()
```

---

### 3.4 Phase 4: Causal Inference Engine (Weeks 8-10)

#### Step 4.1: Granger Causality & PC Algorithm

```python
# src/causal_inference/granger_causality.py

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from typing import Dict, List, Tuple, Set
import networkx as nx
import itertools

class GrangerCausalityAnalyzer:
    """
    Granger Causality: Time series X "Granger-causes" Y if 
    past values of X improve prediction of Y beyond using Y's own past
    
    Assumption: Causation requires TEMPORAL PRECEDENCE
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
    
    def test_causality(
        self,
        data: pd.DataFrame,
        max_lag: int = 10,
        verbose: bool = False
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Test Granger causality between all pairs of metrics
        
        Args:
            data: (num_timesteps, num_metrics) dataframe
            max_lag: maximum lag to test
            verbose: print results
        
        Returns:
            results: dict mapping (cause, effect) -> test_results
        """
        
        results = {}
        metrics = data.columns.tolist()
        
        # Test all ordered pairs
        for cause, effect in itertools.permutations(metrics, 2):
            try:
                # Granger causality test: does cause help predict effect?
                gc_result = grangercausalitytests(
                    data[[effect, cause]],  # Order: effect, cause
                    max_lag,
                    verbose=verbose
                )
                
                # Extract p-values for each lag
                pvals = [gc_result[lag][0][0][1] for lag in range(1, max_lag + 1)]
                best_lag = np.argmin(pvals) + 1
                best_pval = pvals[best_lag - 1]
                
                is_causal = best_pval < self.significance_level
                
                results[(cause, effect)] = {
                    'best_lag': best_lag,
                    'best_pval': best_pval,
                    'is_causal': is_causal,
                    'all_pvals': pvals
                }
                
                if is_causal and verbose:
                    print(f"✓ {cause} → {effect} (lag={best_lag}, p={best_pval:.4f})")
            
            except Exception as e:
                if verbose:
                    print(f"✗ Error testing {cause} → {effect}: {str(e)}")
                results[(cause, effect)] = {
                    'error': str(e)
                }
        
        self.results = results
        return results
    
    def build_causal_graph(self) -> nx.DiGraph:
        """
        Build directed graph where edge A→B means A Granger-causes B
        """
        graph = nx.DiGraph()
        
        for (cause, effect), result in self.results.items():
            if result.get('is_causal', False):
                # Weight edge by -log(p-value) for strength
                pval = result['best_pval']
                weight = -np.log10(pval + 1e-10)  # Avoid log(0)
                graph.add_edge(cause, effect, weight=weight, lag=result['best_lag'])
        
        return graph


class PCAlgorithm:
    """
    PC Algorithm (Peter-Clark) for causal discovery
    
    Learns causal structure from observational data by iteratively:
    1. Starting with fully connected undirected graph
    2. Removing edges where variables are conditionally independent
    3. Orienting remaining edges based on v-structures
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def _conditional_independence_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray = None,
        method: str = 'partial_correlation'
    ) -> Tuple[bool, float]:
        """
        Test if X and Y are conditionally independent given Z
        
        If p_value > significance_level, they are independent
        """
        
        if method == 'partial_correlation':
            if Z is None or len(Z) == 0:
                # Unconditional correlation
                corr = np.corrcoef(X, Y)[0, 1]
                n = len(X)
                t_stat = corr * np.sqrt(n - 2) / (np.sqrt(1 - corr**2) + 1e-10)
                pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
            else:
                # Partial correlation: correlation after removing linear effect of Z
                # X_residual = X - E[X|Z]
                Z_design = np.column_stack([np.ones(len(Z)), Z])
                X_residual = X - Z_design @ np.linalg.lstsq(Z_design, X, rcond=None)[0]
                Y_residual = Y - Z_design @ np.linalg.lstsq(Z_design, Y, rcond=None)[0]
                
                partial_corr = np.corrcoef(X_residual, Y_residual)[0, 1]
                n = len(X)
                degrees_freed = n - 2 - Z.shape[1] if Z.ndim > 1 else n - 3
                t_stat = partial_corr * np.sqrt(degrees_freed) / (np.sqrt(1 - partial_corr**2) + 1e-10)
                pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), degrees_freed))
            
            independent = pval > self.significance_level
            return independent, pval
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def learn_structure(self, data: np.ndarray) -> nx.Graph:
        """
        Learn causal structure using PC algorithm
        
        Args:
            data: (num_samples, num_features)
        
        Returns:
            graph: undirected graph (edges represent relationships)
        """
        
        n_vars = data.shape[1]
        
        # Start with fully connected undirected graph
        graph = nx.complete_graph(n_vars)
        
        # Iteratively remove edges based on conditional independence
        for depth in range(n_vars - 1):
            edges_to_remove = []
            
            for edge in graph.edges():
                i, j = edge
                
                # Find neighbors of i (excluding j)
                neighbors_i = set(graph.neighbors(i)) - {j}
                
                # Test conditional independence with all subsets of neighbors
                for Z_size in range(depth + 1):
                    for Z in itertools.combinations(neighbors_i, Z_size):
                        if len(Z) == 0:
                            Z_data = None
                        else:
                            Z_data = data[:, list(Z)]
                        
                        independent, _ = self._conditional_independence_test(
                            data[:, i],
                            data[:, j],
                            Z_data
                        )
                        
                        if independent:
                            edges_to_remove.append(edge)
                            break  # Found conditioning set, remove edge
            
            # Remove edges
            graph.remove_edges_from(edges_to_remove)
        
        return graph


# Usage example
if __name__ == "__main__":
    # Generate synthetic causal data
    np.random.seed(42)
    n_samples = 1000
    
    # True causal structure: X → Y → Z
    X = np.random.normal(0, 1, n_samples)
    Y = 2 * X + np.random.normal(0, 0.5, n_samples)
    Z = 1.5 * Y + np.random.normal(0, 0.5, n_samples)
    W = np.random.normal(0, 1, n_samples)  # Independent
    
    data = pd.DataFrame({
        'X': X,
        'Y': Y,
        'Z': Z,
        'W': W
    })
    
    # Test Granger causality
    gc = GrangerCausalityAnalyzer(significance_level=0.05)
    results = gc.test_causality(data, max_lag=5, verbose=True)
    
    # Build graph
    causal_graph = gc.build_causal_graph()
    print(f"\nCausal edges found: {list(causal_graph.edges())}")
```

---

### 3.5 Phase 5: Root Cause Ranking & Scoring (Weeks 11-12)

```python
# src/root_cause_ranking/scorer.py

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import pandas as pd

class RootCauseScorer:
    """
    Score anomalies as potential root causes using multiple criteria:
    1. Causal Outflow: How many metrics does this anomaly cause?
    2. Causal Inflow: How many metrics cause this anomaly?
    3. Temporal Priority: How early did this anomaly appear?
    4. Anomaly Severity: How strong is the deviation from normal?
    """
    
    def __init__(self):
        self.weights = {
            'outflow': 0.4,      # 40%: downstream impact
            'inflow': 0.2,       # 20%: upstream dependencies
            'temporal': 0.3,     # 30%: time priority
            'severity': 0.1      # 10%: anomaly strength
        }
    
    def score_root_causes(
        self,
        anomalies: Dict[str, float],  # metric -> anomaly_score
        causal_graph: nx.DiGraph,
        anomaly_timestamps: Dict[str, int],  # metric -> first detection index
        earliest_anomaly_time: int
    ) -> List[Tuple[str, float, Dict]]:
        """
        Score all anomalous metrics as potential root causes
        
        Returns:
            sorted_causes: List of (metric, score, details)
        """
        
        scores = {}
        details = {}
        
        for metric, anomaly_score in anomalies.items():
            if anomaly_score < 0.5:  # Skip low-confidence anomalies
                continue
            
            # 1. Causal Outflow: How many downstream effects?
            outflow = sum(1 for _ in nx.descendants(causal_graph, metric))
            outflow_score = min(outflow / max(len(causal_graph), 1), 1.0)
            
            # 2. Causal Inflow: How many upstream causes?
            inflow = sum(1 for _ in nx.ancestors(causal_graph, metric))
            # Invert: metrics with few upstream causes are more likely root
            inflow_score = 1.0 - min(inflow / max(len(causal_graph), 1), 1.0)
            
            # 3. Temporal Priority: Did this anomaly appear first?
            time_of_anomaly = anomaly_timestamps.get(metric, earliest_anomaly_time)
            time_score = 1.0 - (time_of_anomaly - earliest_anomaly_time) / max(1, 1000)
            time_score = max(time_score, 0.0)
            
            # 4. Severity: How strong is the anomaly?
            severity_score = anomaly_score
            
            # Weighted combination
            final_score = (
                self.weights['outflow'] * outflow_score +
                self.weights['inflow'] * inflow_score +
                self.weights['temporal'] * time_score +
                self.weights['severity'] * severity_score
            )
            
            scores[metric] = final_score
            details[metric] = {
                'outflow_score': outflow_score,
                'inflow_score': inflow_score,
                'temporal_score': time_score,
                'severity_score': severity_score,
                'downstream_metrics': list(nx.descendants(causal_graph, metric)),
                'upstream_metrics': list(nx.ancestors(causal_graph, metric))
            }
        
        # Sort by score (descending)
        sorted_causes = sorted(
            [(metric, score, details[metric]) for metric, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_causes


class CausalChainTracer:
    """Trace causal chain from root cause through all effects"""
    
    @staticmethod
    def trace_to_leaf(
        start_node: str,
        causal_graph: nx.DiGraph
    ) -> List[List[str]]:
        """Find all paths from start_node to leaf nodes"""
        
        paths = []
        
        def dfs(node: str, current_path: List[str]):
            current_path.append(node)
            
            # Get immediate children
            children = list(causal_graph.successors(node))
            
            if not children:  # Leaf node
                paths.append(current_path.copy())
            else:
                for child in children:
                    dfs(child, current_path)
            
            current_path.pop()
        
        dfs(start_node, [])
        return paths
    
    @staticmethod
    def format_causal_chain(chain: List[str]) -> str:
        """Format causal chain as human-readable string"""
        return " → ".join(chain)


# Usage example
if __name__ == "__main__":
    # Example anomalies
    anomalies = {
        'db_query_latency': 0.91,
        'db_connections': 0.89,
        'api_latency': 0.96,
        'error_rate': 0.98,
        'cpu_usage': 0.12  # Low score, not anomalous
    }
    
    # Example causal graph
    causal_graph = nx.DiGraph()
    causal_graph.add_edges_from([
        ('db_query_latency', 'db_connections'),
        ('db_connections', 'api_latency'),
        ('api_latency', 'error_rate')
    ])
    
    # Example timestamps (sample indices)
    anomaly_timestamps = {
        'db_query_latency': 100,
        'db_connections': 110,
        'api_latency': 115,
        'error_rate': 120
    }
    
    # Score root causes
    scorer = RootCauseScorer()
    ranked_causes = scorer.score_root_causes(
        anomalies,
        causal_graph,
        anomaly_timestamps,
        earliest_anomaly_time=100
    )
    
    print("Ranked Root Causes:")
    for i, (metric, score, details) in enumerate(ranked_causes, 1):
        print(f"\n{i}. {metric}: {score:.2%}")
        print(f"   Downstream: {details['downstream_metrics']}")
        print(f"   Upstream: {details['upstream_metrics']}")
    
    # Trace causal chains
    tracer = CausalChainTracer()
    for metric, _, _ in ranked_causes[:1]:
        chains = tracer.trace_to_leaf(metric, causal_graph)
        print(f"\nCausal chains from {metric}:")
        for chain in chains:
            print(f"  {tracer.format_causal_chain(chain)}")
```

---

### 3.6 Phase 6: Reporting & Visualization (Weeks 13-14)

```python
# src/reporting/report_generator.py

import json
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

class RCAReportGenerator:
    """Generate comprehensive RCA reports"""
    
    @staticmethod
    def generate_text_report(
        root_causes: List[Tuple[str, float, Dict]],
        anomalies: Dict[str, float],
        causal_graph: nx.DiGraph,
        failure_duration_minutes: int,
        estimated_impact: str = "Critical"
    ) -> str:
        """Generate human-readable RCA report"""
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        ROOT CAUSE ANALYSIS REPORT                          ║
║                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
╚════════════════════════════════════════════════════════════════════════════╝

INCIDENT SUMMARY
{'═' * 76}
Failure Duration: {failure_duration_minutes} minutes
Estimated Impact: {estimated_impact}
Number of Anomalies Detected: {len(anomalies)}
Confidence Level: High

PRIMARY ROOT CAUSE
{'─' * 76}
Metric: {root_causes[0][0]}
Confidence: {root_causes[0][1]:.1%}

CONTRIBUTING FACTORS
{'─' * 76}
"""
        
        for i, (metric, score, details) in enumerate(root_causes[1:4], 1):
            report += f"{i}. {metric}: {score:.1%}\n"
        
        report += f"""
CAUSAL CHAIN
{'─' * 76}
"""
        
        # Trace causal path
        from src.root_cause_ranking.scorer import CausalChainTracer
        tracer = CausalChainTracer()
        chains = tracer.trace_to_leaf(root_causes[0][0], causal_graph)
        
        for chain in chains:
            report += f"  {tracer.format_causal_chain(chain)}\n"
        
        report += f"""
ANOMALY SUMMARY
{'─' * 76}
"""
        for metric, score in sorted(anomalies.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"  {metric:.<40} {score:>6.1%}\n"
        
        report += f"""
IMMEDIATE REMEDIATION
{'─' * 76}
1. [PRIORITY 1] Immediate action (Time: <5 min)
2. [PRIORITY 2] Secondary action (Time: <30 min)
3. [SHORT-TERM] Within business day

LONG-TERM PREVENTION
{'─' * 76}
- Implement monitoring for early detection
- Add automated remediation
- Update runbooks and alerts

SUPPORTING EVIDENCE
{'─' * 76}
- Temporal precedence confirmed
- Statistical significance: p < 0.05
- Consistent with historical patterns

"""
        
        return report
    
    @staticmethod
    def generate_json_report(
        root_causes: List[Tuple[str, float, Dict]],
        anomalies: Dict[str, float],
        metrics_timeline: pd.DataFrame,
        generated_at: str = None
    ) -> Dict:
        """Generate structured JSON report"""
        
        if generated_at is None:
            generated_at = datetime.now().isoformat()
        
        return {
            'metadata': {
                'generated_at': generated_at,
                'version': '1.0',
                'algorithm_version': 'RCA-v2024-01'
            },
            'summary': {
                'total_anomalies': len(anomalies),
                'primary_root_cause': root_causes[0][0],
                'confidence': float(root_causes[0][1]),
                'num_contributing_factors': len(root_causes)
            },
            'root_causes': [
                {
                    'rank': i + 1,
                    'metric': metric,
                    'confidence': float(score),
                    'details': {
                        k: v if not isinstance(v, set) else list(v)
                        for k, v in details.items()
                    }
                }
                for i, (metric, score, details) in enumerate(root_causes)
            ],
            'anomalies': {
                metric: float(score)
                for metric, score in sorted(
                    anomalies.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            },
            'metrics_timeline': metrics_timeline.to_dict(orient='records')
        }


class RCAVisualizer:
    """Create interactive visualizations"""
    
    @staticmethod
    def plot_metrics_timeline(
        metrics_timeline: pd.DataFrame,
        anomaly_scores: Dict[str, List[float]]
    ) -> go.Figure:
        """Plot metrics over time with anomaly highlights"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Metrics Timeline", "Anomaly Scores"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Plot metrics
        for col in metrics_timeline.columns:
            if col != 'timestamp':
                fig.add_trace(
                    go.Scatter(
                        x=metrics_timeline['timestamp'],
                        y=metrics_timeline[col],
                        mode='lines',
                        name=col
                    ),
                    row=1, col=1
                )
        
        # Plot anomaly scores
        for metric, scores in anomaly_scores.items():
            fig.add_trace(
                go.Scatter(
                    y=scores,
                    mode='lines',
                    name=f"{metric} anomaly",
                    fill='tozeroy'
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="RCA Timeline Analysis")
        return fig
    
    @staticmethod
    def plot_causal_graph(
        causal_graph: nx.DiGraph,
        root_causes: List[Tuple[str, float, Dict]]
    ) -> go.Figure:
        """Visualize causal graph"""
        
        pos = nx.spring_layout(causal_graph, k=2, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in causal_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        root_cause_names = {rc[0] for rc in root_causes[:3]}
        
        for node in causal_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            if node in root_cause_names:
                node_color.append('red')
            else:
                node_color.append('lightblue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=node_color,
                size=20,
                line_width=2
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Causal Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig


# Usage
if __name__ == "__main__":
    # Generate sample report
    root_causes = [
        ('db_query_latency', 0.87, {
            'outflow_score': 0.8,
            'inflow_score': 0.9,
            'temporal_score': 1.0,
            'severity_score': 0.91,
            'downstream_metrics': ['db_connections', 'api_latency'],
            'upstream_metrics': []
        })
    ]
    
    anomalies = {
        'db_query_latency': 0.91,
        'db_connections': 0.89,
        'api_latency': 0.96,
        'error_rate': 0.98
    }
    
    report_gen = RCAReportGenerator()
    text_report = report_gen.generate_text_report(
        root_causes,
        anomalies,
        nx.DiGraph(),
        failure_duration_minutes=45,
        estimated_impact="Critical"
    )
    
    print(text_report)
```

---

## 4. COMPLETE IMPLEMENTATION CHECKLIST

### Weeks 1-4: Foundation
- [ ] Set up Python environment & dependencies
- [ ] Create project structure
- [ ] Implement synthetic data generator
- [ ] Build data preprocessing pipeline
- [ ] Create unit tests for data generation

### Weeks 5-7: Deep Learning
- [ ] Implement LSTM autoencoder architecture
- [ ] Train on synthetic normal data
- [ ] Evaluate anomaly detection on synthetic failures
- [ ] Implement Transformer-based alternative (optional)
- [ ] Achieve >80% recall on synthetic data

### Weeks 8-10: Causal Inference
- [ ] Implement Granger causality tests
- [ ] Implement PC algorithm
- [ ] Build causal graph visualization
- [ ] Test on synthetic causal data
- [ ] Validate temporal precedence detection

### Weeks 11-12: Root Cause Ranking
- [ ] Implement RootCauseScorer
- [ ] Implement CausalChainTracer
- [ ] Test scoring on case studies
- [ ] Evaluate top-k accuracy
- [ ] Achieve >60% top-1 accuracy

### Weeks 13-14: Reporting & Integration
- [ ] Implement report generators
- [ ] Build Streamlit dashboard
- [ ] Integrate with Prometheus/CloudWatch
- [ ] Implement interactive visualization
- [ ] Write comprehensive documentation

### Weeks 15-16: Evaluation & Polish
- [ ] Test on all 5 case studies
- [ ] Evaluate on public benchmarks (NASA, Azure)
- [ ] Write evaluation report
- [ ] Create deployment documentation
- [ ] Package for release (Docker)

---

## 5. KEY TECHNICAL CHALLENGES & SOLUTIONS

### Challenge 1: Handling Missing/Irregular Data

```python
class RobustDataPreprocessor:
    """Handle real-world data quality issues"""
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Forward fill for short gaps (<5 min)
        Linear interpolation for medium gaps (5-60 min)
        Flag long gaps (>60 min) for manual review
        """
        data_clean = data.copy()
        
        for col in data_clean.columns:
            if col == 'timestamp':
                continue
            
            missing_mask = data_clean[col].isna()
            if missing_mask.sum() == 0:
                continue
            
            # Find consecutive missing value groups
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()
            
            for group_id in missing_groups[missing_mask].unique():
                group_mask = missing_groups == group_id
                group_size = group_mask.sum()
                
                if group_size <= 1:  # Single missing value
                    data_clean.loc[group_mask, col] = data_clean.loc[~group_mask, col].interpolate()
                elif group_size <= 12:  # Short gap (1 hour)
                    # Forward fill
                    data_clean.loc[group_mask, col] = data_clean.loc[~group_mask, col].ffill()
                elif group_size <= 288:  # Medium gap (24 hours)
                    # Linear interpolation
                    indices = np.where(group_mask)[0]
                    data_clean.loc[group_mask, col] = np.interp(
                        indices,
                        np.where(~group_mask)[0],
                        data_clean.loc[~group_mask, col]
                    )
                else:  # Long gap - flag for review
                    print(f"WARNING: Long missing period in {col}: {group_size} samples")
        
        return data_clean
```

---

### Challenge 2: Scaling to High Dimensionality

```python
class ScalableAnomalyDetection:
    """Handle hundreds of metrics efficiently"""
    
    @staticmethod
    def select_important_metrics(
        data: pd.DataFrame,
        method: str = 'variance'
    ) -> List[str]:
        """
        Reduce dimensionality by selecting important metrics
        Methods:
        - variance: High variance = more informative
        - correlation: Remove highly correlated metrics
        - mutual_information: Information-theoretic importance
        """
        
        if method == 'variance':
            variances = data.var()
            # Keep top 20 metrics by variance
            return variances.nlargest(20).index.tolist()
        
        elif method == 'correlation':
            corr_matrix = data.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [col for col in upper_triangle.columns 
                      if any(upper_triangle[col] > 0.95)]
            return [col for col in data.columns if col not in to_drop]
        
        return data.columns.tolist()
```

---

### Challenge 3: Handling Concept Drift

```python
class ConceptDriftDetector:
    """Detect when normal behavior changes"""
    
    @staticmethod
    def detect_drift(
        historical_normal: np.ndarray,
        recent_data: np.ndarray,
        window_size: int = 100
    ) -> bool:
        """
        Use Kullback-Leibler divergence to detect distribution shifts
        If recent data distribution significantly different from historical,
        retrain the model
        """
        from scipy.stats import entropy
        
        # Compute KL divergence
        hist_mean = np.mean(historical_normal)
        hist_std = np.std(historical_normal)
        
        recent_mean = np.mean(recent_data[-window_size:])
        recent_std = np.std(recent_data[-window_size:])
        
        # Simplified KL divergence (assumes Gaussian)
        kl_div = 0.5 * (
            np.log(recent_std / hist_std) +
            (hist_std**2 + (hist_mean - recent_mean)**2) / (2 * recent_std**2) - 0.5
        )
        
        # If KL divergence > threshold, concept drift detected
        return kl_div > 0.5
```

---

## 6. EVALUATION STRATEGY

### Benchmark Datasets

1. **NASA Dataset**: Satellite telemetry with known failures
2. **Azure Dataset**: Real cloud platform failures
3. **Yahoo Dataset**: Time series with injected anomalies

### Metrics

```python
class RCAEvaluator:
    """Comprehensive evaluation"""
    
    @staticmethod
    def evaluate(predictions, ground_truth):
        """
        Metrics:
        1. Top-1 Accuracy: Is #1 prediction the true root cause?
        2. Top-3 Accuracy: Is true root cause in top 3?
        3. Mean Reciprocal Rank: Average rank of true cause
        4. Detection Latency: Time from failure to detection
        5. False Positive Rate: False alarms
        """
        
        top_1_correct = predictions[0] == ground_truth
        top_3_correct = ground_truth in predictions[:3]
        rank = next((i+1 for i, p in enumerate(predictions) 
                    if p == ground_truth), len(predictions)+1)
        mrr = 1.0 / rank
        
        return {
            'top_1_accuracy': float(top_1_correct),
            'top_3_accuracy': float(top_3_correct),
            'mean_reciprocal_rank': mrr,
            'rank_of_true_cause': rank
        }
```

---

## 7. DEPLOYMENT & PRODUCTION CONSIDERATIONS

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rca-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rca-system
  template:
    metadata:
      labels:
        app: rca-system
    spec:
      containers:
      - name: rca-system
        image: rca-system:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MODEL_PATH
          value: "/models/best_model.pt"
```

---

## 8. FINAL RECOMMENDATIONS

### For MVP (12 weeks):
- Focus on 5-10 anomaly types
- Synthetic data only
- Basic CLI interface
- 70%+ recall, 60%+ top-1 accuracy

### For Production (20+ weeks):
- All 50 anomaly types
- Real cloud platform integration
- Web dashboard + API
- 85%+ recall, 70%+ top-1 accuracy
- Deployment in Kubernetes

### Research Extensions:
- Novel causal discovery algorithm
- Multi-modal learning (logs + metrics)
- Explainability via SHAP values
- Transfer learning across platforms

This roadmap provides a complete path from concept to production. The project is highly feasible and has strong potential for academic publication and industry adoption!