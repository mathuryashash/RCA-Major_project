# AI-Powered Root Cause Analysis (RCA) System

An intelligent system that automatically diagnoses root causes of production failures in complex distributed systems by analyzing system metrics, logs, and dependencies through advanced machine learning, statistical causal inference, and graph-based reasoning.

## 4-Stage Pipeline

1. **Normal Behavior Learning**: LSTM autoencoders trained on healthy operational data
2. **Anomaly Detection**: Identifies deviations from normal patterns
3. **Causal Inference**: Uses Granger causality and graph algorithms to determine true causes
4. **Root Cause Ranking**: Scores and ranks candidates with confidence percentages

## Project Structure

```
rca-system/
├── data/
│   ├── raw/                 # Raw metrics from monitoring
│   ├── synthetic/           # Generated failure scenarios
│   └── benchmarks/          # NASA, Azure datasets
├── src/
│   ├── data_ingestion/      # Data ingestion modules
│   ├── preprocessing/       # Time series utilities
│   ├── models/              # LSTM autoencoder models
│   ├── anomaly_detection/   # Anomaly detection engine
│   ├── causal_inference/    # Granger causality, PC algorithm
│   ├── root_cause_ranking/  # Scoring and ranking
│   └── reporting/           # Report generation, visualization
├── tests/                   # Unit and integration tests
├── notebooks/               # Exploration and experiments
├── config/                  # Configuration files
├── docker/                  # Docker setup
└── docs/                    # Project documentation
```

## Setup

```bash
# Create environment
conda create -n rca python=3.10
conda activate rca

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_ingestion.synthetic_generator import SyntheticMetricsGenerator
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.anomaly_detection.anomaly_scorer import AnomalyDetector
from src.causal_inference.granger_causality import CausalInferenceEngine
from src.root_cause_ranking.scorer import RootCauseRanker

# Generate synthetic data
generator = SyntheticMetricsGenerator()
normal_data = generator.generate_normal_behavior(duration_days=60)
failure_data, metadata = generator.inject_failure_scenario(
    normal_data, failure_type='database_slow_query', start_idx=10000, duration_samples=200
)

# Run RCA pipeline
# (see notebooks/ for full end-to-end examples)
```
