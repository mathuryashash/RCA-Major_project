"""Shared application state — replaces st.session_state from the Streamlit app."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AppState:
    model_trained: bool = False

    normal_df: Optional[pd.DataFrame] = None
    feat_cols: List[str] = field(default_factory=list)
    detector: Any = None  # models.lstm_autoencoder.AnomalyDetector
    scaler: Any = None

    training_epochs: int = 5
    window_size: int = 12

    last_causal_results: Optional[Dict] = None
    last_root_causes: Optional[List[Dict]] = None
    last_incident_scaled: Optional[pd.DataFrame] = None
    last_anomaly_scores: Optional[Dict[str, float]] = None
    last_anomaly_times: Optional[Dict[str, pd.Timestamp]] = None
    last_report: Optional[Dict[str, str]] = None
