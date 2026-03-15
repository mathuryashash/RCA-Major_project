"""
Module 3b: Temporal Transformer for multi-step metric forecasting and anomaly detection.

Catches step-changes and regime shifts that LSTM Autoencoders tend to miss.
Architecture: Transformer encoder with sinusoidal positional encoding,
projecting to a multi-step forecast horizon.

Anomaly score = sigmoid-normalized MAE between forecast and actuals,
calibrated against the 99th percentile of training error.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input.

    Adds fixed sin/cos signals so the transformer can reason about
    ordering within the metric window.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------


class TemporalTransformerModel(nn.Module):
    """Transformer encoder for multi-step metric forecasting.

    Input: (batch, seq_len, n_features)
    Output: (batch, forecast_horizon, n_features)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 10,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Project raw features into d_model space
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Collapse sequence → forecast output
        # We use the last token's representation to predict the future
        self.output_projection = nn.Linear(d_model, n_features * forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, forecast_horizon, n_features)
        """
        # (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Take the last timestep as a summary representation
        x = x[:, -1, :]  # (batch, d_model)

        # Project to forecast
        out = self.output_projection(x)  # (batch, n_features * horizon)
        out = out.view(-1, self.forecast_horizon, self.n_features)
        return out


# ---------------------------------------------------------------------------
# Detector (train + score wrapper)
# ---------------------------------------------------------------------------


class TemporalTransformerDetector:
    """Wrapper for training and anomaly scoring with Temporal Transformer.

    Mirrors the MetricDeepAnomalyDetector API so it can be used as a
    drop-in secondary signal alongside the LSTM Autoencoder.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        forecast_horizon: int = 10,
        sequence_length: int = 60,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalTransformerModel(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        self._threshold: Optional[np.ndarray] = None  # 99th percentile MAE per metric

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def create_forecast_pairs(self, data: np.ndarray) -> tuple:
        """Create (input_window, target_forecast) pairs from continuous data.

        From a sequence of length L, creates pairs where:
        - input:  data[i : i + seq_len]
        - target: data[i + seq_len : i + seq_len + horizon]

        Args:
            data: (L, n_features) array of metric values.

        Returns:
            (inputs, targets) — numpy arrays of shape
            (N, seq_len, n_features) and (N, horizon, n_features).
        """
        required = self.sequence_length + self.forecast_horizon
        if len(data) < required:
            raise ValueError(
                f"Data length {len(data)} is less than "
                f"sequence_length ({self.sequence_length}) + "
                f"forecast_horizon ({self.forecast_horizon}) = {required}"
            )

        inputs, targets = [], []
        for i in range(len(data) - required + 1):
            inputs.append(data[i : i + self.sequence_length])
            targets.append(
                data[
                    i + self.sequence_length : i
                    + self.sequence_length
                    + self.forecast_horizon
                ]
            )
        return np.array(inputs), np.array(targets)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df_healthy: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> dict:
        """Train on healthy metric data.

        Args:
            df_healthy: DataFrame with numeric metric columns. Length must be
                        >= sequence_length + forecast_horizon.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam LR.

        Returns:
            dict with 'epoch_losses' list and 'final_loss' float.
        """
        data = df_healthy.values.astype(np.float32)
        inputs, targets = self.create_forecast_pairs(data)

        t_inputs = torch.from_numpy(inputs).to(self.device)
        t_targets = torch.from_numpy(targets).to(self.device)

        dataset = TensorDataset(t_inputs, t_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        epoch_losses = []
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        # Compute 99th-percentile MAE per metric on training data
        self.model.eval()
        with torch.no_grad():
            all_preds = self.model(t_inputs)  # (N, horizon, features)
            mae = torch.abs(all_preds - t_targets).mean(dim=1)  # (N, features)
            mae_np = mae.cpu().numpy()
            self._threshold = np.percentile(mae_np, 99, axis=0)  # (features,)
            # Avoid zero thresholds (constant metrics)
            self._threshold = np.maximum(self._threshold, 1e-8)

        return {
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, df_current: pd.DataFrame) -> np.ndarray:
        """Score metric windows. Returns anomaly score [0, 1] per metric.

        Uses MAE between forecast and actual for the **final** window.
        Normalises via sigmoid around training error 99th percentile.

        Args:
            df_current: DataFrame with the same columns used for training.
                        Must have length >= sequence_length + forecast_horizon.

        Returns:
            1-D numpy array of shape (n_features,) with scores in [0, 1].
        """
        if self._threshold is None:
            raise ValueError("Model must be trained before scoring.")

        data = df_current.values.astype(np.float32)
        required = self.sequence_length + self.forecast_horizon
        if len(data) < required:
            return np.zeros(self.n_features)

        # Use the final window
        input_window = data[
            -(self.sequence_length + self.forecast_horizon) : -self.forecast_horizon
        ]
        actual_future = data[-self.forecast_horizon :]

        t_input = (
            torch.from_numpy(input_window).unsqueeze(0).to(self.device)
        )  # (1, seq, feat)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(t_input)  # (1, horizon, feat)
            pred_np = pred.squeeze(0).cpu().numpy()  # (horizon, feat)

        mae_per_metric = np.abs(pred_np - actual_future).mean(axis=0)  # (features,)

        # Sigmoid normalisation: score = sigmoid(k * (mae / threshold - 1))
        # At mae == threshold → score ≈ 0.5
        scores = 1.0 / (1.0 + np.exp(-10.0 * (mae_per_metric / self._threshold - 1.0)))
        return scores

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True when the model has been trained and thresholds computed."""
        return self._threshold is not None
