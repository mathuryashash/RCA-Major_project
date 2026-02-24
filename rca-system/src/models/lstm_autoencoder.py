"""
LSTM Autoencoder for Time-Series Anomaly Detection

Architecture:
- Encoder: 2-layer LSTM that compresses the input sequence into a latent vector
- Decoder: 2-layer LSTM that reconstructs the sequence from the latent vector
- Anomaly Score: Reconstruction error (MSE) — high error = anomaly

Training strategy:
- Train ONLY on normal (healthy) data
- The model learns to reconstruct normal patterns well
- Anomalies produce high reconstruction error because the model has
  never seen them during training

Reference:
    Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly
    Detection", ICML 2016 Anomaly Detection Workshop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict, List, Optional
import os


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for multivariate time-series anomaly detection.

    The model is trained on normal data only (unsupervised). High reconstruction
    error during inference signals deviation from learned normal patterns.

    Args:
        input_size: Number of metrics/features per timestep
        sequence_length: Length of each input window
        hidden_size: LSTM hidden state dimension
        latent_size: Bottleneck dimension
        num_layers: Number of LSTM layers in encoder and decoder
        dropout: Dropout probability between LSTM layers
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # ---- Encoder ----
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Project LSTM hidden state to compact latent vector
        self.encoder_to_latent = nn.Linear(hidden_size, latent_size)

        # ---- Decoder ----
        # Expand latent vector back to LSTM hidden size
        self.latent_to_decoder = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Project back to original input dimension
        self.output_layer = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            latent: (batch_size, latent_size)
        """
        _, (hidden, _) = self.encoder_lstm(x)
        # Use the final layer's hidden state
        latent = self.encoder_to_latent(hidden[-1])
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to sequence.

        Args:
            latent: (batch_size, latent_size)

        Returns:
            reconstruction: (batch_size, sequence_length, input_size)
        """
        # Expand latent to hidden size, then repeat for each timestep
        decoder_input = self.latent_to_decoder(latent)           # (B, H)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.sequence_length, 1)  # (B, T, H)

        decoder_out, _ = self.decoder_lstm(decoder_input)        # (B, T, H)
        reconstruction = self.output_layer(decoder_out)           # (B, T, F)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Returns:
            reconstruction: (batch_size, sequence_length, input_size)
            latent: (batch_size, latent_size)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction MSE.

        Args:
            x: (batch_size, sequence_length, input_size)
            reduction: 'none' -> per-sample, 'mean' -> scalar, 'feature' -> per-feature

        Returns:
            Tensor of reconstruction errors
        """
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            squared_err = (x - reconstruction) ** 2  # (B, T, F)

            if reduction == 'none':
                # Mean over time and features -> per-sample scalar
                return squared_err.mean(dim=[1, 2])
            elif reduction == 'feature':
                # Mean over time -> per-feature error for each sample
                return squared_err.mean(dim=1)  # (B, F)
            else:  # 'mean'
                return squared_err.mean()


class AnomalyDetectionTrainer:
    """
    Manages training of LSTMAutoencoder on normal data, threshold calibration,
    and anomaly scoring during inference.

    Usage:
        trainer = AnomalyDetectionTrainer(model)
        trainer.fit(train_windows, val_windows, epochs=50)
        trainer.calibrate_thresholds(val_windows)
        scores = trainer.get_anomaly_scores(test_windows)
        flags  = trainer.detect_anomalies(test_windows)
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        device: Optional[str] = None,
        lr: float = 1e-3
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

        # Calibration thresholds (set after calling calibrate_thresholds)
        self.threshold_per_feature: Optional[np.ndarray] = None
        self.threshold_global: Optional[float] = None

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder on normal data windows.

        Args:
            train_windows: (N_train, window_size, n_features) float32 numpy array
            val_windows:   (N_val,   window_size, n_features) float32 numpy array
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            patience: Early stopping patience
            save_path: If provided, save best model weights here

        Returns:
            Training history dict with 'train_loss' and 'val_loss'
        """
        train_tensor = torch.from_numpy(train_windows)
        val_tensor = torch.from_numpy(val_windows)

        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # ---- Training ----
            self.model.train()
            epoch_train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                reconstruction, _ = self.model(batch)
                loss = self.criterion(reconstruction, batch)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # ---- Validation ----
            val_loss = self._validate(val_tensor)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:>4}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            # ---- Early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        # Load best weights if saved
        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"Best model loaded from {save_path}")

        return self.history

    def calibrate_thresholds(
        self,
        val_windows: np.ndarray,
        percentile: float = 99.0
    ):
        """
        Set anomaly detection thresholds from validation (normal) data.

        Strategy: Threshold = Nth percentile of reconstruction errors on
        normal validation data. ~1% of normal data will exceed threshold.

        Args:
            val_windows: Normal validation windows (N, T, F)
            percentile: Percentile for threshold (default: 99 -> 1% false positive rate)
        """
        val_tensor = torch.from_numpy(val_windows).to(self.device)
        self.model.eval()

        with torch.no_grad():
            # Per-feature errors: (N, F)
            feature_errors = self.model.get_reconstruction_error(
                val_tensor, reduction='feature'
            ).cpu().numpy()

            # Per-sample errors: (N,)
            sample_errors = feature_errors.mean(axis=1)

        self.threshold_per_feature = np.percentile(feature_errors, percentile, axis=0)
        self.threshold_global = float(np.percentile(sample_errors, percentile))

        print(
            f"Thresholds calibrated at {percentile}th percentile:\n"
            f"  Global threshold: {self.threshold_global:.6f}\n"
            f"  Per-feature thresholds: {self.threshold_per_feature.round(6)}"
        )

    def get_anomaly_scores(
        self,
        windows: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores for input windows.

        Args:
            windows: (N, window_size, n_features)

        Returns:
            dict with keys:
            'sample_scores': (N,)  — overall score per window (0=normal, 1=calibrated threshold)
            'feature_scores': (N, F) — per-feature scores
            'raw_errors': (N,) — raw MSE values
        """
        self.model.eval()
        tensor = torch.from_numpy(windows).to(self.device)

        with torch.no_grad():
            feature_errors = self.model.get_reconstruction_error(
                tensor, reduction='feature'
            ).cpu().numpy()  # (N, F)

        sample_errors = feature_errors.mean(axis=1)  # (N,)

        if self.threshold_global is not None:
            sample_scores = sample_errors / (self.threshold_global + 1e-8)
            feature_scores = feature_errors / (self.threshold_per_feature + 1e-8)
        else:
            # Not calibrated: return raw errors normalized by their max
            sample_scores = sample_errors / (sample_errors.max() + 1e-8)
            feature_scores = feature_errors / (feature_errors.max(axis=0) + 1e-8)

        return {
            'sample_scores': sample_scores.astype(np.float32),
            'feature_scores': feature_scores.astype(np.float32),
            'raw_errors': sample_errors.astype(np.float32)
        }

    def detect_anomalies(
        self,
        windows: np.ndarray,
        score_threshold: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Binary anomaly flags for input windows.

        Args:
            windows: (N, T, F)
            score_threshold: A sample is anomalous when normalized score > this.
                             Default 1.0 = exactly the calibrated threshold.
                             Lower = more sensitive, Higher = less sensitive.

        Returns:
            dict with:
            'is_anomaly': (N,) bool array
            'anomalous_features': (N, F) bool array — which features are anomalous
            'scores': dict from get_anomaly_scores()
        """
        scores = self.get_anomaly_scores(windows)

        is_anomaly = scores['sample_scores'] > score_threshold
        anomalous_features = scores['feature_scores'] > score_threshold

        return {
            'is_anomaly': is_anomaly,
            'anomalous_features': anomalous_features,
            'scores': scores
        }

    def _validate(self, val_tensor: torch.Tensor) -> float:
        """Compute validation loss."""
        self.model.eval()
        with torch.no_grad():
            val_tensor = val_tensor.to(self.device)
            reconstruction, _ = self.model(val_tensor)
            loss = self.criterion(reconstruction, val_tensor)
        return loss.item()

    def save(self, path: str):
        """Save model weights and thresholds."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'threshold_global': self.threshold_global,
            'threshold_per_feature': self.threshold_per_feature,
            'history': self.history
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights and thresholds."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.threshold_global = state.get('threshold_global')
        self.threshold_per_feature = state.get('threshold_per_feature')
        self.history = state.get('history', {'train_loss': [], 'val_loss': []})
        print(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== LSTM Autoencoder Smoke Test ===\n")

    # Synthetic data
    T, F = 2000, 10
    normal_data = np.random.randn(T, F).astype(np.float32) * 0.5 + 2.0

    # Create windows manually
    window_size = 60
    windows = np.array([normal_data[i:i+window_size] for i in range(0, T - window_size, 5)],
                       dtype=np.float32)

    split = int(len(windows) * 0.8)
    train_w = windows[:split]
    val_w   = windows[split:]

    # Build model
    model = LSTMAutoencoder(
        input_size=F,
        sequence_length=window_size,
        hidden_size=32,
        latent_size=16,
        num_layers=2
    )

    trainer = AnomalyDetectionTrainer(model, lr=1e-3)
    print(f"Training on {len(train_w)} windows, validating on {len(val_w)} windows")
    trainer.fit(train_w, val_w, epochs=5, batch_size=16)

    trainer.calibrate_thresholds(val_w)

    # Anomaly detection on injected anomaly
    anomaly_data = normal_data.copy()
    anomaly_data[1900:] += 10.0  # Inject anomaly
    anomaly_windows = np.array([
        anomaly_data[i:i+window_size] for i in range(1900, 2000 - window_size, 5)
    ], dtype=np.float32)

    if len(anomaly_windows) > 0:
        results = trainer.detect_anomalies(anomaly_windows)
        print(f"\nAnomaly detection results on injected anomaly:")
        print(f"  Anomaly flags: {results['is_anomaly']}")
        print(f"  Sample scores: {results['scores']['sample_scores']}")
    else:
        print("(No anomaly windows generated — dataset too small for smoke test)")
