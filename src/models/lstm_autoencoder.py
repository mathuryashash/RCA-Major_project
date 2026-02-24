import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for multivariate time-series anomaly detection.
    Trained ONLY on normal data to learn system baselines.
    """
    
    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 2, 
                 latent_size: int = 32, dropout: float = 0.2):
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
        
        # Decoder: reconstruct sequence from latent representation
        self.latent_to_decoder = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_size, n_features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = self.encoder_to_latent(hidden[-1])  # (batch, latent_size)
        
        # Decode: Repeat latent vector for sequence length
        decoder_input = self.latent_to_decoder(latent)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        decoder_out, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction, latent
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates per-metric reconstruction errors (MSE across the sequence).
        Returns a tensor of shape (batch_size, n_features).
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            errors = (x - reconstruction) ** 2
            metric_scores = errors.mean(dim=1)  # Mean across time
        return metric_scores


class AnomalyDetector:
    """
    Wrapper for LSTMAutoencoder: handles training on healthy data, 
    threshold calibration via validation set, and anomaly detection.
    """
    
    def __init__(self, n_features: int, window_size: int = 60, device: str = 'cpu'):
        self.window_size = window_size
        self.device = device
        self.model = LSTMAutoencoder(n_features).to(device)
        self.threshold_per_metric = None
        
    def create_windows(self, values: np.ndarray, stride: int = 1) -> torch.Tensor:
        """Create sliding windows of shape (num_windows, window_size, n_features)"""
        values = values.astype(np.float32)
        windows = []
        for i in range(0, len(values) - self.window_size + 1, stride):
            windows.append(values[i:i + self.window_size])
        return torch.tensor(np.array(windows))
        
    def train(self, normal_array: np.ndarray, epochs: int = 20, lr: float = 1e-3, 
              val_split: float = 0.2, batch_size: int = 32):
        """Standardized training using early stopping and threshold calibration."""
        windows = self.create_windows(normal_array, stride=5)
        
        split = int(len(windows) * (1 - val_split))
        train_data = windows[:split]
        val_data = windows[split:]
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            val_loss = self._validate(val_data, criterion)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_autoencoder_model.pt')
                
        # Load best model for calibration
        self.model.load_state_dict(torch.load('best_autoencoder_model.pt'))
        self._calibrate_thresholds(val_data)
        
    def _validate(self, val_data: torch.Tensor, criterion: nn.Module) -> float:
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            recon, _ = self.model(val_data)
            return criterion(recon, val_data).item()
            
    def _calibrate_thresholds(self, val_data: torch.Tensor, percentile: int = 99):
        """Set detection thresholds per-feature using validation error percentiles."""
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            scores = self.model.get_anomaly_scores(val_data).cpu().numpy()
            self.threshold_per_metric = np.percentile(scores, percentile, axis=0)
            
    def detect(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Runs anomaly detection on a new dataframe.
        Returns a dataframe mapping the anomaly score and boolean flag for each feature back to the original timestamps.
        """
        if self.threshold_per_metric is None:
            raise ValueError("Model requires training or threshold calibration before detection.")
            
        data_vals = df[feature_columns].values
        windows = self.create_windows(data_vals, stride=1).to(self.device)
        
        scores = self.model.get_anomaly_scores(windows).cpu().numpy()
        
        # Normalize: > 1.0 means anomalous
        normalized = scores / (self.threshold_per_metric + 1e-8)
        
        # The first window looks backwards; it produces a single score output at window_size-1.
        result_idx = df.index[self.window_size - 1:]
        
        results = {}
        for i, col in enumerate(feature_columns):
            results[f"{col}_score"] = normalized[:len(result_idx), i]
            results[f"{col}_is_anomaly"] = scores[:len(result_idx), i] > self.threshold_per_metric[i]
            
        res_df = pd.DataFrame(results, index=result_idx)
        return res_df

if __name__ == "__main__":
    # Test harness
    print("Testing LSTM Autoencoder Model Initialization...")
    model = AnomalyDetector(n_features=10)
    print("Success. Run synthetic_generator.py -> training pipeline for further usage.")
