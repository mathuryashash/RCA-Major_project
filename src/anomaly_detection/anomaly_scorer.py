import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models.lstm_autoencoder import LSTMAutoencoder

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
        self.n_features = n_features
        
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
            
    def _calibrate_thresholds(self, val_data: torch.Tensor, percentile: int = 95):
        """Set detection thresholds per-feature using validation error percentiles."""
        self.model.eval()
        with torch.no_grad():
            val_data = val_data.to(self.device)
            scores = self.model.get_anomaly_scores(val_data).cpu().numpy()
            self.threshold_per_metric = np.percentile(scores, percentile, axis=0)
            print(f"Calibrated thresholds for {self.n_features} features: {self.threshold_per_metric}")
            
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
    import sys
    import os
    # Add parent dir to path to find synthetic generator if run standalone
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_ingestion.synthetic_generator import SyntheticMetricsGenerator
    
    print("Testing pipeline from Synthetic Data to Autoencoder...")
    gen = SyntheticMetricsGenerator()
    normal_data = gen.generate_normal_behavior(duration_days=30)
    
    # Exclude timestamp
    feat_cols = [c for c in normal_data.columns if c != 'timestamp']
    # Normally we'd use a StandardScaler here in a real pipeline
    
    detector = AnomalyDetector(n_features=len(feat_cols), window_size=12, device='cpu')
    print("Training Anomaly Detector on 30 days of baseline data...")
    detector.train(normal_data[feat_cols].values, epochs=5) # few epochs for test
    print("Thresholds calibrated successfully!")
