import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """
    Module 3: LSTM Autoencoder for time-series anomaly detection.
    PRD spec: 2-layer encoder (128→64 units) with symmetric decoder (64→128).
    """

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder: 2-layer stacked LSTM (128 → 64)
        self.encoder_layer1 = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.encoder_layer2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.encoder_dropout = nn.Dropout(dropout)

        # Decoder: symmetric 2-layer LSTM (64 → 128)
        self.decoder_layer1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.decoder_layer2 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.decoder_dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(128, num_features)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)

        # Encode: layer1 (features → 128), dropout, layer2 (128 → 64)
        enc1_out, _ = self.encoder_layer1(x)
        enc1_out = self.encoder_dropout(enc1_out)
        _, (hidden, cell) = self.encoder_layer2(enc1_out)

        # Decode: repeat bottleneck hidden state across the sequence
        seq_len = x.shape[1]
        bottleneck = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, 64)

        dec1_out, _ = self.decoder_layer1(bottleneck)
        dec1_out = self.decoder_dropout(dec1_out)
        dec2_out, _ = self.decoder_layer2(dec1_out)

        # Output mapping: 128 → num_features
        reconstruction = self.output_layer(dec2_out)
        return reconstruction


class MetricDeepAnomalyDetector:
    def __init__(self, num_features, sequence_length=60, hidden_size=64):
        self.sequence_length = sequence_length
        self.model = LSTMAutoencoder(num_features=num_features, hidden_size=hidden_size)
        self.criterion = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.thresholds = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_sequences(self, data):
        """
        Converts 2D array (timesteps, features) into 3D sequences (samples, seq_len, features)
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences)

    def train(self, df_healthy: pd.DataFrame, epochs=10, batch_size=32):
        """
        Trains the Autoencoder on healthy data to establish baseline reconstruction errors.
        """
        self.model.train()
        data = df_healthy.values
        sequences = self.create_sequences(data)

        tensor_x = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(tensor_x, tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                self.optimizer.zero_grad()
                reconstruction = self.model(batch_x)
                loss = self.criterion(reconstruction, batch_x).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
            )

        # Establish thresholds (99th percentile of reconstruction error on training data)
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(tensor_x)
            errors = self.criterion(reconstruction, tensor_x).mean(dim=1).cpu().numpy()
            self.thresholds = np.percentile(errors, 99, axis=0)

    def score(self, df_current: pd.DataFrame):
        """
        Returns anomaly scores mapped to [0,1] based on established thresholds.
        """
        if self.thresholds is None:
            raise ValueError("Model must be trained before scoring.")

        self.model.eval()
        data = df_current.values
        sequences = self.create_sequences(data)

        if len(sequences) == 0:
            return np.zeros(data.shape[1])

        tensor_x = torch.FloatTensor(sequences).to(self.device)
        with torch.no_grad():
            reconstruction = self.model(tensor_x)
            # Get errors for the latest sequence
            errors = self.criterion(reconstruction, tensor_x).mean(dim=1).cpu().numpy()

            # Use the error of the last sequence for real-time scoring
            latest_error = errors[-1]

            # Map to [0,1] using a sigmoid-like function around the threshold
            # If error == threshold, score = 0.5
            scores = 1 / (1 + np.exp(-10 * (latest_error / self.thresholds - 1)))
            return scores


if __name__ == "__main__":
    print("Testing LSTM Autoencoder...")
    # Generate dummy healthy data
    t = np.linspace(0, 100, 500)
    healthy_data = pd.DataFrame(
        {
            "cpu": np.sin(t) + np.random.normal(0, 0.1, 500),
            "memory": np.cos(t) + np.random.normal(0, 0.1, 500),
        }
    )

    detector = MetricDeepAnomalyDetector(num_features=2, sequence_length=30)
    print("Training...")
    detector.train(healthy_data, epochs=5)

    # Generate anomaly data
    anomaly_data = healthy_data.copy().tail(60).reset_index(drop=True)
    anomaly_data.loc[40:, "cpu"] += 5.0  # Add a massive spike

    print("\nScoring Anomaly Window...")
    scores = detector.score(anomaly_data)
    print(f"Anomaly Scores [cpu, memory]: {scores}")
