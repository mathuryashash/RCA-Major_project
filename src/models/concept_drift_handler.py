import logging
import pandas as pd
import torch
import os
from typing import Optional
from datetime import datetime, timedelta

from anomaly_detection.anomaly_scorer import AnomalyDetector
from data_ingestion.prometheus_connector import PrometheusDataIngestion
from causal_inference.deployment_listener import DeploymentEventListener

class ConceptDriftHandler:
    """
    Automated Baseline Retraining / Fine-Tuning Pipeline.
    
    In production (PRD Phase 4), deployments change the "normal" metric baseline.
    If the system doesn't learn this "concept drift", it will generate endless false positives.
    
    This module:
    1. Listens for recent deployments.
    2. Waits for a "soak period" (e.g. 2 hours) to ensure the deployment didn't immediately fail.
    3. Fetches new post-deployment data.
    4. Fine-tunes the LSTM Autoencoder so the new metrics become the "new normal".
    """

    def __init__(
        self, 
        model_path: str = "best_autoencoder_model.pt",
        soak_period_hours: float = 2.0,
        fine_tune_epochs: int = 5
    ):
        self.model_path = model_path
        self.soak_period_hours = soak_period_hours
        self.fine_tune_epochs = fine_tune_epochs
        
        self.prom_client = PrometheusDataIngestion()
        self.event_listener = DeploymentEventListener() # Just used to read the log here
        
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def trigger_retraining_if_needed(self, feature_count: int) -> bool:
        """
        Checks if a deployment recently finished its soak period.
        If yes, performs fine-tuning.
        Returns True if retraining occurred.
        """
        # 1. Look for recent deployments
        recent_events = self.event_listener.load_events(max_days_old=1)
        if recent_events.empty:
            return False
            
        # We only care about deployments that have 'soaked' long enough to collect a new baseline
        # but haven't been factored in yet.
        now = datetime.utcnow()
        if recent_events['timestamp'].dt.tz is not None:
             now = now.replace(tzinfo=recent_events['timestamp'].dt.tz)
             
        # Find latest deployment
        latest_deployment = recent_events.sort_values(by='timestamp', ascending=False).iloc[0]
        dep_time = latest_deployment['timestamp']
        
        hours_since_deploy = (now - dep_time).total_seconds() / 3600.0
        
        if hours_since_deploy < self.soak_period_hours:
            self.logger.info(f"Recent deployment at {dep_time}. Soaking. ({hours_since_deploy:.1f}/{self.soak_period_hours} hours)")
            return False
            
        # Optional: In a real system you'd keep track of "last_retrain_time" in a DB
        # to ensure we don't retrain repeatedly for the same deployment.
        # For this prototype, we'll assume an external cron job triggers this EXACTLY once 
        # after the soak period.
        
        self.logger.info(f"Deployment soak period finished. Triggering Fine-Tuning for Concept Drift.")
        self._fine_tune_model(dep_time, feature_count)
        return True

    def _fine_tune_model(self, deployment_time: pd.Timestamp, feature_count: int):
        """
        Fetches the last N hours of data (since deployment) and fine-tunes the LSTM.
        """
        if not os.path.exists(self.model_path):
             self.logger.error(f"Cannot fine-tune. Base model not found at {self.model_path}")
             return
             
        self.logger.info('Fetching post-deployment metric data...')
        
        # Calculate how many hours of data we have since the deployment
        now = datetime.utcnow()
        if deployment_time.tz is not None:
             now = now.replace(tzinfo=deployment_time.tz)
             
        hours_post_deploy = (now - deployment_time).total_seconds() / 3600.0
        
        # We don't want to use days=... if it's less than a day
        # In a real environment, we'd pass exact start/end to ingest_prometheus_metrics
        # Here we mock the new data collection
        
        # Mocking the new data retrieval based on the existing Prom connector structure
        # df_new = self.prom_client.ingest_prometheus_metrics(...)
        
        # MOCK DATA FOR DEMONSTRATION
        self.logger.info(f"Simulating fetch of {hours_post_deploy:.1f} hours of new data...")
        import numpy as np
        
        # Create a tiny realistic dataset matching the feature count
        # In a real scenario, you'd apply your DataImputer and Scaler here before training.
        samples = int(hours_post_deploy * 12) # 12 windows per hour (5 min)
        # Shift the mean slightly to represent the "concept drift" of the new deployment
        new_normal_data = np.random.normal(0.6, 0.1, (samples, feature_count))
        
        # Initialize an AnomalyDetector with the same specs
        detector = AnomalyDetector(n_features=feature_count, window_size=12)
        
        # Load existing weights
        detector.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.logger.info("Loaded base model weights.")
        
        # Fine tune
        self.logger.info(f"Fine-tuning LSTM for {self.fine_tune_epochs} epochs on new distribution...")
        detector.train(
             normal_array=new_normal_data,
             epochs=self.fine_tune_epochs, 
             lr=1e-4, # Lower learning rate for fine-tuning
             batch_size=16
        )
        
        # detector.train() automatically saves the new best weights to 'best_autoencoder_model.pt'
        self.logger.info(f"Model successfully fine-tuned and updated to handle concept drift.")

if __name__ == "__main__":
     handler = ConceptDriftHandler(soak_period_hours=0.01) # Set super low for testing
     
     # Write a fake deployment to the log right now
     import json
     fake_event = {
          "timestamp": datetime.utcnow().isoformat() + "Z",
          "description": "Concept Drift Test Deployment", 
          "type": "deployment"
     }
     with open("deployment_events.jsonl", "w") as f:
          f.write(json.dumps(fake_event) + "\n")
          
     print("Wrote fake deployment event just now.")
     
     import time
     print("Waiting a few seconds for soak period...")
     time.sleep(3)
     
     # Create dummy model file to load
     # (Need to init a dummy model and save it so the loader doesn't crash)
     from anomaly_detection.anomaly_scorer import AnomalyDetector
     dummy = AnomalyDetector(n_features=10)
     torch.save(dummy.model.state_dict(), "best_autoencoder_model.pt")
     
     # Trigger retrain
     handler.trigger_retraining_if_needed(feature_count=10)
