import pandas as pd
import numpy as np
import logging
from typing import List, Dict

class AlertDampener:
    """
    Alert Dampening logic to prevent alert fatigue.
    
    Capabilities:
    - Requires an anomaly to persist for N consecutive windows before alerting.
    - Suppresses flapping (metric alternating between anomalous/normal rapidly).
    - Can implement a cool-down period post-alert.
    """

    def __init__(self, required_consecutive_windows: int = 3, cooldown_windows: int = 6):
        """
        Args:
            required_consecutive_windows: Number of times an anomaly must trigger in a row
                                          to be considered a real alert. (e.g. 3 x 5mins = 15 mins)
            cooldown_windows: After triggering a hard alert, ignore subsequent 
                              triggers for this many windows to prevent spam.
        """
        self.req_windows = required_consecutive_windows
        self.cooldown_windows = cooldown_windows
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        
    def dampen_batch(self, df_scores: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Processes a full dataframe of anomaly scores sequentially to dampen alerts.
        Expects columns named `{col}_is_anomaly` (boolean).
        
        Returns a DataFrame identical to input but with updated `{col}_hard_alert` columns.
        """
        self.logger.info(f"Dampening alerts for {len(feature_columns)} metrics over {len(df_scores)} windows.")
        
        # Sort index just in case
        df_sorted = df_scores.sort_index()
        results = pd.DataFrame(index=df_sorted.index)
        
        for col in feature_columns:
            anomaly_col = f"{col}_is_anomaly"
            if anomaly_col not in df_sorted.columns:
                continue
                
            raw_flags = df_sorted[anomaly_col].values.astype(int)
            hard_alerts = np.zeros_like(raw_flags, dtype=bool)
            
            consecutive_count = 0
            cooldown_counter = 0
            
            for i, is_anom in enumerate(raw_flags):
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    # While in cooldown, we don't trigger new alerts but we can reset consecutive count
                    # or keep it paused. Let's just reset it.
                    consecutive_count = 0
                    continue
                    
                if is_anom:
                    consecutive_count += 1
                else:
                    # Reset consecutive count if it goes back to normal
                    consecutive_count = 0
                    
                if consecutive_count >= self.req_windows:
                    hard_alerts[i] = True
                    cooldown_counter = self.cooldown_windows
                    consecutive_count = 0
                    
            results[f"{col}_hard_alert"] = hard_alerts
            
        # Add the results to the original dataframe
        df_out = pd.concat([df_sorted, results], axis=1)
        
        total_raw = df_scores[[f"{c}_is_anomaly" for c in feature_columns if f"{c}_is_anomaly" in df_scores.columns]].sum().sum()
        total_dampened = results.sum().sum()
        self.logger.info(f"Dampening results: {total_raw} raw anomalies reduced to {total_dampened} hard alerts.")
        
        return df_out

if __name__ == "__main__":
    dampener = AlertDampener(required_consecutive_windows=3, cooldown_windows=5)
    
    # Simulate an anomaly column
    # Should trigger at index 4 (3rd consecutive 1)
    # Then cooldown for 5 windows (5..9)
    # Then trigger again at index 12
    is_anomaly = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
    df = pd.DataFrame({"cpu_spik_is_anomaly": is_anomaly})
    
    print("Raw anomalies:")
    print(df["cpu_spik_is_anomaly"].values)
    
    result = dampener.dampen_batch(df, feature_columns=["cpu_spik"])
    
    print("\nHard Alerts (After Dampening):")
    print(result["cpu_spik_hard_alert"].values.astype(int))
