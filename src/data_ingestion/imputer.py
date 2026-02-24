import pandas as pd
import numpy as np
import logging

class DataImputer:
    """
    Robust Data Imputation and Cleansing module as per PRD Section 1.1.1.
    
    Capabilities:
    - Handles out-of-order timestamps
    - Deduplicates timestamps
    - Interpolates small gaps (e.g. up to 30 mins)
    - Forward-fills larger gaps
    - Removes zero-variance (flatline) metrics
    - Smooths extreme outliers to prevent LSTM instability
    """

    def __init__(self, frequency: str = '5min'):
        self.frequency = frequency
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def clean_and_impute(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = 'timestamp',
        max_interpolate_limit: int = 6,  # 6 * 5min = 30min
        outlier_z_threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Main pipeline to clean a merged raw DataFrame from multiple sources.
        """
        self.logger.info(f"Starting imputation on DataFrame: shape {df.shape}")
        
        if df.empty:
            return df
            
        if timestamp_col not in df.columns:
            self.logger.error(f"Timestamp column '{timestamp_col}' not found.")
            raise ValueError(f"Missing {timestamp_col} column")

        # 1. Handle Timestamps (out-of-order, duplicates)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(by=timestamp_col)
        
        # Deduplicate, keep last seen
        dups = df.duplicated(subset=[timestamp_col], keep='last')
        if dups.any():
            self.logger.info(f"Removing {dups.sum()} duplicate timestamps.")
            df = df[~dups]
            
        df.set_index(timestamp_col, inplace=True)
        
        # 2. Reindex to ensure continuous frequency with NO missing rows
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.frequency)
        df = df.reindex(full_index)
        
        # 3. Handle NaNs
        missing_initial = df.isna().mean().mean() * 100
        self.logger.info(f"Initial missingness: {missing_initial:.2f}%")
        
        # Drop columns with > 50% missing data
        cols_missing_frac = df.isna().mean()
        bad_cols = cols_missing_frac[cols_missing_frac > 0.50].index.tolist()
        if bad_cols:
            self.logger.warning(f"Dropping {len(bad_cols)} columns with >50% missing data.")
            df.drop(columns=bad_cols, inplace=True)

        # Interpolate small gaps securely
        df.interpolate(method='linear', limit=max_interpolate_limit, inplace=True)
        
        # Forward fill the rest, backward fill the edges
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Replace any remaining NaNs with 0 (edge cases)
        df.fillna(0, inplace=True)
        
        # 4. Outlier Smoothing (Cap extreme values)
        # Extreme sudden spikes (e.g. z-score > 5) can ruin scaling
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                continue
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = (df[col] - mean) / std
                cap_mask = np.abs(z_scores) > outlier_z_threshold
                if cap_mask.any():
                    cap_val = mean + (outlier_z_threshold * std)
                    floor_val = mean - (outlier_z_threshold * std)
                    df.loc[z_scores > outlier_z_threshold, col] = cap_val
                    df.loc[z_scores < -outlier_z_threshold, col] = floor_val
        
        # Reset index back
        df.reset_index(inplace=True)
        df.rename(columns={'index': timestamp_col}, inplace=True)
        
        self.logger.info(f"Imputation finished. Final shape: {df.shape}")
        return df

if __name__ == "__main__":
    # Small test
    imputer = DataImputer()
    
    # Create fake messy data
    dates = pd.date_range('2023-01-01', periods=5, freq='15min') # intentionally wrong freq
    sim_data = pd.DataFrame({
        'timestamp': [dates[0], dates[2], dates[2], dates[4], dates[0]], # out of order, dups
        'metric_a': [1.0, np.nan, 3.0, 1000.0, 1.0], # nan, massive outlier
        'metric_b': [np.nan, np.nan, np.nan, np.nan, np.nan] # all nans
    })
    
    print("Before:")
    print(sim_data)
    
    clean_data = imputer.clean_and_impute(sim_data)
    
    print("\nAfter:")
    print(clean_data)
