import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class DimensionalityReducer:
    """
    Dimensionality Reduction Pipeline for handling vast numbers of production metrics.
    
    Capabilities:
    1. Flatline Filtering: Removes metrics with near-zero variance.
    2. Correlation Grouping: Clusters highly correlated metrics and selects a representative 
       to prevent the causal inference engine from being overwhelmed by redundant data.
    """

    def __init__(
        self, 
        variance_threshold: float = 1e-5, 
        correlation_threshold: float = 0.95
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def filter_low_variance(
        self, 
        df: pd.DataFrame, 
        exclude_cols: List[str] = ['timestamp']
    ) -> pd.DataFrame:
        """
        Removes flatline metrics (e.g., constant 0 values).
        """
        self.logger.info(f"Checking for low variance metrics. Initial columns: {df.shape[1]}")
        cols_to_check = [c for c in df.columns if c not in exclude_cols]
        
        variances = df[cols_to_check].var()
        low_var_cols = variances[variances < self.variance_threshold].index.tolist()
        
        if low_var_cols:
            self.logger.info(f"Removing {len(low_var_cols)} low variance metrics.")
            df = df.drop(columns=low_var_cols)
            
        return df

    def group_correlated_metrics(
        self, 
        df: pd.DataFrame, 
        exclude_cols: List[str] = ['timestamp']
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Finds highly correlated groups of metrics and keeps only one representative 
        metric per group.
        
        Returns:
            Reduced DataFrame
            Dictionary mapping the representative metric to its group members
        """
        cols = [c for c in df.columns if c not in exclude_cols]
        if len(cols) < 2:
            return df, {c: [c] for c in cols}
            
        self.logger.info(f"Computing correlation matrix for {len(cols)} metrics...")
        
        # Calculate Spearman correlation (handles non-linear monotonic relationships well)
        corr_matrix = df[cols].corr(method='spearman').abs()
        
        # Fill NaNs with 0 (which might happen if variance was exactly 0 in some sub-window)
        corr_matrix = corr_matrix.fillna(0)
        
        # Convert correlation to distance
        # Ensure values are strictly within [0, 1] due to floating point inaccuracies
        distance_matrix = 1 - corr_matrix.clip(0, 1)
        
        # Make distance matrix symmetric and set diagonal to 0
        np.fill_diagonal(distance_matrix.values, 0)
        
        # Convert to condensed distance matrix for scipy
        condensed_dist = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        self.logger.info("Performing hierarchical clustering...")
        Z = linkage(condensed_dist, method='average')
        
        # Form flat clusters
        # A distance of (1 - correlation_threshold) means items within cluster 
        # have correlation >= correlation_threshold
        cluster_labels = fcluster(Z, t=1 - self.correlation_threshold, criterion='distance')
        
        # Group metrics by cluster
        clusters: Dict[int, List[str]] = {}
        for metric, label in zip(cols, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(metric)
            
        self.logger.info(f"Found {len(clusters)} unique clusters from {len(cols)} metrics.")
        
        # Select representative and build reduction mapping
        representative_mapping = {}
        cols_to_keep = []
        
        for label, group in clusters.items():
            # For simplicity, pick the first metric with highest variance as representative
            if len(group) == 1:
                representative = group[0]
            else:
                variances = df[group].var()
                representative = variances.idxmax()
                
            representative_mapping[representative] = group
            cols_to_keep.append(representative)
            
        self.logger.info(f"Reduced dimensions from {len(cols)} to {len(cols_to_keep)}.")
        
        final_cols = exclude_cols + cols_to_keep
        return df[final_cols], representative_mapping

    def reduce(self, df: pd.DataFrame, exclude_cols: List[str] = ['timestamp']) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Runs the full dimensionality reduction pipeline.
        """
        if df.empty:
             return df, {}
             
        df_filtered = self.filter_low_variance(df, exclude_cols=exclude_cols)
        
        df_reduced, mapping = self.group_correlated_metrics(df_filtered, exclude_cols=exclude_cols)
        
        return df_reduced, mapping

if __name__ == "__main__":
    reducer = DimensionalityReducer(variance_threshold=0.01, correlation_threshold=0.9)
    
    # 50 data points
    timestamps = pd.date_range("2023-01-01", periods=50, freq="1min")
    
    # Base signal
    signal = np.sin(np.linspace(0, 10, 50))
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "flatline": np.ones(50),                         # Zero variance
        "metric_a": signal,                              # Rep
        "metric_a_copy": signal + np.random.normal(0, 0.05, 50), # Highly correlated
        "metric_b": np.cos(np.linspace(0, 10, 50)),      # Independent
    })
    
    print(f"Original shape: {df.shape}")
    
    reduced_df, mapping = reducer.reduce(df)
    
    print(f"\nReduced shape: {reduced_df.shape}")
    print("\nRepresentative Mapping:")
    for rep, group in mapping.items():
        print(f"  {rep} represents: {group}")
