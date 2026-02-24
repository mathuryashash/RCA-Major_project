import logging
import pandas as pd
import numpy as np
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
from typing import List, Optional

class PrometheusDataIngestion:
    """
    Data Ingestion Manager for Prometheus as per PRD Section 1.1.1.
    
    Capabilities:
    - Queries Prometheus API for metrics
    - Fetches historical data (last N days)
    - Automatically handles 5-minute resolution (300 seconds)
    - Merges multiple time series
    - Outputs a pandas DataFrame shaped (timestamps, metrics)
    """

    def __init__(self, url: str = "http://localhost:9090", disable_ssl: bool = True):
        self.prom = PrometheusConnect(url=url, disable_ssl=disable_ssl)
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        
    def check_connection(self) -> bool:
        """Verify connection to Prometheus server"""
        try:
            return self.prom.check_prometheus_connection()
        except Exception as e:
            self.logger.error(f"Failed to connect to Prometheus: {e}")
            return False

    def fetch_metrics(
        self, 
        query: str, 
        start_time: datetime, 
        end_time: datetime, 
        step: str = '5m'
    ) -> pd.DataFrame:
        """
        Fetches a specific PromQL query and formats it into a DataFrame.
        """
        metric_data = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        if not metric_data:
            self.logger.warning(f"No data returned for query: {query}")
            return pd.DataFrame()
            
        dfs = []
        for series in metric_data:
            labels = series['metric']
            
            # Construct a clear metric name from labels
            name_parts = []
            if '__name__' in labels:
                name_parts.append(labels['__name__'])
            for k, v in labels.items():
                if k != '__name__':
                    name_parts.append(f"{k}_{v}")
            
            metric_name = "_".join(name_parts) if name_parts else 'unknown_metric'
            
            # Clean name for columns
            metric_name = "".join([c if c.isalnum() else "_" for c in metric_name])
            
            values = series['values']
            df_series = pd.DataFrame(values, columns=['timestamp', metric_name])
            df_series['timestamp'] = pd.to_datetime(df_series['timestamp'], unit='s')
            df_series[metric_name] = pd.to_numeric(df_series[metric_name])
            df_series.set_index('timestamp', inplace=True)
            dfs.append(df_series)
            
        if not dfs:
            return pd.DataFrame()
            
        # Merge all series
        final_df = pd.concat(dfs, axis=1)
        
        # Resample to ensure exact 5-minute alignments
        # Ffill and Bfill handle small gaps, linear interpolation for smoothed gaps
        final_df = final_df.resample('5min').mean().interpolate(method='linear')
        
        return final_df

    def ingest_prometheus_metrics(
        self, 
        days: int = 7, 
        queries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Queries Prometheus API for metrics and merges them.
        
        Output Format: pandas DataFrame
        Shape: (num_timesteps, num_metrics)
        Columns: timestamp, metric_1, metric_2, ..., metric_n
        """
        if queries is None:
            # Default queries suitable for infrastructure monitoring (e.g. Node Exporter)
            queries = [
                'rate(node_cpu_seconds_total{mode!="idle"}[5m])',
                'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes',
                'rate(node_network_receive_bytes_total[5m])',
                'rate(node_network_transmit_bytes_total[5m])',
                'rate(node_disk_read_bytes_total[5m])',
                'rate(node_disk_written_bytes_total[5m])'
            ]
            
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_dataframes = []
        for i, query in enumerate(queries):
            self.logger.info(f"Fetching data for query {i+1}/{len(queries)}")
            df = self.fetch_metrics(query, start_time, end_time)
            
            if not df.empty:
                # If a query returns multiple series (e.g., cpu0, cpu1...), mean aggregate them 
                # for a single high-level feature to prevent explosion, or keep them separate.
                # Here we keep only the top 5 most active series to prevent column explosion.
                if df.shape[1] > 5:
                    top_cols = df.var().nlargest(5).index
                    df = df[top_cols]
                all_dataframes.append(df)
                
        if not all_dataframes:
             self.logger.error("No data fetched from Prometheus. Returning empty DataFrame.")
             return pd.DataFrame()
             
        # Combine all queries into one large DataFrame
        combined_df = pd.concat(all_dataframes, axis=1)
        
        # Drop columns with all NaNs that might have appeared during join
        combined_df.dropna(axis=1, how='all', inplace=True)
        
        # Handle remaining NaNs
        combined_df = combined_df.ffill().bfill().fillna(0)
        
        # PRD specifies exact format with explicit 'timestamp' column
        combined_df.reset_index(inplace=True)
        return combined_df

if __name__ == "__main__":
    # Standard quick test
    ingester = PrometheusDataIngestion()
    if ingester.check_connection():
        print("Connected to Prometheus successfully!")
        
        # Example: Fetch 1 hour of data to test
        df = ingester.ingest_prometheus_metrics(days=1)
        print(f"Ingested DataFrame Shape: {df.shape}")
        if not df.empty:
            print("Columns:")
            print(df.columns.tolist())
    else:
        print("Warning: Could not connect to Prometheus at http://localhost:9090.")
        print("Set up a local Prometheus instance to test this ingestion script fully.")
