import pandas as pd
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional

class LogIntegrator:
    """
    Log Parsing and Integration Manager as per PRD Section 1.1.1.
    
    Capabilities:
    - Parse JSON structured logs and plain text logs
    - Extract timestamps, severity levels
    - Group into 5-minute statistical windows (error counts, log volumes)
    - Output a metrics-compatible DataFrame
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Regex for common log formats if JSON is not available
        self.syslog_pattern = re.compile(
            r'^(?P<timestamp>[A-Z][a-z]{2}\s+\d+\s\d{2}:\d{2}:\d{2})\s+'
            r'(?P<host>\S+)\s+'
            r'(?P<process>[^:]+):\s+'
            r'(?P<message>.*)$'
        )
        self.level_pattern = re.compile(r'\b(ERROR|WARN|WARNING|INFO|DEBUG|FATAL|CRITICAL)\b', re.IGNORECASE)

    def parse_json_log(self, log_line: str) -> Optional[Dict]:
        """Extract timestamp and severity from a JSON log line."""
        try:
            log_data = json.loads(log_line)
            
            # Common timestamp keys
            ts_keys = ['timestamp', 'time', '@timestamp', 'ts', 'date']
            timestamp = None
            for key in ts_keys:
                if key in log_data:
                    timestamp = log_data[key]
                    break
                    
            if not timestamp:
                return None
                
            # Common severity level keys
            level_keys = ['level', 'severity', 'log.level']
            level = 'INFO'  # Default
            for key in level_keys:
                if key in log_data:
                    level = str(log_data[key]).upper()
                    break
                    
            if level == 'WARNING':
                level = 'WARN'
                
            return {
                'timestamp': timestamp,
                'level': level,
                'message': log_data.get('message', '')
            }
        except json.JSONDecodeError:
            return None

    def parse_text_log(self, log_line: str) -> Optional[Dict]:
        """Attempt to extract info from standard plaintext log lines."""
        # Simple heuristic: try to find a timestamp-looking string 
        # and a log level keywords.
        # Handling full diverse regexes is complex, so here is a basic approach.
        
        match = self.syslog_pattern.match(log_line)
        if match:
            # Assuming current year for syslog timestamp "Oct 11 22:14:15"
            ts_str = match.group('timestamp')
            try:
                # Add current year
                ts_str = f"{datetime.now().year} {ts_str}"
                timestamp = datetime.strptime(ts_str, "%Y %b %d %H:%M:%S")
            except ValueError:
                timestamp = None
        else:
            # Fallback timestamp extraction (very basic ISO8601 attempt)
            iso_match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', log_line)
            if iso_match:
                timestamp = iso_match.group(0)
            else:
                return None
                
        # Find level
        level_match = self.level_pattern.search(log_line)
        level = level_match.group(1).upper() if level_match else 'INFO'
        if level == 'WARNING':
            level = 'WARN'
            
        return {
            'timestamp': timestamp,
            'level': level,
            'message': log_line
        }

    def process_log_file(self, file_path: str, format_type: str = 'json') -> pd.DataFrame:
        """
        Reads a log file and aggregates errors/warnings into 5-minute bins.
        """
        self.logger.info(f"Processing log file: {file_path}")
        parsed_logs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    data = None
                    if format_type == 'json':
                        data = self.parse_json_log(line)
                    else:
                        data = self.parse_text_log(line)
                        
                    if data and data['timestamp']:
                        parsed_logs.append(data)
                        
        except FileNotFoundError:
            self.logger.error(f"Log file not found: {file_path}")
            return pd.DataFrame()
            
        if not parsed_logs:
            self.logger.warning("No valid log entries could be parsed.")
            return pd.DataFrame()

        df = pd.DataFrame(parsed_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # We want to extract boolean columns for different severity states
        df['is_error'] = df['level'].isin(['ERROR', 'FATAL', 'CRITICAL']).astype(int)
        df['is_warn'] = (df['level'] == 'WARN').astype(int)
        df['is_info'] = df['level'].isin(['INFO', 'DEBUG']).astype(int)
        
        # Set index and aggregate into 5-minute windows
        df.set_index('timestamp', inplace=True)
        
        # Resample logic:
        # Sum of errors, warns, and total count per 5 minutes
        resampled_df = df[['is_error', 'is_warn', 'is_info']].resample('5min').sum()
        
        # Rename columns to serve as proper metrics
        # E.g. 'log_errors_count', 'log_warns_count'
        resampled_df.rename(columns={
            'is_error': 'log_error_count',
            'is_warn': 'log_warn_count',
            'is_info': 'log_info_count'
        }, inplace=True)
        
        resampled_df['log_total_volume'] = resampled_df.sum(axis=1)
        
        # Reset index to bring timestamp back as a regular column
        resampled_df.reset_index(inplace=True)
        
        return resampled_df
        
    def merge_with_metrics(self, metrics_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the 5-minute binned log data with the existing metrics dataframe.
        """
        if logs_df.empty:
            return metrics_df
            
        if metrics_df.empty:
            return logs_df
            
        # Ensure timestamp alignment
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
        
        # Left merge to keep metrics timeline intact
        merged_df = pd.merge(metrics_df, logs_df, on='timestamp', how='left')
        
        # Fill missing log slots with 0 (no logs during that window)
        log_columns = [col for col in merged_df.columns if col.startswith('log_')]
        merged_df[log_columns] = merged_df[log_columns].fillna(0)
        
        return merged_df

if __name__ == "__main__":
    # Test stub
    integrator = LogIntegrator()
    print("Log integrator established.")
