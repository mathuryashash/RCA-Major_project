import os
import time
import re
import json
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.common.config import load_config

class LogHandler(FileSystemEventHandler):
    """
    Handles file change events and reads new lines from log files.
    """
    def __init__(self, source_config):
        self.label = source_config['label']
        self.path = os.path.abspath(source_config['path'])
        self.format = source_config['format']
        self.last_position = self._get_file_size()
        
        # Pre-compile regex for common formats
        self.plaintext_re = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)$')
        self.syslog_re = re.compile(r'^(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}) (\S+) (\S+): (.+)$')

    def _get_file_size(self):
        if os.path.exists(self.path):
            return os.path.getsize(self.path)
        return 0

    def on_modified(self, event):
        if not event.is_directory and os.path.abspath(event.src_path) == self.path:
            self._read_new_lines()

    def _read_new_lines(self):
        current_size = os.path.getsize(self.path)
        
        # Handle file rotation (if file was truncated or replaced)
        if current_size < self.last_position:
            self.last_position = 0
            
        if current_size > self.last_position:
            with open(self.path, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                
                for line in new_lines:
                    parsed = self._parse_line(line.strip())
                    if parsed:
                        print(f"[{self.label}] Parsed: {parsed}")
                        # In a real system, we'd send this to a processing queue or DB here

    def _parse_line(self, line):
        if not line: return None
        
        record = {
            "timestamp": None,
            "level": "INFO",
            "source_file": self.path,
            "raw_message": line
        }
        
        try:
            if self.format == "json":
                data = json.loads(line)
                record["timestamp"] = data.get("ts") or data.get("timestamp")
                record["level"] = data.get("level", "INFO").upper()
                record["raw_message"] = data.get("msg") or data.get("message")
            elif self.format == "plaintext":
                match = self.plaintext_re.match(line)
                if match:
                    record["timestamp"] = match.group(1)
                    record["level"] = match.group(2).upper()
                    record["raw_message"] = match.group(3)
            elif self.format == "syslog":
                match = self.syslog_re.match(line)
                if match:
                    record["timestamp"] = match.group(1)
                    # Syslog typically doesn't have a level in the message itself in the same way
                    record["raw_message"] = match.group(4)
        except Exception as e:
            print(f"Error parsing line: {e}")
            
        return record

def start_log_ingestion():
    config = load_config()
    if not config or 'log_sources' not in config:
        print("No log sources configured.")
        return

    observer = Observer()
    handlers = []

    for source in config['log_sources']:
        # Ensure directory exists for log file
        log_dir = os.path.dirname(os.path.abspath(source['path']))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created directory: {log_dir}")
            
        # Create empty log file if it doesn't exist
        if not os.path.exists(source['path']):
            with open(source['path'], 'a'):
                os.utime(source['path'], None)
            print(f"Created log file: {source['path']}")

        handler = LogHandler(source)
        observer.schedule(handler, log_dir, recursive=False)
        handlers.append(handler)
        print(f"Monitoring {source['label']} at {source['path']}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_log_ingestion()
