import os
import time
import threading
from src.common.config import load_config
from src.ingestion.log_ingest import LogHandler
from src.preprocessing.log_parser import LogTemplateExtractor
from watchdog.observers import Observer

class RCASystem:
    def __init__(self):
        self.config = load_config()
        self.log_extractor = LogTemplateExtractor()
        self.observer = Observer()
        
    def _process_log_record(self, record, label):
        """
        Callback for when a new log record is parsed.
        """
        # Extract template using Drain3
        template_info = self.log_extractor.extract_template(record['raw_message'])
        
        # Combine information
        processed_record = {
            **record,
            "template": template_info['template'],
            "cluster_id": template_info['cluster_id'],
            "is_new_template": template_info['is_new']
        }
        
        print(f"[{label}] Template {processed_record['cluster_id']}: {processed_record['template']}")
        
        # Periodic state saving (in a real system, you'd do this more strategically)
        # self.log_extractor.save_state()

    def start(self):
        print("Starting Automated RCA System...")
        
        if not self.config or 'log_sources' not in self.config:
            print("Configuration error. Exiting.")
            return

        # Setup Log Ingestion with a custom callback for processing
        for source in self.config['log_sources']:
            log_path = os.path.abspath(source['path'])
            log_dir = os.path.dirname(log_path)
            
            # Ensure path exists
            os.makedirs(log_dir, exist_ok=True)
            if not os.path.exists(log_path):
                with open(log_path, 'a'): os.utime(log_path, None)
            
            # Subclass LogHandler to add our processing logic
            class ProcessingLogHandler(LogHandler):
                def __init__(self, source_config, callback):
                    super().__init__(source_config)
                    self.callback = callback
                
                def _read_new_lines(self):
                    current_size = os.path.getsize(self.path)
                    if current_size < self.last_position: self.last_position = 0
                    if current_size > self.last_position:
                        with open(self.path, 'r') as f:
                            f.seek(self.last_position)
                            new_lines = f.readlines()
                            self.last_position = f.tell()
                            for line in new_lines:
                                parsed = self._parse_line(line.strip())
                                if parsed:
                                    self.callback(parsed, self.label)

            handler = ProcessingLogHandler(source, self._process_log_record)
            self.observer.schedule(handler, log_dir, recursive=False)
            print(f"Monitoring {source['label']} at {source['path']}")

        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.log_extractor.save_state()
            self.observer.stop()
        self.observer.join()

if __name__ == "__main__":
    system = RCASystem()
    system.start()
