from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import os

class LogTemplateExtractor:
    """
    Uses Drain3 to extract templates from log messages.
    """
    def __init__(self, persistence_path="data/drain3_state.bin"):
        self.config = TemplateMinerConfig()
        # Basic config to handle wildcards
        self.config.drain_sim_th = 0.5
        self.config.drain_depth = 4
        
        self.persistence_path = persistence_path
        self.miner = TemplateMiner(config=self.config)
        
        if os.path.exists(self.persistence_path):
            self._load_state()

    def _load_state(self):
        try:
            with open(self.persistence_path, "rb") as f:
                self.miner.load_state(f)
            print(f"Loaded Drain3 state from {self.persistence_path}")
        except Exception as e:
            print(f"Error loading Drain3 state: {e}")

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, "wb") as f:
                self.miner.save_state(f)
        except Exception as e:
            print(f"Error saving Drain3 state: {e}")

    def extract_template(self, log_message):
        """
        Parses a log message and returns its template and ID.
        """
        result = self.miner.add_log_message(log_message)
        template = result['template_mined']
        cluster_id = result['cluster_id']
        change_type = result['change_type']
        
        return {
            "template": template,
            "cluster_id": cluster_id,
            "is_new": change_type == "cluster_created"
        }

if __name__ == "__main__":
    extractor = LogTemplateExtractor()
    logs = [
        "User 12345 failed login from 192.168.1.1",
        "User 67890 failed login from 10.0.0.5",
        "Connection refused to postgres:5432",
        "Connection refused to mysql:3306"
    ]
    
    for log in logs:
        print(f"Log: {log}")
        print(f"Template: {extractor.extract_template(log)}")
        print("-" * 20)
    
    extractor.save_state()
