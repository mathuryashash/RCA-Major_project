import logging
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import pandas as pd
import threading

class DeploymentEventListener:
    """
    Listens for deployment events via Webhooks (e.g., GitHub, GitLab).
    Stores these events locally so the Causal Inference Engine can correlate
    anomalies with code changes.
    """
    
    def __init__(self, port: int = 8080, log_file: str = "deployment_events.jsonl"):
        self.port = port
        self.log_file = log_file
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        self.server = None
        self.thread = None

    def start(self):
        """Starts the webhook listener in a background thread."""
        handler_class = self._make_handler_class()
        self.server = HTTPServer(('', self.port), handler_class)
        self.logger.info(f"Starting Deployment Webhook listener on port {self.port}...")
        
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stops the webhook listener."""
        if self.server:
            self.logger.info("Stopping Webhook listener...")
            self.server.shutdown()
            self.server.server_close()

    def _make_handler_class(self):
        """Creates an inner handler class bound to this instance."""
        outer_self = self
        
        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                try:
                    payload = json.loads(post_data.decode('utf-8'))
                    outer_self._process_payload(payload, self.headers)
                    
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                except json.JSONDecodeError:
                    outer_self.logger.error("Received invalid JSON payload.")
                    self.send_response(400)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                # Suppress default HTTP logging to keep console clean
                pass
                
        return WebhookHandler

    def _process_payload(self, payload: dict, headers):
        """
        Parses the webhook payload.
        Currently handles a generic format or GitHub style 'push' / 'workflow_run' events.
        """
        event_time = datetime.utcnow().isoformat() + "Z"
        event_desc = "Unknown deployment event"
        event_type = "deployment"
        
        # GitHub Action workflow success
        if 'action' in payload and 'workflow_run' in payload:
            if payload['action'] == 'completed' and payload['workflow_run']['conclusion'] == 'success':
                repo = payload.get('repository', {}).get('name', 'unknown_repo')
                workflow = payload['workflow_run'].get('name', 'Deploy')
                event_desc = f"Successful '{workflow}' deployment on {repo}"
        # Generic Custom Webhook
        elif 'message' in payload:
            event_desc = payload['message']
            event_type = payload.get('type', 'deployment')
            
        event_record = {
            "timestamp": event_time,
            "description": event_desc,
            "type": event_type
        }
        
        self.logger.info(f"Recorded Event: {event_desc}")
        
        # Append to JSONL file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event_record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write event to disk: {e}")

    def load_events(self, max_days_old: int = 7) -> pd.DataFrame:
        """
        Loads the recorded deployment events into a pandas DataFrame 
        suitable for the Causal Inference Engine.
        """
        try:
            df = pd.read_json(self.log_file, lines=True)
            if df.empty:
                return pd.DataFrame()
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by age
            cutoff = datetime.utcnow() - pd.Timedelta(days=max_days_old)
            # Make cutoff timezone aware if DataFrame is timezone aware
            if df['timestamp'].dt.tz is not None:
                cutoff = cutoff.replace(tzinfo=df['timestamp'].dt.tz)
                
            df = df[df['timestamp'] >= cutoff]
            return df
        except FileNotFoundError:
            self.logger.warning(f"No event log found at {self.log_file}")
            return pd.DataFrame()

if __name__ == "__main__":
    import time
    listener = DeploymentEventListener(port=8080)
    
    print("Starting listener on port 8080...")
    print("Test it via: curl -X POST -H \"Content-Type: application/json\" -d '{\"message\":\"Testing manual deploy\"}' http://localhost:8080/")
    
    listener.start()
    
    try:
        # Keep main thread alive
        time.sleep(5)
        print("Stopping listener test.")
    finally:
        listener.stop()
