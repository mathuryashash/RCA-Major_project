import threading
import time
import requests
import json
import subprocess
import os
from src.main import RCASystem
from scripts.generate_failures import simulate_db_migration_failure

def run_system():
    """Runs the RCA ingestion and parsing system."""
    system = RCASystem()
    system.start()

def run_api():
    """Starts the FastAPI server."""
    # Run the server in a separate process for simplicity in this demo
    # In a real environment, these would be managed by a process manager
    subprocess.Popen(["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"], shell=True)
    time.sleep(3) # Wait for API to start

def demo():
    print("--- STARTING RCA SYSTEM DEMO ---")
    
    # 1. Start System in Background
    system_thread = threading.Thread(target=run_system, daemon=True)
    system_thread.start()
    
    # 2. Start API in Background
    run_api()
    
    # 3. Simulate Failure
    print("\n--- SIMULATING FAILURE (Case Study 1) ---")
    simulate_db_migration_failure()
    
    # 4. Analyze via API
    print("\n--- QUERYING RCA API ---")
    try:
        response = requests.post("http://localhost:8000/analyze", json={
            "start_time": "2026-03-01T14:00:00Z",
            "end_time": "2026-03-01T21:05:00Z"
        })
        
        if response.status_code == 202:
            data = response.json()
            incident_id = data['incident_id']
            print(f"Analysis triggered. Incident ID: {incident_id}")
            
            # Poll for report
            time.sleep(2)
            report_resp = requests.get(f"http://localhost:8000/report/{incident_id}")
            if report_resp.status_code == 200:
                report = report_resp.json()
                print("\n--- FINAL RCA REPORT ---")
                print(f"Status: {report['status']}")
                print(f"Detected At: {report['detected_at']}")
                print("\nRanked Root Causes:")
                for cause in report['ranked_causes']:
                    print(f" - {cause['cause']} (Confidence: {cause['confidence']:.2f})")
                print(f"\nNarrative: {report['narrative']}")
                
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Note: Ensure uvicorn is installed and running.")

    print("\n--- DEMO COMPLETE ---")

if __name__ == "__main__":
    demo()
