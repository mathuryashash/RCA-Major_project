"""Configuration constants for the telemetry collector."""

import os
from pathlib import Path

SCHEMA_VERSION = 1
SYSTEM_CADENCE_S = 30
PROCESS_CADENCE_S = 300
PROCESS_BURST_CADENCE_S = 30
EVENT_POLL_S = 300
GAP_FACTOR = 1.5

BURST_CPU_PCT = 80.0
BURST_MEM_PCT = 85.0
BURST_DISK_BUSY_PCT = 80.0
TOP_N = 15
PROC_RETENTION_DAYS = 30
EVENT_RETENTION_DAYS = 365
#: Metric history. Chosen to match the event window so an incident found in
#: the Event Log still has telemetry to explain it -- a shorter figure would
#: leave the app listing year-old faults it can no longer analyse. At the
#: measured 3.33 MB/day this bounds the database near 1.2 GB rather than
#: letting it grow for the life of the machine.
SAMPLE_RETENTION_DAYS = 365
EVENT_CHANNELS = ("System", "Application")
# Only these event families are retained. The collector advances its watermark
# across every event, but discards all non-allowlisted records before storage.
EVENT_ALLOWLIST: dict[str, set[int] | None] = {
    "Microsoft-Windows-Kernel-Power": {41},
    "Application Error": {1000},
    "Application Hang": {1002},
    "disk": {7, 51, 153},
    "Microsoft-Windows-WHEA-Logger": None,
    "Microsoft-Windows-Resource-Exhaustion-Detector": {2004},
    "Microsoft-Windows-WindowsUpdateClient": None,
    "MsiInstaller": None,
}

MUTEX_NAME = "Local\\RCATelemetryCollector"
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 2
STOP_WAIT_S = 35


def gap_threshold_s() -> float:
    return SYSTEM_CADENCE_S * GAP_FACTOR


def app_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    return Path(local_app_data) / "RCA" if local_app_data else Path.home() / ".rca"


def db_path() -> Path:
    return app_dir() / "telemetry.db"


def log_path() -> Path:
    return app_dir() / "collector.log"


def desktop_log_path() -> Path:
    """Separate file for the GUI process.

    Two processes cannot share one RotatingFileHandler on Windows: rollover
    renames the file, which fails while the other process holds it open, and
    logging swallows that error -- so records are dropped from both writers.
    """
    return app_dir() / "desktop.log"


def stop_flag_path() -> Path:
    return app_dir() / "stop.flag"


def source_launcher_path() -> Path:
    return app_dir() / "telemetry_launcher.py"
