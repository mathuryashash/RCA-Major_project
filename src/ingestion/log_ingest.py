import os
import time
import re
import json
import stat
import threading
from queue import Queue, Empty
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
from src.common.config import load_config


class LogFileWatcher:
    """
    Real-time log file watcher using watchdog for efficient OS-level events.

    Features:
    - Track file position to avoid re-reading
    - Handle multiple log files concurrently
    - File rotation detection (inode change)
    - Configurable polling interval for fallback
    - Thread-safe line buffer with Queue
    - Catch-up mode on startup
    """

    def __init__(
        self, poll_interval_seconds: float = 1.0, max_buffer_size: int = 10000
    ):
        self.poll_interval = poll_interval_seconds
        self.max_buffer_size = max_buffer_size
        self._observer = None
        self._handlers = {}
        self._running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread = None

    def start(self, log_sources: list) -> None:
        """
        Start watching all configured log sources.

        Args:
            log_sources: List of source configs with 'label', 'path', 'format' keys
        """
        with self._lock:
            if self._running:
                return

            self._stop_event.clear()
            self._observer = Observer()

            for source in log_sources:
                label = source["label"]
                path = os.path.abspath(source["path"])
                log_format = source["format"]

                log_dir = os.path.dirname(path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                if not os.path.exists(path):
                    with open(path, "a"):
                        os.utime(path, None)

                handler = LogHandler(
                    label, path, log_format, max_buffer_size=self.max_buffer_size
                )
                handler._do_catch_up()
                self._observer.schedule(handler, log_dir, recursive=False)
                self._handlers[path] = handler

            self._observer.start()
            self._running = True

            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def stop(self) -> None:
        """Stop watching all log files."""
        with self._lock:
            if not self._running:
                return

            self._stop_event.set()
            if self._poll_thread:
                self._poll_thread.join(timeout=2.0)

            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            self._handlers.clear()
            self._running = False

    def get_new_lines(self) -> dict:
        """
        Get new lines since last read from all watched files.

        Returns:
            Dict mapping label -> list of parsed log records
        """
        results = {}
        with self._lock:
            for path, handler in self._handlers.items():
                lines = handler.get_new_lines()
                if lines:
                    results[handler.label] = lines
        return results

    def _poll_loop(self) -> None:
        """Background polling loop for files that may not trigger events."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break

            with self._lock:
                for path, handler in self._handlers.items():
                    try:
                        handler._check_for_updates()
                    except PermissionError:
                        pass
                    except FileNotFoundError:
                        handler._on_file_deleted()
                    except Exception:
                        pass

    @property
    def is_running(self) -> bool:
        return self._running


class LogHandler(FileSystemEventHandler):
    """
    Handles file change events and reads new lines from log files.
    """

    plaintext_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)$")
    syslog_re = re.compile(r"^(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}) (\S+) (\S+): (.+)$")

    def __init__(
        self, label: str, path: str, log_format: str, max_buffer_size: int = 10000
    ):
        self.label = label
        self.path = os.path.abspath(path)
        self.format = log_format
        self.max_buffer_size = max_buffer_size
        self._line_buffer = Queue(maxsize=max_buffer_size)
        self._last_position = 0
        self._last_inode = None
        self._file_exists = os.path.exists(path)

        if self._file_exists:
            self._last_inode = self._get_file_inode()
            self._last_position = os.path.getsize(self.path)

        self._lock = threading.Lock()

    def _get_file_inode(self) -> int:
        """Get the inode number of the file for rotation detection."""
        try:
            return os.stat(self.path).st_ino
        except (OSError, FileNotFoundError):
            return None

    def _get_file_size(self) -> int:
        if os.path.exists(self.path):
            return os.path.getsize(self.path)
        return 0

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and os.path.abspath(event.src_path) == self.path:
            self._check_for_updates()

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and os.path.abspath(event.src_path) == self.path:
            self._on_file_created()

    def _on_file_created(self) -> None:
        """Handle file creation (e.g., after rotation)."""
        with self._lock:
            self._file_exists = True
            self._last_inode = self._get_file_inode()
            self._last_position = 0

    def _on_file_deleted(self) -> None:
        """Handle file deletion."""
        with self._lock:
            self._file_exists = False
            self._last_position = 0

    def _check_for_updates(self) -> None:
        """Check and process any new lines in the file."""
        if not self._file_exists and not os.path.exists(self.path):
            self._on_file_deleted()
            return

        current_inode = self._get_file_inode()
        current_size = self._get_file_size()

        if current_inode is None:
            return

        if self._last_inode is not None and current_inode != self._last_inode:
            self._last_position = 0

        self._last_inode = current_inode
        self._file_exists = True

        if current_size < self._last_position:
            self._last_position = 0

        if current_size > self._last_position:
            self._read_new_lines()

    def _read_new_lines(self) -> None:
        """Read new lines from file and add to buffer."""
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._last_position)

                lines_read = 0
                for line in f:
                    if lines_read >= self.max_buffer_size:
                        break
                    line = line.strip()
                    if line:
                        parsed = self._parse_line(line)
                        if parsed:
                            try:
                                self._line_buffer.put_nowait(parsed)
                            except:
                                pass
                            lines_read += 1

                self._last_position = f.tell()
        except PermissionError:
            pass
        except FileNotFoundError:
            self._on_file_deleted()
        except Exception:
            pass

    def _do_catch_up(self) -> None:
        """Read existing lines on startup (catch-up mode)."""
        if os.path.exists(self.path):
            self._read_new_lines()

    def get_new_lines(self) -> list:
        """Get and clear buffered lines."""
        lines = []
        try:
            while True:
                lines.append(self._line_buffer.get_nowait())
        except Empty:
            pass
        return lines

    def _parse_line(self, line: str) -> dict:
        """Parse a log line into a structured record."""
        if not line:
            return None

        record = {
            "timestamp": None,
            "level": "INFO",
            "source": self.label,
            "source_file": self.path,
            "raw_message": line,
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
                    record["raw_message"] = match.group(4)
        except Exception:
            pass

        return record


def start_log_ingestion():
    config = load_config()
    if not config or "log_sources" not in config:
        print("No log sources configured.")
        return

    watcher_config = config.get("log_watching", {})
    poll_interval = watcher_config.get("poll_interval_seconds", 1.0)
    max_buffer = watcher_config.get("max_buffer_size", 10000)
    enabled = watcher_config.get("enabled", True)

    if not enabled:
        print("Log watching is disabled in config.")
        return

    watcher = LogFileWatcher(
        poll_interval_seconds=poll_interval, max_buffer_size=max_buffer
    )
    watcher.start(config["log_sources"])

    try:
        while True:
            time.sleep(1)
            new_logs = watcher.get_new_lines()
            for label, records in new_logs.items():
                for record in records:
                    print(f"[{label}] {record}")
    except KeyboardInterrupt:
        print("\nStopping log watcher...")
        watcher.stop()


def read_full_file(path: str, fmt: str) -> list:
    """
    Read entire log file at once (backward compatible).

    Args:
        path: Path to log file
        fmt: Log format (json, plaintext, syslog)

    Returns:
        List of parsed log records
    """
    records = []
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return records

    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            handler = LogHandler("temp", abs_path, fmt)
            parsed = handler._parse_line(line)
            if parsed:
                records.append(parsed)

    return records


if __name__ == "__main__":
    start_log_ingestion()
