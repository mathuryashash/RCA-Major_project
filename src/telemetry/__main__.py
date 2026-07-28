"""Command-line control for the local telemetry collector."""

import argparse
import sys
import time

from . import config, schedule, store
from .collector import Collector, acquire_singleton, consent_granted, grant_consent, request_stop


def _open():
    connection = store.connect(config.db_path())
    store.init_schema(connection)
    return connection


def _delete_data() -> int:
    schedule.unregister()
    request_stop()
    deadline = time.monotonic() + config.STOP_WAIT_S
    paths = [config.db_path().with_name(config.db_path().name + suffix) for suffix in ("", "-wal", "-shm")]
    while time.monotonic() < deadline:
        try:
            for path in paths:
                path.unlink(missing_ok=True)
            print("Deleted collected telemetry and trained collection state.")
            return 0
        except PermissionError:
            time.sleep(0.25)
    print("Collector did not stop; no data was deleted.", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="telemetry")
    parser.add_argument("command", choices=["run", "install", "uninstall", "status", "accept-consent", "delete-all-data"])
    parser.add_argument("--capture-messages", action="store_true", help="opt in to redacted EventData message storage")
    args = parser.parse_args(argv)
    conn = _open()
    if args.command == "accept-consent":
        print(f"Telemetry is stored locally at {config.db_path()}. It records system metrics, selected process names and Event Log metadata. No data leaves this machine.")
        grant_consent(conn)
        if args.capture_messages:
            store.set_meta(conn, "capture_messages", "1")
        return 0
    if args.command == "status":
        count = store.sample_count(conn)
        print(f"consent: {'granted' if consent_granted(conn) else 'NOT GRANTED'}")
        print(f"scheduled: {schedule.is_registered()}")
        print(f"samples: {count} (~{count * config.SYSTEM_CADENCE_S / 86400:.2f} days)")
        return 0
    if args.command == "delete-all-data":
        conn.close()
        return _delete_data()
    if args.command == "uninstall":
        return 0 if schedule.unregister() else 1
    if not consent_granted(conn):
        print("Consent not granted. Run 'python -m telemetry accept-consent' first.", file=sys.stderr)
        return 1
    if args.command == "install":
        return 0 if schedule.register() else 1
    if not acquire_singleton():
        return 0
    capture_messages = args.capture_messages or store.get_meta(conn, "capture_messages", "0") == "1"
    Collector(conn, capture_messages=capture_messages).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
