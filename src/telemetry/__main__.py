"""Command-line control for the local telemetry collector."""

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import config, schedule, store
from .collector import Collector, acquire_singleton, consent_granted, grant_consent, request_stop


def _open():
    connection = store.connect(config.db_path())
    store.init_schema(connection)
    return connection


def _remove_rendered_figures() -> None:
    """Clear rendered figures the desktop app left in the temp directory.

    The graph and timeline are drawn by writing HTML to a mkdtemp directory,
    cleaned at interpreter exit -- which never runs if the app is killed or
    the machine loses power. Those files hold the metric values behind an
    incident and live outside the data directory, so "erase all local data"
    walked straight past them; two such directories, several megabytes each,
    were still present here.
    """
    for leftover in Path(tempfile.gettempdir()).glob("rca_desktop_*"):
        shutil.rmtree(leftover, ignore_errors=True)


def _delete_data() -> int:
    """Erase everything this application has written to this machine.

    Deleting only the database left the rest of the directory behind: the
    trained model, generated reports naming real processes, and collector.log,
    which records exception tracebacks carrying the user's profile path. The
    command promises to erase all local data, so the directory goes as a whole.
    """
    if not schedule.unregister():
        # The launcher it points at is about to be deleted, so a surviving
        # entry means a failing console window at every logon until it goes.
        print(f"Could not remove the startup entry. Delete it by hand from {schedule.startup_dir()}.", file=sys.stderr)
    request_stop()
    # This process holds collector.log open through its own logger, and Windows
    # will not remove an open file. Close that handler only -- logging.shutdown()
    # would take every other handler in the process down with it.
    collector_log = logging.getLogger("telemetry")
    for handler in list(collector_log.handlers):
        collector_log.removeHandler(handler)
        handler.close()

    app_dir = config.app_dir()
    database = config.db_path()
    started = False
    deadline = time.monotonic() + config.STOP_WAIT_S
    while time.monotonic() < deadline:
        try:
            # Deleting the database is the proof that the collector has let go:
            # Windows holds the file locked until it closes the connection.
            # Nothing may be removed before this succeeds. stop.flag lives in
            # this same directory, and clearing the directory first would
            # retract the very signal the collector is waiting to read -- it
            # polls once per cadence, so it would never see it and never exit.
            database.unlink(missing_ok=True)
        except PermissionError:
            time.sleep(0.25)
            continue
        started = True
        shutil.rmtree(app_dir, ignore_errors=True)
        _remove_rendered_figures()
        if not app_dir.exists():
            print("Deleted all local telemetry, reports, models and logs.")
            return 0
        time.sleep(0.25)
    if started:
        print("Some data could not be deleted; run the command again.", file=sys.stderr)
    else:
        print("Collector is still running; nothing was deleted.", file=sys.stderr)
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
        if not schedule.register():
            print("Could not write the startup entry.", file=sys.stderr)
            return 1
        started = schedule.start_now()
        print(f"Registered at {schedule.startup_dir()}. Collection starts at every logon.")
        print("Collector started now." if started else "Start it now with: python -m telemetry run")
        return 0
    if not acquire_singleton():
        print("Another collector is already running; this instance will exit.", file=sys.stderr)
        return 0
    capture_messages = args.capture_messages or store.get_meta(conn, "capture_messages", "0") == "1"
    Collector(conn, capture_messages=capture_messages).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
