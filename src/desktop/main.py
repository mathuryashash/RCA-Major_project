"""Desktop app entry point."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)


def _ensure_std_handles() -> None:
    """Give the process somewhere to write before anything tries to.

    A windowed frozen build (``console=False``) starts with no console, so
    file descriptors 1 and 2 are invalid and sys.stdout/sys.stderr are None.
    Without this the packaged application died about forty seconds in with
    0xC0000409 in Qt6Core and no message anywhere; with it, the same build
    runs. Launching the unfixed executable with its output redirected also
    made the crash disappear, which is why it never reproduced from source or
    in a console build.

    The precise mechanism is not established. It is not an exception inside a
    slot: measured, PySide6 keeps running after one, with or without valid
    descriptors. Setting sys.stdout alone does not fix it either -- the
    failure is at the descriptor level, which is why this uses dup2.

    Must run before PySide6 is imported, and before any library that writes a
    warning at import time.
    """
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
    except OSError:                                  # nothing more we can do
        return
    for fd in (1, 2):
        try:
            os.fstat(fd)                             # valid already: leave it
        except OSError:
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
    if devnull > 2:
        os.close(devnull)
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")           # noqa: SIM115 - process lifetime
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")           # noqa: SIM115 - process lifetime


def _install_crash_logging() -> None:
    """Send unhandled exceptions to a log file rather than nowhere.

    PySide6 does not abort on an exception raised inside a slot -- measured,
    the loop keeps running -- it prints a traceback and carries on. In a
    windowed build there is nothing to print to, so a timer that fails every
    thirty seconds does so invisibly, and the only symptom is a view that
    quietly stops updating.

    Writes to its own file. Sharing collector.log with the collector process
    looked tidier, but rollover renames the file, Windows refuses while the
    other process holds it open, and logging swallows the error -- measured,
    six records written and three survive, from both writers.

    Installed before the Qt and view imports below, so a mispackaged build
    that fails at import time is recorded too. QThread.run() exceptions
    arrive through sys.excepthook, not threading.excepthook; nothing here
    uses threading.Thread.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    from telemetry import config

    logger = logging.getLogger("desktop")
    logger.setLevel(logging.INFO)
    logger.propagate = False            # never reaches the collector's handler
    if not logger.handlers:
        try:
            config.app_dir().mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                config.desktop_log_path(), maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUPS, encoding="utf-8",
            )
        except OSError:
            return                      # unwritable profile must not stop the app
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        logger.addHandler(handler)

    # Stamp the build into the log: a submitted log that does not say which
    # version produced it costs a round trip to find out.
    from version import __version__
    logger.info("RCA Desktop %s starting", __version__)

    previous = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        logger.error("Unhandled exception", exc_info=(exc_type, exc, tb))
        # Chain, so a console build or a source run still prints to the
        # terminal instead of losing the traceback to the log alone.
        previous(exc_type, exc, tb)

    sys.excepthook = _excepthook


_ensure_std_handles()
_install_crash_logging()

from PySide6.QtWidgets import QApplication  # noqa: E402 - needs valid handles first

from desktop.theme import apply_theme  # noqa: E402
from desktop.main_window import MainWindow  # noqa: E402


def _ensure_collector_running() -> None:
    """Start the collector if it is not already up.

    The Startup-folder entry only fires at logon, so a collector that dies
    mid-session stays dead for the rest of the day and the baseline silently
    stops growing. Launching here is safe and idempotent: the new process
    takes the singleton mutex, and a duplicate exits immediately.

    Collection still requires consent -- this never starts recording on a
    machine where the user has not agreed to it.
    """
    from telemetry import config, schedule, store
    from telemetry.collector import consent_granted

    if not config.db_path().exists():
        return                      # nothing collected yet, so no consent yet

    try:
        connection = store.connect(config.db_path())
        try:
            granted = consent_granted(connection)
        finally:
            connection.close()
        if granted:
            schedule.start_now()
    except Exception:  # noqa: BLE001 - the GUI must open regardless
        pass


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("RCA Desktop")
    apply_theme(app)

    _ensure_collector_running()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
