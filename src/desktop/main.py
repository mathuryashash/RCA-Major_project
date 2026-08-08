"""Desktop app entry point."""

import os
import sys
import threading

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)


def _ensure_std_handles() -> None:
    """Give the process somewhere to write before anything tries to.

    A windowed frozen build (``console=False``) starts with no console, so
    file descriptors 1 and 2 are invalid and sys.stdout/sys.stderr are None.
    Any write then raises, and PySide6 turns an unhandled exception inside a
    slot into qFatal -- the process dies with 0xC0000409 in Qt6Core, tens of
    seconds in, with no message anywhere. Launch the same executable with its
    output redirected and it runs fine, which is why this never showed up in
    a console build or from source.

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


_ensure_std_handles()

from PySide6.QtWidgets import QApplication  # noqa: E402 - needs valid handles first

from desktop.theme import apply_theme  # noqa: E402
from desktop.main_window import MainWindow  # noqa: E402


def _install_crash_logging() -> None:
    """Send unhandled exceptions to the collector log rather than nowhere.

    PySide6 does not abort on an exception raised inside a slot -- measured,
    the loop keeps running -- it prints a traceback and carries on. In a
    windowed build there is nothing to print to, so a timer that fails every
    thirty seconds does so invisibly, and the only symptom is a view that
    quietly stops updating. Production needs those on disk.
    """
    from telemetry.logsetup import get_logger

    # Must sit under "telemetry": that is the logger logsetup attaches the
    # rotating file handler to, and a sibling name would propagate only to the
    # root logger -- straight back to the nowhere this exists to escape.
    logger = get_logger("telemetry.desktop")

    def _excepthook(exc_type, exc, tb):
        logger.error("Unhandled exception", exc_info=(exc_type, exc, tb))

    def _threadhook(args):
        logger.error(
            "Unhandled exception in %s", args.thread and args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _excepthook
    threading.excepthook = _threadhook


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
    _install_crash_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("RCA Desktop")
    apply_theme(app)

    _ensure_collector_running()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
