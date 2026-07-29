"""Desktop app entry point."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
if _src not in sys.path:
    sys.path.insert(0, _src)

from PySide6.QtWidgets import QApplication

from desktop.theme import apply_theme
from desktop.main_window import MainWindow


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
