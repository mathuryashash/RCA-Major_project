"""First-run consent, asked in the interface rather than on a command line.

Consent existed only as `RCA-Collector.exe accept-consent`, so someone who
opened the desktop application and never read the README was never asked --
and collection silently never began. For a tool that records process names
and Event Log entries, the moment of asking belongs in front of the user.
"""

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from telemetry import config, store
from telemetry.collector import consent_granted, grant_consent

DISCLOSURE = f"""
<p><b>This application records what your computer is doing, so it can explain
a problem after it happens.</b></p>

<p><b>What is recorded, every 30 seconds:</b> CPU, memory, disk, network,
battery and GPU readings, <b>the name of the application you are currently
using</b>, and how long since you last touched the keyboard or mouse.
<b>Every 5 minutes — and every 30 seconds whenever the machine is busy:</b> the
names of the busiest programs, with their CPU, memory and I/O. <b>As they
occur:</b> a fixed list of Windows Event Log entries — crashes, unexpected
shutdowns, disk errors and update activity.</p>

<p><b>Be clear about what that means.</b> The application name and the idle
timer together form a record of when this machine was in use and roughly what
it was used for, and the busiest-program list says much the same. These are
the most personal things collected here, so they are also the shortest-lived:
both are removed after {config.FOREGROUND_APP_RETENTION_DAYS} days, while the
numeric readings beside them are kept longer.</p>

<p><b>What is never recorded:</b> window titles, page addresses, keystrokes,
file contents, browsing history, or the text of documents. Knowing a browser
was in focus does not record which site was open. Event Log message text is
stored only if you ask for it, and is redacted for paths, usernames, URLs and
email addresses first.</p>

<p><b>Where it goes:</b> a database on this machine at
<code>{config.db_path()}</code>. Nothing is uploaded. The application makes no
network connections. Readings are kept {config.SAMPLE_RETENTION_DAYS} days,
process samples {config.PROC_RETENTION_DAYS} days, events
{config.EVENT_RETENTION_DAYS} days, and the application-name record
{config.FOREGROUND_APP_RETENTION_DAYS} days.</p>

<p><b>One honest caveat about those limits.</b> Old data is removed by the
collector while it runs. If you stop it — with <code>uninstall</code>, which
keeps what was already collected — nothing expires after that, and what is on
disk stays until you erase it with <code>delete-all-data</code>. If a database
is ever found damaged it is renamed aside rather than deleted, and that copy
is not covered by the limits above either; both are reported in the
<b>Captured Data</b> tab.</p>

<p><b>What agreeing sets up:</b> the collector is registered to start at every
logon, so recording continues while this window is closed — that is what makes
a usable baseline possible. The application is also added to your Start menu
and to Add&nbsp;/&nbsp;Remove&nbsp;Programs. No administrator rights are used,
and nothing is written outside your own user profile.</p>

<p><b>Undoing this:</b> removing <b>LocalRCA</b> from Add&nbsp;/&nbsp;Remove
Programs, or running <code>RCA-Collector.exe uninstall</code>, reverses all
three and stops collection while keeping what was collected.
<code>delete-all-data</code> erases the database, the trained model, generated
reports and logs as well. Reports you export yourself go wherever you save
them and are not tracked.</p>

<p>Nothing is collected and nothing is registered unless you agree.</p>
"""


class ConsentDialog(QDialog):
    """The disclosure, with agreeing as a deliberate act."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Before collection begins")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)
        disclosure = QLabel(DISCLOSURE)
        disclosure.setWordWrap(True)                # QLabel detects the markup
        layout.addWidget(disclosure)

        self.capture_messages = QCheckBox(
            "Also store Event Log message text (redacted). Off by default."
        )
        layout.addWidget(self.capture_messages)

        buttons = QDialogButtonBox()
        # Not "OK": the button should say what it agrees to.
        buttons.addButton("Start collecting", QDialogButtonBox.AcceptRole)
        buttons.addButton("Not now", QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def ensure_consent(parent=None) -> bool:
    """Return whether collection may proceed, asking once if never asked.

    Declining is remembered only in the sense that nothing is recorded; the
    question is asked again next launch rather than being silently final.
    """
    try:
        connection = store.connect(config.db_path())
        # A first run has no database at all, so the consent flag lives in a
        # table that does not exist yet. Without this the lookup raises on
        # exactly the install this dialog exists to handle, and the caller's
        # broad handler swallows it -- never asking, never collecting.
        store.init_schema(connection)
    except Exception:                               # noqa: BLE001 - GUI must open
        return False

    try:
        if consent_granted(connection):
            return True

        dialog = ConsentDialog(parent)
        if dialog.exec() != QDialog.Accepted:
            return False

        grant_consent(connection)
        if dialog.capture_messages.isChecked():
            store.set_meta(connection, "capture_messages", "1")
        return True
    finally:
        connection.close()
