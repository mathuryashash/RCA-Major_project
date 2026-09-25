"""Show what the collector is actually capturing, and how much of it."""

import time
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from telemetry import config, schedule, store, updates
from telemetry.analysis import MODELLED_COLUMNS, store_summary
from telemetry.collector import request_stop

#: column -> (group, human label, unit). Order here is the display order.
CHANNELS = [
    ("cpu_pct", "CPU", "Utilisation", "%"),
    ("cpu_pct_max_core", "CPU", "Busiest core", "%"),
    ("cpu_freq_mhz", "CPU", "Frequency", "MHz"),
    ("cpu_freq_ratio", "CPU", "Frequency ratio (throttle proxy)", ""),
    ("cpu_busy_s_delta", "CPU", "Busy time per tick", "core-s"),
    ("gpu_util_pct", "GPU", "Utilisation", "%"),
    ("gpu_mem_used_bytes", "GPU", "Memory used", "bytes"),
    ("gpu_temp_c", "GPU", "Temperature", "°C"),
    ("mem_pct", "Memory", "Used", "%"),
    ("mem_available_mb", "Memory", "Available", "MB"),
    ("mem_used_bytes", "Memory", "Used", "bytes"),
    ("swap_pct", "Memory", "Swap used", "%"),
    ("swap_used_bytes", "Memory", "Swap in use", "bytes"),
    ("swap_used_delta", "Memory", "Swap change per tick", "bytes"),
    ("disk_read_bps", "Disk", "Read rate", "B/s"),
    ("disk_write_bps", "Disk", "Write rate", "B/s"),
    ("disk_busy_pct", "Disk", "Busy", "%"),
    ("disk_free_pct", "Disk", "Free space", "%"),
    ("net_sent_bps", "Network", "Sent", "B/s"),
    ("net_recv_bps", "Network", "Received", "B/s"),
    ("process_count", "Load", "Running processes", ""),
    ("battery_pct", "Power", "Charge", "%"),
    ("battery_drain_rate", "Power", "Drain rate", "%/h"),
    ("power_plugged", "Power", "On mains", "0/1"),
    ("user_idle_sec", "Context", "Idle time", "s"),
    ("foreground_app", "Context", "Foreground app", ""),
]


#: How old the newest sample may be before collection is reported as stopped.
#: Ten missed ticks: long enough to ride out a supervisor restart (fifteen
#: seconds of backoff plus startup) and the first tick after the machine
#: wakes, short enough that a dead collector is flagged within minutes rather
#: than the hour it once went unnoticed behind "Collecting every 30 seconds".
STALE_AFTER_S = 10 * config.SYSTEM_CADENCE_S

#: How long after a restart click to keep saying "waiting" before admitting
#: the restart did not work. A few ticks, plus the collector's own startup.
RESTART_GRACE_S = 6 * config.SYSTEM_CADENCE_S


def sample_age_s(last_ts, now: float | None = None) -> float | None:
    """Seconds since the newest stored sample, or None if there is none."""
    if last_ts is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, now - last_ts.timestamp())


def describe_age(seconds: float) -> str:
    """Minutes, hours or days, rounded -- precise enough to act on."""
    minutes = seconds / 60
    if minutes < 90:
        value, unit = round(minutes), "minute"
    elif minutes < 48 * 60:
        value, unit = round(minutes / 60), "hour"
    else:
        value, unit = round(minutes / 1440), "day"
    return f"{value} {unit}{'s' if value != 1 else ''}"


def _local_clock(last_ts, now: float) -> str:
    """When the last sample landed, in the user's own time zone."""
    stamp = datetime.fromtimestamp(last_ts.timestamp())
    same_day = stamp.date() == datetime.fromtimestamp(now).date()
    return stamp.strftime("%H:%M" if same_day else "%Y-%m-%d %H:%M")


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


class DataView(QWidget):
    """A plain account of the collected store: volume, span, and channels.

    Two audiences read this screen. Someone who just wants to know their PC
    is being watched over and nothing is leaving the device needs one
    sentence and a colour. Someone auditing the tool -- a reviewer, a
    technical user, the person who wrote it -- needs the full channel table,
    the raw counts and the on-disk path. Advanced mode is how both get their
    screen without either seeing the other's.
    """

    def __init__(self, state=None, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Everything below is measured on this machine and stored locally. "
            "Nothing is generated, and nothing leaves the device."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Simple mode's entire answer: one status line, coloured by whether
        # collection is actually healthy. Scanning a QFormLayout for problems
        # means reading every row at equal visual weight; this exists so
        # "everything is fine" or "something needs attention" is legible at
        # a glance, and it reuses the same coverage/gap numbers the raw store
        # box below computes so the two cannot disagree with each other.
        self.summary_card = QGroupBox("Status")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("—")
        self.summary_label.setObjectName("dataSummaryNeutral")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        self.summary_card.setLayout(summary_layout)
        layout.addWidget(self.summary_card)

        self.store_box = QGroupBox("Collected Store")
        form = QFormLayout()
        self.labels = {}
        for key, caption in (
            ("samples", "System samples (every 30s)"),
            ("proc_samples", "Process samples (every 5min)"),
            ("events", "Windows Event Log entries"),
            ("coverage", "Coverage of that span"),
            ("sampling_gaps", "Breaks in collection"),
            ("gaps", "Event Log coverage gaps"),
            ("span", "Collecting since"),
            ("size", "Size on disk"),
            ("retention", "Kept for"),
            ("path", "Database"),
        ):
            self.labels[key] = QLabel("—")
            self.labels[key].setTextInteractionFlags(Qt.TextSelectableByMouse)
            # Retention and the database path are long single lines; without
            # wrapping they forced a horizontal scrollbar across the whole tab.
            self.labels[key].setWordWrap(True)
            self.labels[key].setMinimumWidth(1)
            form.addRow(caption, self.labels[key])
        self.store_box.setLayout(form)
        layout.addWidget(self.store_box)

        self.channel_box = QGroupBox("Captured Channels")
        channel_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Group", "Channel", "Latest value", "Unit", "Used by model"]
        )
        self.table.verticalHeader().setVisible(False)
        # 430px of forced height sat on top of the form above it, pushing the
        # page past the window and putting a second scrollbar beside the
        # table's own -- two nested bars in the same corner, which is what the
        # eye reads as the panel sliding under itself. The table still scrolls
        # internally for the full channel list; it simply no longer demands
        # more height than the page can give.
        self.table.setMinimumHeight(220)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        channel_layout.addWidget(self.table)
        self.channel_box.setLayout(channel_layout)
        layout.addWidget(self.channel_box, stretch=1)

        # An off switch. Recording which applications someone uses, every 30
        # seconds, with no way to stop short of uninstalling is not a defensible
        # position for a tool whose case rests on privacy. The mechanism already
        # existed -- the collector and the supervisor both poll a stop flag --
        # it had simply never been offered to the person being recorded.
        controls = QHBoxLayout()
        self.pause_button = QPushButton("Pause collection")
        self.pause_button.setAccessibleName("Pause or resume telemetry collection")
        self.pause_button.clicked.connect(self._toggle_collection)
        self.collection_state = QLabel("")
        self.collection_state.setWordWrap(True)
        # Offered only while collection has stopped without being paused.
        # The supervisor is meant to make this unnecessary, and it failed
        # anyway -- a rebuild deleted the collector and it quit for good -- so
        # the fallback has to be one click, not a command line.
        self.restart_button = QPushButton("Restart collection")
        self.restart_button.setAccessibleName("Restart telemetry collection, which has stopped")
        self.restart_button.clicked.connect(self._restart_collection)
        self.restart_button.setVisible(False)
        self._last_sample_ts = None
        self._restart_requested_at: float | None = None
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        # The only control in this application that can open a socket, so it
        # says so on its face rather than in a settings pane somewhere.
        self.update_button = QPushButton("Check for updates")
        self.update_button.setAccessibleName("Check online for a newer release")
        self.update_button.clicked.connect(self._check_for_update)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.restart_button)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.update_button)
        controls.addWidget(self.collection_state, stretch=1)
        layout.addLayout(controls)

        # Raw store box and full channel table are advanced-only; the status
        # card and controls stay visible in both modes since they are what
        # this screen does, not detail about how it does it.
        self.set_advanced(bool(state.advanced_mode) if state else False)
        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(30_000)
        # Parented to the view so it dies with it; a bare singleShot would
        # call refresh() on a widget that may already have been destroyed.
        self._recheck = QTimer(self)
        self._recheck.setSingleShot(True)
        self._recheck.timeout.connect(self.refresh)

    def set_advanced(self, enabled: bool):
        self.store_box.setVisible(enabled)
        self.channel_box.setVisible(enabled)

    def _check_for_update(self):
        """Ask GitHub whether a newer release exists. Nothing else.

        Consent is asked once, in plain terms, because this is the only place
        the application contacts anything. Declining is remembered, and the
        check never runs on its own -- no timer, no startup probe.
        """
        from version import __version__

        try:
            connection = store.connect(config.db_path())
            store.init_schema(connection)
        except Exception as exc:  # noqa: BLE001
            self.collection_state.setText(f"Could not check: {exc}")
            return

        if not updates.is_enabled(connection):
            answer = QMessageBox.question(
                self, "Check for updates",
                "This contacts github.com to read the version number of the "
                "newest release, and nothing else.\n\n"
                "No telemetry is sent, nothing is downloaded or installed, and "
                "the check only ever runs when you press this button.\n\n"
                "It is the only part of this application that uses the "
                "network. Allow it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.collection_state.setText("Update checks stay off.")
                return
            updates.set_enabled(connection, True)

        self.update_button.setEnabled(False)
        self.collection_state.setText("Checking …")
        status = updates.check(__version__, conn=connection)
        self.update_button.setEnabled(True)

        if not status.checked:
            self.collection_state.setText(status.reason)
        elif status.available:
            self.collection_state.setText(
                f"Version {status.latest} is available (you have {status.current}). "
                f"Download it from {status.url}"
            )
        else:
            self.collection_state.setText(
                f"Up to date ({__version__})." if not status.reason
                else f"{status.reason} ({__version__})."
            )

    def _toggle_collection(self):
        """Stop or restart collection, and say what actually happened.

        Pausing is not instant: the collector polls the flag once per
        thirty-second cycle, so telling the user it has stopped the moment the
        button is clicked would be a small lie of exactly the kind this
        project spends its effort removing.
        """
        try:
            if schedule.collection_paused():
                schedule.resume_collection()
                # The newest sample is still from before the pause, so without
                # this the next refresh would call a resumed collector stopped.
                self._restart_requested_at = time.monotonic()
            else:
                request_stop()
                self._restart_requested_at = None
        except Exception as exc:  # noqa: BLE001 - the view must survive either way
            self.collection_state.setText(f"Could not change collection: {exc}")
            return
        self._refresh_collection_state(just_toggled=True)

    def _restart_collection(self):
        """Relaunch a collector that stopped without being asked to."""
        try:
            started = schedule.restart_collection()
        except Exception as exc:  # noqa: BLE001 - the view must survive either way
            started, reason = False, str(exc)
        else:
            reason = "the collector executable could not be found"
        if not started:
            self._set_collection_text(f"Could not restart collection: {reason}.", error=True)
            return
        self._restart_requested_at = time.monotonic()
        self._refresh_collection_state()
        self._refresh_stall_summary()
        # Check back once the first tick should have landed, rather than
        # leaving the user to wait for the regular thirty-second refresh.
        self._recheck.start(int(2 * config.SYSTEM_CADENCE_S * 1000))

    def stalled_for_s(self, now: float | None = None) -> float | None:
        """How long collection has been stopped, or None if it has not.

        Paused is not stopped: the user asked for that, and the view already
        says so. Nor is an empty store -- that is "not started yet", which has
        its own message.
        """
        try:
            if schedule.collection_paused():
                return None
        except Exception:  # noqa: BLE001
            return None
        age = sample_age_s(self._last_sample_ts, now)
        return age if age is not None and age > STALE_AFTER_S else None

    def _set_collection_text(self, text: str, error: bool = False):
        # errorText is the existing red used for failures elsewhere in the app.
        self.collection_state.setText(text)
        name = "errorText" if error else ""
        if self.collection_state.objectName() != name:
            self.collection_state.setObjectName(name)
            self._repolish(self.collection_state)

    def _refresh_collection_state(self, just_toggled: bool = False):
        try:
            paused = schedule.collection_paused()
        except Exception:  # noqa: BLE001
            return
        self.pause_button.setText("Resume collection" if paused else "Pause collection")
        cadence = config.SYSTEM_CADENCE_S
        if paused:
            self.restart_button.setVisible(False)
            self._set_collection_text(
                f"Paused. The collector stops within about {cadence} seconds and "
                "stays stopped across restarts until you resume. Data already "
                "collected is kept."
                if just_toggled else
                "Collection is paused. Nothing new is being recorded."
            )
            return

        # "Collecting" was only ever inferred from the absence of a pause. The
        # collector can also simply not be running, and when a rebuild took it
        # away for an hour this line kept promising samples every 30 seconds.
        # The age of the newest sample is the one thing that cannot lie.
        stalled = self.stalled_for_s()
        waiting = (
            self._restart_requested_at is not None
            and time.monotonic() - self._restart_requested_at < RESTART_GRACE_S
        )
        if stalled is None:
            self._restart_requested_at = None
            self.restart_button.setVisible(False)
            self._set_collection_text(
                "Collecting. Resumed under the supervisor." if just_toggled
                else f"Collecting every {cadence} seconds."
            )
            return

        self.restart_button.setVisible(True)
        self.restart_button.setEnabled(not waiting)
        if waiting:
            self._set_collection_text(
                "Restarting collection — waiting for the first new sample, "
                "which should arrive within about a minute."
            )
        elif self._restart_requested_at is not None:
            self._set_collection_text(
                f"Collection is still stopped after a restart: the newest sample "
                f"is {describe_age(stalled)} old. The reason is usually in "
                f"{config.log_path()}.",
                error=True,
            )
        else:
            self._set_collection_text(
                f"Collection has stopped: the newest sample is "
                f"{describe_age(stalled)} old, but one should arrive every "
                f"{cadence} seconds. Nothing new is being recorded.",
                error=True,
            )

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _refresh_stall_summary(self) -> bool:
        """Replace the health sentence when collection has stopped.

        Coverage and breaks describe history, and a store with 98% coverage
        and a collector that died an hour ago would otherwise read as green --
        which is the one state that most needs to be noticed.
        """
        stalled = self.stalled_for_s()
        if stalled is None:
            return False
        now = time.time()
        if self._restart_requested_at is not None and (
                time.monotonic() - self._restart_requested_at < RESTART_GRACE_S):
            style = "dataSummaryWarn"
            text = ("\U0001F7E1 Restarting collection — this turns green once "
                    "new samples arrive.")
        else:
            style = "dataSummaryError"
            text = (
                f"\U0001F534 Collection has stopped — nothing has been recorded "
                f"since {_local_clock(self._last_sample_ts, now)} "
                f"({describe_age(stalled)} ago). Press Restart collection below."
            )
        self.summary_label.setObjectName(style)
        self.summary_label.setText(text)
        self._repolish(self.summary_label)
        return True

    def _refresh_summary(self, summary: dict | None):
        """One coloured sentence: the whole of simple mode's health check."""
        if summary is None or not summary.get("exists"):
            self.summary_label.setObjectName("dataSummaryNeutral")
            self.summary_label.setText(
                "Not collecting yet — the collector starts on its own and "
                "this fills in within about a day."
            )
            self._repolish(self.summary_label)
            return

        days = 0.0
        if summary.get("first_ts") is not None:
            hours = (summary["last_ts"] - summary["first_ts"]).total_seconds() / 3600
            days = hours / 24
        coverage = summary.get("coverage_pct", 0.0)
        breaks = summary.get("sampling_gaps", 0)

        if self._refresh_stall_summary():
            return
        if coverage >= 90 and breaks <= 3:
            style = "dataSummaryGood"
            text = (
                f"\U0001F7E2 Healthy — LocalRCA has been watching this PC for "
                f"{days:.0f} day{'s' if days != 1 else ''} with good coverage "
                f"({coverage:.0f}%)."
            )
        elif coverage >= 50:
            style = "dataSummaryWarn"
            text = (
                f"\U0001F7E1 Partial data — {days:.0f} day{'s' if days != 1 else ''} of "
                f"history, but only {coverage:.0f}% coverage ({breaks} break"
                f"{'s' if breaks != 1 else ''}). Results may be less complete."
            )
        else:
            style = "dataSummaryWarn"
            text = (
                f"\U0001F7E1 Just getting started — {days:.0f} day"
                f"{'s' if days != 1 else ''} of history so far "
                f"({coverage:.0f}% coverage). Keep the app running to build "
                f"up enough clean data to train on."
            )
        self.summary_label.setObjectName(style)
        self.summary_label.setText(text)
        self._repolish(self.summary_label)

    def refresh(self):
        try:
            summary = store_summary()
        except Exception as exc:  # noqa: BLE001 - a locked or busy database must not stop the view
            # Unknown is not stale: a busy database says nothing about whether
            # the collector is alive, so no stopped warning on this path.
            self._last_sample_ts = None
            self._refresh_collection_state()
            # This runs on the UI thread every thirty seconds against a
            # database the collector is writing to. A failure here used to
            # print a traceback nobody could see and leave the numbers frozen
            # at their last good values, which reads as "collection stalled".
            self.labels["path"].setText(f"{config.db_path()} — could not be read: {exc}")
            # Blank the counters too. Leaving them at their last good values is
            # the frozen-numbers symptom this exists to remove.
            for key in ("samples", "proc_samples", "events", "coverage",
                        "sampling_gaps", "gaps", "span", "size", "retention"):
                self.labels[key].setText("—")
            self._refresh_summary(None)
            return

        self._last_sample_ts = summary["last_ts"] if summary["exists"] else None
        self._refresh_collection_state()

        if not summary["exists"]:
            self.labels["path"].setText(f"{config.db_path()} — not created yet")
            for key in ("samples", "proc_samples", "events", "coverage",
                        "sampling_gaps", "gaps", "span", "size", "retention"):
                self.labels[key].setText("—")
            self.table.setRowCount(0)
            self._refresh_summary(summary)
            return

        self.labels["samples"].setText(f"{summary['samples']:,}")
        self.labels["proc_samples"].setText(f"{summary['proc_samples']:,}")
        self.labels["events"].setText(f"{summary['events']:,}")
        self.labels["gaps"].setText(f"{summary['gaps']:,}")
        # Span alone implies continuous collection. It is not: sleep,
        # reboots and collector crashes all leave holes, and training needs
        # unbroken runs rather than total elapsed time.
        self.labels["coverage"].setText(
            f"{summary['samples']:,} of {summary['expected_samples']:,} expected "
            f"({summary['coverage_pct']:.0f}%)"
        )
        self.labels["sampling_gaps"].setText(
            f"{summary['sampling_gaps']:,} breaks, {summary['gap_hours']:.1f} h not collected"
        )
        self.labels["size"].setText(_human_bytes(summary["size_bytes"]))
        # Retention is otherwise invisible: the app deletes the user's history
        # on a schedule and nothing in the interface said so.
        self.labels["retention"].setText(
            f"metrics {config.SAMPLE_RETENTION_DAYS} days · "
            f"process detail {config.PROC_RETENTION_DAYS} days · "
            f"events {config.EVENT_RETENTION_DAYS} days · "
            f"foreground app {config.FOREGROUND_APP_RETENTION_DAYS} days"
        )
        self.labels["path"].setText(str(summary["path"]))
        if summary["first_ts"] is not None:
            hours = (summary["last_ts"] - summary["first_ts"]).total_seconds() / 3600
            self.labels["span"].setText(
                f"{summary['first_ts']:%Y-%m-%d %H:%M} UTC  ({hours:.1f} hours ago)"
            )

        latest = summary["latest"]
        available = summary["available"]
        self.table.setRowCount(len(CHANNELS))
        for row, (column, group, label, unit) in enumerate(CHANNELS):
            value = latest.get(column)
            if column not in available:
                shown = "not available on this machine"
            elif value is None:
                # Rate channels are empty on the first tick after a gap.
                shown = "—  (pending next tick)"
            elif column.endswith("_bytes"):
                shown = _human_bytes(float(value))
            elif isinstance(value, float):
                shown = f"{value:,.2f}"
            else:
                shown = str(value)

            # A channel is collected first and modelled only once it has
            # history; a new one would otherwise invalidate every earlier row.
            modelled = "yes" if column in MODELLED_COLUMNS else "collected only"
            for col, text in enumerate((group, label, shown, unit, modelled)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        self._refresh_summary(summary)
