"""Show what the collector is actually capturing, and how much of it."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from telemetry import config
from telemetry.analysis import MODELLED_COLUMNS, store_summary

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


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


class DataView(QWidget):
    """A plain account of the collected store: volume, span, and channels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Everything below is measured on this machine and stored locally. "
            "Nothing is generated, and nothing leaves the device."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        store_box = QGroupBox("Collected Store")
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
            form.addRow(caption, self.labels[key])
        store_box.setLayout(form)
        layout.addWidget(store_box)

        channel_box = QGroupBox("Captured Channels")
        channel_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Group", "Channel", "Latest value", "Unit", "Used by model"]
        )
        self.table.verticalHeader().setVisible(False)
        # Show a useful number of channels before scrolling.
        self.table.setMinimumHeight(430)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        channel_layout.addWidget(self.table)
        channel_box.setLayout(channel_layout)
        layout.addWidget(channel_box, stretch=1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_button)

        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(30_000)

    def refresh(self):
        try:
            summary = store_summary()
        except Exception as exc:  # noqa: BLE001 - a locked or busy database must not stop the view
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
            return

        if not summary["exists"]:
            self.labels["path"].setText(f"{config.db_path()} — not created yet")
            for key in ("samples", "proc_samples", "events", "coverage",
                        "sampling_gaps", "gaps", "span", "size", "retention"):
                self.labels[key].setText("—")
            self.table.setRowCount(0)
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
