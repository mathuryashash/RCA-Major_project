"""Main window — tab shell wiring Stage 1 and Stage 2 views together."""

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QSizePolicy, QTabWidget, QLabel, QHBoxLayout,
    QVBoxLayout, QWidget, QCheckBox, QFileDialog, QMessageBox, QPushButton,
)

from desktop.branding import app_icon
from desktop.state import AppState
from desktop.views.data_view import DataView
from desktop.views.stage1_view import Stage1View
from desktop.views.stage2_view import Stage2View
from version import __version__


#: Below this the three tabs cannot lay out without clipping their controls.
MINIMUM_SIZE = (1024, 640)

#: Comfortable on a large display without swallowing the whole desktop.
PREFERRED_SIZE = (1500, 950)


def _scrollable(widget):
    """Wrap a view so oversized content scrolls instead of being cut off."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)          # expand to fill when there is room
    area.setFrameShape(QScrollArea.NoFrame)
    return area


class MainWindow(QMainWindow):
    def _size_to_screen(self) -> None:
        """Fit the window to the display it opens on.

        A fixed 1400x900 is larger than a 1366x768 laptop panel, so the window
        opened with its lower edge -- the export buttons and the status line --
        off the bottom of the screen, and on a small display there was no way
        to reach them. Take the preferred size where it fits, most of the
        available area where it does not, and maximise when even the minimum
        would overflow.
        """
        screen = QApplication.primaryScreen()
        if screen is None:                          # offscreen or headless
            self.resize(*PREFERRED_SIZE)
            return

        available = screen.availableGeometry()      # excludes the taskbar
        self.setMinimumSize(*MINIMUM_SIZE)

        if available.width() < MINIMUM_SIZE[0] or available.height() < MINIMUM_SIZE[1]:
            self.showMaximized()
            return

        width = min(PREFERRED_SIZE[0], int(available.width() * 0.9))
        height = min(PREFERRED_SIZE[1], int(available.height() * 0.9))
        self.resize(width, height)
        # Centre on the screen the user is actually looking at, rather than
        # wherever Qt would place it on a multi-monitor desktop.
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )

    def __init__(self):
        super().__init__()
        # The version belongs where a user reporting a problem will see it
        # without being asked to go looking for it.
        self.setWindowTitle(f"AI-Powered Root Cause Analysis  v{__version__}")
        self.setWindowIcon(app_icon())
        self._size_to_screen()

        self.state = AppState()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)

        # One compact row rather than two stacked blocks. The old header spent
        # ~80px of every tab largely restating the window title, on the most
        # valuable vertical space on the screen.
        header = QHBoxLayout()
        title = QLabel("Local Root Cause Analysis")
        title.setObjectName("heroTitle")
        subtitle = QLabel("— slowdowns, stalls and crashes on this machine")
        subtitle.setObjectName("heroSubtitle")
        # The header sits outside the scroll areas, so a label that cannot
        # shrink here sets the minimum width for the whole application -- this
        # one alone accounted for 624px of a 1155px floor against a declared
        # minimum of 1024. It shrinks by eliding rather than by wrapping:
        # allowing it to wrap while a trailing spacer took all the spare width
        # left it one word per line, a thin vertical column beside the title.
        subtitle.setMinimumWidth(1)
        subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header.addWidget(title)
        header.addSpacing(10)
        # The subtitle takes the slack itself, so there is no trailing spacer
        # competing with it for the same space.
        header.addWidget(subtitle, stretch=1)
        # A single, global, persisted switch for the whole app. Every raw
        # stat, ML hyperparameter and file path this app has (channel table,
        # training sliders, Granger lag, DB path) is a real answer to "what
        # exactly are you doing on my machine" -- but showing all of it by
        # default to someone who just wants to know why their PC is slow
        # buries the one sentence they came for under a spreadsheet. One
        # switch, not three, because these all answer the same underlying
        # question (am I a technical reviewer or not) and a per-tab toggle
        # risks it being on in one tab and forgotten-off in another.
        self._settings = QSettings("LocalRCA", "Desktop")
        self.advanced_toggle = QCheckBox("Advanced")
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setToolTip(
            "Show raw collected channels, ML training parameters, Granger "
            "causality settings and file paths. Off by default; the app "
            "behaves identically either way — this only changes what is "
            "shown, never what is collected or computed."
        )
        self.advanced_toggle.setChecked(
            self._settings.value("advanced_mode", False, type=bool)
        )
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)
        # In the header rather than on a tab: a user filing a bug report may
        # be on any tab, and the one that is broken may not render at all.
        # The app never sends anything itself, so this is the support path --
        # a local zip the user reads and attaches by hand.
        self.diagnostics_button = QPushButton("Export diagnostics…")
        self.diagnostics_button.setObjectName("diagnosticsButton")
        self.diagnostics_button.setToolTip(
            "Save a zip of logs, app/Windows version and collection health "
            "to attach to a bug report. Contains no collected telemetry, no "
            "database and no model; your username is redacted. Nothing is "
            "sent anywhere."
        )
        self.diagnostics_button.clicked.connect(self.export_diagnostics)
        header.addWidget(self.diagnostics_button)
        header.addSpacing(8)
        header.addWidget(self.advanced_toggle)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.stage1 = Stage1View(self.state)
        self.stage2 = Stage2View(self.state)
        self.data_view = DataView(self.state)
        # Every tab scrolls. The window declares a 1024x640 minimum it could
        # not actually render: measured, Stage 2 asks for 1168px of height and
        # the Captured Data table for 1752px of width, so at the minimum size
        # content was clipped with no way to reach it. The same arithmetic bites
        # anyone running at 150% scaling, where the effective desktop shrinks by
        # a third. Scrolling is the honest answer -- the alternative is a
        # minimum size larger than many laptop screens.
        self.tabs.addTab(_scrollable(self.data_view), "Captured Data")
        self.tabs.addTab(_scrollable(self.stage1), "1 — Baseline && Training")
        self.tabs.addTab(_scrollable(self.stage2), "2 — Run RCA Inference")
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.stage1.model_trained.connect(self._on_model_trained)

        # A model trained in an earlier session is still a usable model. Without
        # this, reopening the app left Stage 2 locked until the user retrained,
        # discarding a perfectly good artifact for no reason.
        from desktop.workers import model_path
        from pipeline import engine

        if engine.model_status(model_path()).exists:
            self.state.model_trained = True
            self.stage2.set_enabled(True)
            self.statusBar().showMessage("Existing model loaded — Stage 2 ready", 5000)

        # Apply the persisted mode once every view exists, then let the
        # toggle drive it live from here on.
        self._apply_advanced_mode(self.advanced_toggle.isChecked())

    def export_diagnostics(self) -> Path | None:
        """Ask where to save the diagnostics zip, write it, report the result."""
        from telemetry import diagnostics

        desktop = Path.home() / "Desktop"
        start = (desktop if desktop.is_dir() else Path.home()) / diagnostics.default_filename()
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export diagnostics", str(start), "Zip archive (*.zip)")
        if not chosen:
            return None                         # cancelled
        if not chosen.lower().endswith(".zip"):
            chosen += ".zip"

        # Summarising a large store reads every sample timestamp, which takes
        # a noticeable moment; a busy cursor says the click registered.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            path = diagnostics.export(chosen)
        except Exception as error:  # noqa: BLE001 - shown to the user, not swallowed
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "Export diagnostics",
                f"Could not write the diagnostics file:\n{error}")
            return None
        QApplication.restoreOverrideCursor()

        self.statusBar().showMessage(f"Diagnostics saved to {path}", 5000)
        QMessageBox.information(
            self, "Export diagnostics",
            f"Saved to:\n{path}\n\n"
            "Includes logs, app and Windows version, and collection health "
            "counts, with your username redacted. It does not include any "
            "collected telemetry, the database or the trained model. Nothing "
            "has been sent — attach the file to a bug report if you choose.")
        return path

    def _on_advanced_toggled(self, checked: bool):
        self._settings.setValue("advanced_mode", checked)
        self._apply_advanced_mode(checked)

    def _apply_advanced_mode(self, checked: bool):
        self.state.advanced_mode = checked
        self.data_view.set_advanced(checked)
        self.stage1.set_advanced(checked)
        self.stage2.set_advanced(checked)

    def _on_model_trained(self):
        self.state.model_trained = True
        self.stage2.set_enabled(True)
        self.statusBar().showMessage("Model trained — Stage 2 unlocked", 5000)
