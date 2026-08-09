"""Reusable widget that renders a Plotly figure inside a QWebEngineView.

Writes each figure to a temp HTML file (with plotly.js embedded inline —
no network access needed) and loads it via a file:// URL, since
QWebEngineView.setHtml() silently truncates content over ~2MB and a
fully self-contained Plotly export is larger than that.
"""

import atexit
import os
import shutil
import tempfile

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False


class PlotlyWebView(QWidget):
    """A Plotly figure with a legend and a full-screen view.

    Both figures are detailed and unreadable in a tab a few hundred pixels
    tall, so each carries a caption explaining what is drawn and can be
    opened full screen.
    """

    def __init__(self, parent=None, title: str = "", legend: str = ""):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._title = title
        self._legend = legend
        self._figure = None

        self._tmp_dir = tempfile.mkdtemp(prefix="rca_desktop_")
        atexit.register(shutil.rmtree, self._tmp_dir, ignore_errors=True)
        self._file_counter = 0

        if title or legend:
            header = QHBoxLayout()
            caption = QLabel(legend)
            caption.setWordWrap(True)
            caption.setObjectName("figureLegend")
            header.addWidget(caption, stretch=1)
            self.expand_button = QPushButton("Full screen")
            self.expand_button.setToolTip("Open this figure full screen (Esc to close)")
            self.expand_button.clicked.connect(self.open_full_screen)
            header.addWidget(self.expand_button, alignment=Qt.AlignTop)
            layout.addLayout(header)

        if _WEBENGINE_AVAILABLE:
            self.view = QWebEngineView()
            layout.addWidget(self.view, stretch=1)
        else:
            self.view = QLabel(
                "QtWebEngine is not installed — graph view unavailable.\n"
                "Run: pip install PySide6-Addons"
            )
            layout.addWidget(self.view, stretch=1)

    def show_figure(self, fig) -> None:
        self._figure = fig
        if not _WEBENGINE_AVAILABLE:
            return
        self._render(fig, self.view)

    def _render(self, fig, view) -> None:
        self._file_counter += 1
        html_path = os.path.join(self._tmp_dir, f"figure_{self._file_counter}.html")
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        view.setUrl(QUrl.fromLocalFile(html_path))

    def open_full_screen(self) -> None:
        """Show the current figure on its own, with the legend beside it."""
        if self._figure is None or not _WEBENGINE_AVAILABLE:
            return
        dialog = _FullScreenFigure(self._figure, self._title, self._legend, self)
        dialog.exec()


class _FullScreenFigure(QDialog):
    """The same figure, given the whole screen."""

    def __init__(self, fig, title: str, legend: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or "Figure")
        self.setObjectName("fullScreenFigure")

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        heading = QLabel(f"<b>{title}</b>")
        heading.setObjectName("figureTitle")
        header.addWidget(heading, stretch=1)
        close_button = QPushButton("✕  Close")
        close_button.setToolTip("Close (Esc)")
        close_button.clicked.connect(self.accept)
        header.addWidget(close_button)
        layout.addLayout(header)

        if legend:
            caption = QLabel(legend)
            caption.setWordWrap(True)
            caption.setObjectName("figureLegend")
            layout.addWidget(caption)

        # Its own view and temp directory: reusing the parent's would replace
        # the figure showing behind this dialog.
        self._panel = PlotlyWebView()
        self._panel.show_figure(fig)
        layout.addWidget(self._panel, stretch=1)

        self.showFullScreen()
