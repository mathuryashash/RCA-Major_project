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

    def show_placeholder(self, message: str) -> None:
        """Fill the panel before anything has been plotted.

        An untouched QWebEngineView paints blank white, which against a dark
        application reads as a broken chart rather than an empty one.
        """
        if not _WEBENGINE_AVAILABLE:
            return
        self.view.setHtml(
            "<html><body style=\"margin:0;height:100vh;display:flex;"
            "align-items:center;justify-content:center;background:#151a2e;"
            "color:#7c8aa5;font-family:Inter,Segoe UI,sans-serif;font-size:14px\">"
            f"<div>{message}</div></body></html>"
        )

    def show_figure(self, fig) -> None:
        self._figure = fig
        if not _WEBENGINE_AVAILABLE:
            return
        self._render(fig, self.view)

    def _render(self, fig, view, fill: bool = False) -> None:
        """Write the figure to a temp file and point the view at it.

        ``fill`` makes the figure track the window instead of its own fixed
        height. The figures are built at 420 and 520 pixels, which is right
        inside a tab but left most of a full screen empty below them.
        """
        self._file_counter += 1
        html_path = os.path.join(self._tmp_dir, f"figure_{self._file_counter}.html")

        if fill:
            import plotly.graph_objects as go

            # Copy: the caller's figure is still on display behind the dialog
            # and must keep its own height.
            fig = go.Figure(fig)
            fig.update_layout(height=None, autosize=True, margin=dict(t=60, b=60, l=60, r=40))

        html = fig.to_html(
            include_plotlyjs=True, full_html=True,
            default_height="100%" if fill else None,
            config={"responsive": True} if fill else None,
        )
        # The page body is white by default, so any area the plot does not
        # occupy shows as a white band against the dark application.
        html = html.replace(
            "<body>", '<body style="margin:0;height:100vh;background:#151a2e">', 1,
        )
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

    def __init__(self, fig, title: str, legend: str, owner=None):
        super().__init__(owner)
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

        # Its own view, so the figure behind this dialog is left alone, but the
        # owner's temp directory: constructing another PlotlyWebView made a new
        # mkdtemp and a new atexit handler on every open, leaking a directory
        # of rendered HTML per expansion for the life of the process.
        if _WEBENGINE_AVAILABLE:
            self.view = QWebEngineView()
            layout.addWidget(self.view, stretch=1)
            owner._render(fig, self.view, fill=True)
        else:
            layout.addWidget(QLabel("QtWebEngine is not installed."), stretch=1)

        self.showFullScreen()
