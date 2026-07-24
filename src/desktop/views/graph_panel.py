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

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False


class PlotlyWebView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tmp_dir = tempfile.mkdtemp(prefix="rca_desktop_")
        atexit.register(shutil.rmtree, self._tmp_dir, ignore_errors=True)
        self._file_counter = 0

        if _WEBENGINE_AVAILABLE:
            self.view = QWebEngineView()
            layout.addWidget(self.view)
        else:
            from PySide6.QtWidgets import QLabel
            self.view = QLabel(
                "QtWebEngine is not installed — graph view unavailable.\n"
                "Run: pip install PySide6-Addons"
            )
            layout.addWidget(self.view)

    def show_figure(self, fig) -> None:
        if not _WEBENGINE_AVAILABLE:
            return
        self._file_counter += 1
        html_path = os.path.join(self._tmp_dir, f"figure_{self._file_counter}.html")
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        self.view.setUrl(QUrl.fromLocalFile(html_path))
