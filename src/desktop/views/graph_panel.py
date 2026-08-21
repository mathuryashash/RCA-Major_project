"""A matplotlib figure in a native Qt canvas, with a legend and a full view.

This used to embed Plotly in a QWebEngineView, which meant shipping a browser
engine to draw two charts: 258 MB of WebEngine DLLs, 29 MB of resources, 53 MB
of translations and a 20 MB software OpenGL fallback -- about a third of the
installed application. A Qt canvas does the same job for the 28 MB matplotlib
already needs.

Two things improved as a side effect. Figures no longer round-trip through
temporary HTML files, which previously left rendered metric values in the
user's temp directory and had to be cleaned up explicitly by
`delete-all-data`. And pan and zoom now come from the standard matplotlib
toolbar rather than from JavaScript. The loss is hover tooltips, which the
edge labels and the legend largely cover.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class FigurePanel(QWidget):
    """A figure with a caption and a full-screen view.

    Both figures are detailed and unreadable in a tab a few hundred pixels
    tall, so each carries a caption explaining what is drawn and can be opened
    full screen.
    """

    #: A preferred height, not a floor. Inside a scroll area Qt sizes the page
    #: to its content's sizeHint, so two figures asking for their natural
    #: height pushed the results panel past the window and produced a second
    #: scrollbar beside the ones the tables already have. Asking for less lets
    #: the page fit on a normal display while the size policy still expands the
    #: figure to fill whatever room there is.
    PREFERRED_HEIGHT = 340

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self.PREFERRED_HEIGHT)
        return hint

    def __init__(self, parent=None, title: str = "", legend: str = ""):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._title = title
        self._legend = legend
        self._figure = None

        if title or legend:
            header = QHBoxLayout()
            caption = QLabel(legend)
            caption.setWordWrap(True)
            caption.setObjectName("figureLegend")
            header.addWidget(caption, stretch=1)
            self.expand_button = QPushButton("Full screen")
            self.expand_button.setToolTip("Open this figure full screen (Esc to close)")
            self.expand_button.setAccessibleName(f"Open {title or 'figure'} full screen")
            self.expand_button.clicked.connect(self.open_full_screen)
            header.addWidget(self.expand_button, alignment=Qt.AlignTop)
            self._layout.addLayout(header)

        self.placeholder = QLabel("")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("figurePlaceholder")
        self._layout.addWidget(self.placeholder, stretch=1)

        self.canvas = None
        self.toolbar = None

    def show_placeholder(self, message: str) -> None:
        """Fill the panel before anything has been plotted."""
        self.placeholder.setText(message)
        self.placeholder.setVisible(True)
        if self.canvas is not None:
            self.canvas.setVisible(False)
        if self.toolbar is not None:
            self.toolbar.setVisible(False)

    def show_figure(self, fig) -> None:
        """Display a figure, replacing whatever was shown before."""
        self._figure = fig
        self.placeholder.setVisible(False)

        # Rebuilt rather than re-pointed: a matplotlib Figure holds a reference
        # to exactly one canvas, so reusing the widget across figures leaves
        # the previous figure attached to a canvas that no longer draws it.
        self._detach_canvas()

        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumHeight(180)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas, stretch=1)
        self.canvas.draw_idle()

    def _detach_canvas(self) -> None:
        for widget in (self.toolbar, self.canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self.canvas = None
        self.toolbar = None

    def open_full_screen(self) -> None:
        """Show the current figure on its own, with the legend beside it."""
        if self._figure is None:
            return
        dialog = _FullScreenFigure(self._figure, self._title, self._legend, self)
        dialog.exec()
        # The dialog borrowed the figure. Re-attaching it here rather than
        # leaving a dead canvas behind is what keeps the tab usable afterwards.
        self.show_figure(self._figure)


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

        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(NavigationToolbar2QT(canvas, self))
        layout.addWidget(canvas, stretch=1)
        canvas.draw_idle()

        self.showFullScreen()
