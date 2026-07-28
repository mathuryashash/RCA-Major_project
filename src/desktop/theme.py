"""Dark theme QSS for the desktop application."""

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0f1628;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 10.5pt;
}

QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: #151a2e;
    border-radius: 8px;
}

QTabBar::tab {
    background: #1a1f3a;
    color: #a0aec0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabBar::tab:selected {
    background: rgba(102, 126, 234, 0.18);
    color: #e2e8f0;
    border-bottom: 2px solid #667eea;
}

QGroupBox {
    background-color: rgba(30, 33, 48, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-top: 1.2em;
    padding: 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #e2e8f0;
}

QPushButton {
    background: #667eea;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}

QPushButton:hover { background: #7688ee; }
QPushButton:pressed { background: #5568d3; }
QPushButton:disabled { background: #2a2f47; color: #6b7280; }

QPushButton#primaryAction {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
    padding: 10px 28px;
    font-size: 11pt;
}

QSlider::groove:horizontal {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #667eea;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QProgressBar {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    text-align: center;
    color: #e2e8f0;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #f093fb);
    border-radius: 6px;
}

QTableWidget {
    background-color: #151a2e;
    gridline-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
}

QHeaderView::section {
    background-color: #1a1f3a;
    color: #a0aec0;
    padding: 6px;
    border: none;
    font-weight: 600;
}

QPlainTextEdit {
    background-color: #0b0f1c;
    color: #7bed9f;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    font-family: Consolas, monospace;
}

QStatusBar {
    background: #1a1f3a;
    color: #a0aec0;
}

QLabel#heroTitle {
    font-size: 20pt;
    font-weight: 800;
    color: #e2e8f0;
    padding: 6px 0;
}

QLabel#heroSubtitle {
    color: #a0aec0;
    font-size: 10pt;
    padding-bottom: 8px;
}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(DARK_QSS)
