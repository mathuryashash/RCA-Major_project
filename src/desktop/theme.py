"""Dark theme QSS for the desktop application.

One accent, used to mean something. Cyan marks anything the user can act on;
amber and red are reserved for warnings and failures, so colour carries state
rather than decoration. The previous indigo-to-purple gradient competed with a
green terminal console, which left two unrelated colour stories on one screen.

Fonts are deliberately stock. A distinctive display face would need bundling
into the PyInstaller build, and a missing-font fallback is a poor trade for a
tool whose value is the data on screen.
"""

# --- palette -----------------------------------------------------------------
BG          = "#0d1117"   # app background
SURFACE     = "#161b22"   # panes, tables
SURFACE_ALT = "#1c2128"   # group boxes, headers
# 0.35, not 0.10. Composited over SURFACE this is #62686e at 3.07:1, which
# clears WCAG SC 1.4.11 for non-text contrast; the old value rendered at
# 1.32:1, so every frame, input outline and table border in the application
# was very nearly invisible. Text contrast was never the problem here -- body
# text sits at 14.64:1 -- it was the structure around it.
BORDER      = "rgba(240, 246, 252, 0.35)"
# Cell separators in a data table carry meaning, so they get most of the way
# there too. Row striping below is decorative and deliberately does not claim
# compliance: a table is readable without it.
GRIDLINE    = "rgba(240, 246, 252, 0.22)"
ROW_ALT     = "#232a33"
TEXT        = "#e6edf3"
TEXT_MUTED  = "#8b949e"

ACCENT      = "#2dd4bf"   # actionable
ACCENT_DIM  = "#14b8a6"
ACCENT_DEEP = "#0f766e"
WARNING     = "#f59e0b"
DANGER      = "#f87171"

RADIUS = "8px"

DARK_QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 11pt;
}}

/* Tabs read as one surface with their pane: no pane border, and the selected
   tab shares the pane's background so the two visually connect. */
QTabWidget::pane {{
    border: none;
    background-color: {SURFACE};
    border-top-right-radius: {RADIUS};
    border-bottom-left-radius: {RADIUS};
    border-bottom-right-radius: {RADIUS};
}}

QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 9px 22px;
    margin-right: 2px;
    border-top-left-radius: {RADIUS};
    border-top-right-radius: {RADIUS};
    font-weight: 600;
    letter-spacing: 0.4px;
}}

QTabBar::tab:hover {{ color: {TEXT}; }}

QTabBar::tab:selected {{
    background: {SURFACE};
    color: {ACCENT};
    border-top: 2px solid {ACCENT};
}}

QGroupBox {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    margin-top: 1.2em;
    padding: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    font-size: 9.5pt;
}}

/* Every action button shares the accent; nothing flat sits beside something
   gradient-filled with no difference in meaning. */
QPushButton {{
    background: {ACCENT_DEEP};
    color: {TEXT};
    border: 1px solid {ACCENT_DIM};
    border-radius: {RADIUS};
    padding: 8px 20px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}

QPushButton:hover  {{ background: {ACCENT_DIM}; color: #06201d; }}
QPushButton:pressed{{ background: {ACCENT}; color: #06201d; }}
QPushButton:disabled {{
    background: transparent;
    color: #57606a;
    border: 1px solid rgba(240, 246, 252, 0.08);
}}

QPushButton#primaryAction {{
    background: {ACCENT};
    color: #06201d;
    border: none;
    padding: 11px 28px;
    font-size: 11pt;
}}
QPushButton#primaryAction:hover   {{ background: #5eead4; }}
QPushButton#primaryAction:disabled {{
    background: transparent;
    color: #57606a;
    border: 1px solid rgba(240, 246, 252, 0.08);
}}

QSlider::groove:horizontal {{
    height: 5px;
    background: rgba(240, 246, 252, 0.10);
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 15px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSpinBox, QDateTimeEdit, QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {TEXT};
}}
QSpinBox:focus, QDateTimeEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT_DIM}; }}

QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: rgba(240, 246, 252, 0.05);
    text-align: center;
    color: {TEXT};
    height: 18px;
}}

QProgressBar::chunk {{
    background: {ACCENT_DIM};
    border-radius: 5px;
}}

QTableWidget {{
    background-color: {SURFACE};
    alternate-background-color: {ROW_ALT};
    gridline-color: {GRIDLINE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
}}

QTableWidget::item:selected {{
    background: {ACCENT_DEEP};
    color: {TEXT};
}}

/* Separates header from body; without it the two surfaces ran together. */
QHeaderView::section {{
    background-color: {SURFACE_ALT};
    color: {TEXT_MUTED};
    padding: 7px 6px;
    border: none;
    border-bottom: 1px solid {ACCENT_DEEP};
    font-weight: 600;
    letter-spacing: 0.4px;
}}

/* A log pane, not a retro terminal. Green-on-black told a colour story the
   rest of the app did not share. */
QPlainTextEdit {{
    background-color: {BG};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: {RADIUS};
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 10pt;
    padding: 6px;
}}

QStatusBar {{
    background: {SURFACE_ALT};
    color: {TEXT_MUTED};
}}

/* Compact header: the old 20pt title plus subtitle consumed ~80px of every
   tab, which is the most valuable vertical space on the screen. */
QLabel#heroTitle {{
    font-size: 14pt;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: 0.3px;
    padding: 0;
}}

QLabel#heroSubtitle {{
    color: {TEXT_MUTED};
    font-size: 9.5pt;
    padding: 0 0 4px 0;
}}

QLabel#warningBanner {{
    color: {WARNING};
    font-weight: 600;
}}

QLabel#errorText {{ color: {DANGER}; }}

/* The verdict of an analysis, stated once at the size it deserves.
   Before this the finding lived only in the Report tab -- fourth of four, raw
   markdown, in the smallest and lowest-contrast style in this sheet -- while
   the tab that opens by default showed four-decimal scores. The application
   was honest in its report and confident in its interface, and the interface
   is what gets read. */
QLabel#verdictSupported, QLabel#verdictCorrelation, QLabel#verdictUntested {{
    font-size: 16px;
    font-weight: 600;
    padding: 12px 14px;
    border-radius: 8px;
}}
QLabel#verdictSupported {{
    color: {TEXT};
    background: rgba(45, 212, 191, 0.10);
    border-left: 4px solid {ACCENT};
}}
/* Warning-coloured, not danger-coloured: declining to make a causal claim is
   correct behaviour, not a fault. */
QLabel#verdictCorrelation {{
    color: {TEXT};
    background: rgba(245, 158, 11, 0.10);
    border-left: 4px solid {WARNING};
}}
QLabel#verdictUntested {{
    color: {TEXT};
    background: rgba(139, 148, 158, 0.10);
    border-left: 4px solid {TEXT_MUTED};
}}

/* Every button carried a stylesheet background, which stops Qt drawing the
   native focus rectangle -- so Tab moved an invisible cursor across Run RCA,
   Train, Find Incidents and both consent buttons. WCAG 2.4.7 failure.

   The ring costs a pixel on each edge over the base 1px border and the
   padding gives it back. Qt does not re-run sizeFromContents on a
   pseudo-state change, so this is belt and braces rather than load-bearing --
   but wrong numbers would shrink the button by 4px the day that changes. */
QPushButton:focus {{
    border: 2px solid {ACCENT};
    padding: 7px 19px;
}}
QPushButton#primaryAction:focus {{
    border: 2px solid {TEXT};
}}
QTabBar::tab:focus {{ border: 2px solid {ACCENT}; }}
QTableWidget:focus, QPlainTextEdit:focus {{ border: 1px solid {ACCENT}; }}

/* The two controls the button pass missed: measured at 0 changed pixels
   between focused and blurred, against 1573 on a QPlainTextEdit. The checkbox
   is the Event Log opt-in, which makes it the worst widget here to leave
   unindicated -- a keyboard user could not see which control they were about
   to toggle to store message text.

   The transparent border at rest is what keeps the geometry stable: adding a
   border only on focus would shift the groove and the label by two pixels
   every time focus arrived. Verified: sizeHint is identical in both states. */
QCheckBox {{
    border: 2px solid transparent;
    border-radius: 6px;
    padding: 2px;
}}
QCheckBox:focus {{ border-color: {ACCENT}; }}

QSlider {{
    border: 2px solid transparent;
    border-radius: 6px;
}}
QSlider:focus {{ border-color: {ACCENT}; }}
"""


def apply_theme(app) -> None:
    app.setStyleSheet(DARK_QSS)
