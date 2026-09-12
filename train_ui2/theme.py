"""Shared theme + compact widget factory for UI 2.0.

Everything is deliberately small: 11px base font, 22px controls, 4px grid
spacing. The old UI mixed 10/11/12px fonts with 24-32px controls and fixed
widths, which is what made elements overlap.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

ACCENT = "#4a90d9"
OK = "#4caf7d"
WARN = "#e0a33e"
ERR = "#e05c5c"
BG = "#1e1e1e"
PANEL = "#252526"
FIELD = "#2d2d30"
LINE = "#3a3a3a"
TXT = "#d4d4d4"
DIM = "#8a8a8a"

QSS = f"""
* {{ font-family: "Segoe UI", sans-serif; font-size: 11px; color: {TXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {LINE}; background: {BG}; }}
QTabBar::tab {{
    background: {PANEL}; padding: 5px 12px; margin-right: 2px;
    border: 1px solid {LINE}; border-bottom: none;
    border-top-left-radius: 3px; border-top-right-radius: 3px;
}}
QTabBar::tab:selected {{ background: {BG}; color: {ACCENT}; font-weight: bold; }}
QGroupBox {{
    border: 1px solid {LINE}; border-radius: 4px;
    margin-top: 9px; padding: 6px 6px 6px 6px; background: {PANEL};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; padding: 0 4px;
    color: {ACCENT}; font-weight: bold; font-size: 11px;
}}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px;
    padding: 1px 4px; min-height: 20px; max-height: 22px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox QAbstractItemView {{
    background: {FIELD}; border: 1px solid {LINE}; selection-background-color: {ACCENT};
}}
QPushButton {{
    background: #333337; border: 1px solid {LINE}; border-radius: 3px;
    padding: 3px 10px; min-height: 20px;
}}
QPushButton:hover {{ background: #3d3d42; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #101010; }}
QPushButton:disabled {{ color: #6a6a6a; background: #2a2a2c; }}
QPushButton#primary {{ background: {ACCENT}; color: #101010; font-weight: bold; }}
QPushButton#primary:hover {{ background: #5aa0e9; }}
QPushButton#danger {{ background: #5a2b2b; color: #f0c0c0; }}
QPushButton#danger:hover {{ background: {ERR}; color: #101010; }}
QTableWidget, QPlainTextEdit, QTextEdit {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px;
    gridline-color: {LINE}; selection-background-color: {ACCENT};
}}
QHeaderView::section {{
    background: {PANEL}; border: none; border-right: 1px solid {LINE};
    border-bottom: 1px solid {LINE}; padding: 3px 6px; color: {DIM};
}}
QCheckBox {{ spacing: 5px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {LINE};
    border-radius: 2px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QScrollBar:vertical {{ background: {BG}; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #4a4a4f; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 9px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #4a4a4f; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QProgressBar {{
    background: {FIELD}; border: 1px solid {LINE}; border-radius: 3px;
    text-align: center; height: 16px; font-size: 10px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QToolTip {{ background: {FIELD}; color: {TXT}; border: 1px solid {ACCENT}; }}
QSplitter::handle {{ background: {LINE}; }}
"""


def group(title: str, layout=None) -> QGroupBox:
    box = QGroupBox(title)
    lay = layout or QGridLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(6, 6, 6, 6)
    box.setLayout(lay)
    return box


def label(text: str, color: str = TXT, bold: bool = False, size: int = 11, word_wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    weight = "bold" if bold else "normal"
    lbl.setStyleSheet(
        f"color:{color}; font-size:{size}px; font-weight:{weight}; background:transparent;")
    if word_wrap:
        lbl.setWordWrap(True)
    return lbl


def field_label(text: str, tooltip: str = "") -> QLabel:
    """Right-aligned compact caption for a control row."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{DIM}; background:transparent;")
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lbl.setMinimumWidth(78)
    lbl.setMaximumWidth(120)
    if tooltip:
        lbl.setToolTip(tooltip)
    return lbl


def spin(spec_min: float, spec_max: float, value: float, is_int: bool,
         step: float = 1.0, decimals: int = 6, tooltip: str = ""):
    if is_int:
        w = QSpinBox()
        w.setRange(int(spec_min), int(spec_max))
        w.setSingleStep(max(1, int(step)))
        w.setValue(int(value))
    else:
        w = QDoubleSpinBox()
        w.setRange(float(spec_min), float(spec_max))
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(float(value))
    w.setFixedWidth(92)
    w.setToolTip(tooltip)
    return w


def combo(items: list, current=None, tooltip: str = "") -> QComboBox:
    c = QComboBox()
    c.addItems([str(i) for i in items])
    c.setFixedWidth(92)
    if current is not None:
        idx = c.findText(str(current))
        if idx >= 0:
            c.setCurrentIndex(idx)
    if tooltip:
        c.setToolTip(tooltip)
    return c


def check(text: str, checked: bool = False, tooltip: str = "") -> QCheckBox:
    c = QCheckBox(text)
    c.setChecked(checked)
    if tooltip:
        c.setToolTip(tooltip)
    return c


def button(text: str, slot=None, kind: str = "", tooltip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    if tooltip:
        b.setToolTip(tooltip)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{LINE}; background:{LINE}; max-height:1px;")
    f.setFixedHeight(1)
    return f


def stack(*widgets: QWidget, spacing: int = 4, margin: int = 0) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setSpacing(spacing)
    lay.setContentsMargins(margin, margin, margin, margin)
    for x in widgets:
        lay.addWidget(x)
    return w


def decimals_for(min_value: float) -> int:
    """Enough decimals to represent `min_value` and one order below it."""
    import math
    if min_value <= 0:
        return 4
    return max(0, min(8, int(math.ceil(-math.log10(min_value))) + 1))


def color_for(name: str) -> QColor:
    palette = {
        "fps": QColor(OK), "return": QColor(ACCENT), "policy_loss": QColor("#c586c0"),
        "value_loss": QColor(WARN), "entropy": QColor("#9cdcfe"), "kl": QColor(ERR),
        "days": QColor(OK), "bases": QColor(ACCENT), "people": QColor("#dcdcaa"),
        "score": QColor("#c586c0"),
    }
    return palette.get(name, QColor(TXT))
