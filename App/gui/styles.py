from __future__ import annotations

BG_DEEP   = "#0a0a0f"
BG_PANEL  = "#12121a"
BG_CARD   = "#1a1a26"
BG_BORDER = "#2a2a3d"
ACCENT    = "#00e5ff"
TEXT_DIM  = "#6b7280"
TEXT_MAIN = "#e2e8f0"
DANGER    = "#ef4444"
SUCCESS   = "#22c55e"
WARNING   = "#f59e0b"

FONT_FAMILY = "Courier New"


def darken_hex(color: str, factor: float = 0.18) -> str:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = max(0, min(255, int(r * factor + (1 - factor) * 25)))
    g = max(0, min(255, int(g * factor + (1 - factor) * 25)))
    b = max(0, min(255, int(b * factor + (1 - factor) * 25)))
    return f"#{r:02x}{g:02x}{b:02x}"


GLOBAL_STYLESHEET = f"""
* {{
    font-family: '{FONT_FAMILY}';
    color: {TEXT_MAIN};
}}

QMainWindow, QWidget#root {{
    background-color: {BG_DEEP};
}}

QWidget#panel {{
    background-color: {BG_PANEL};
}}

QWidget#card {{
    background-color: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-radius: 4px;
}}

/* ── Navbar ── */
QWidget#navbar {{
    background-color: {BG_PANEL};
    border-bottom: 1px solid {BG_BORDER};
    min-height: 56px;
    max-height: 56px;
}}

/* ── Outlined buttons ── */
QPushButton {{
    background-color: transparent;
    border: 1px solid {ACCENT};
    color: {ACCENT};
    font-family: '{FONT_FAMILY}';
    font-size: 13px;
    font-weight: bold;
    padding: 8px 22px;
    border-radius: 2px;
    min-height: 36px;
}}
QPushButton:hover  {{ background-color: {darken_hex(ACCENT)}; }}
QPushButton:pressed{{ background-color: {darken_hex(ACCENT, 0.05)}; }}
QPushButton:disabled {{ border-color: {BG_BORDER}; color: {TEXT_DIM}; }}

QPushButton#success {{
    border-color: {SUCCESS}; color: {SUCCESS};
}}
QPushButton#success:hover  {{ background-color: {darken_hex(SUCCESS)}; }}

QPushButton#danger {{
    border-color: {DANGER}; color: {DANGER};
}}
QPushButton#danger:hover  {{ background-color: {darken_hex(DANGER)}; }}

QPushButton#warning {{
    border-color: {WARNING}; color: {WARNING};
}}
QPushButton#warning:hover  {{ background-color: {darken_hex(WARNING)}; }}

/* ── Labels ── */
QLabel {{
    background: transparent;
    font-size: 12px;
}}
QLabel#title {{
    font-size: 42px;
    font-weight: bold;
    color: {ACCENT};
}}
QLabel#subtitle {{
    font-size: 13px;
    color: {TEXT_DIM};
}}
QLabel#section {{
    font-size: 12px;
    font-weight: bold;
    color: {ACCENT};
}}
QLabel#nav-title {{
    font-size: 14px;
    font-weight: bold;
    color: {ACCENT};
}}
QLabel#dim {{
    color: {TEXT_DIM};
    font-size: 11px;
}}
QLabel#success {{
    color: {SUCCESS};
}}
QLabel#warning {{
    color: {WARNING};
}}
QLabel#danger {{
    color: {DANGER};
}}

/* ── Inputs ── */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_CARD};
    border: 1px solid {BG_BORDER};
    color: {TEXT_MAIN};
    font-size: 11px;
    padding: 5px 8px;
    border-radius: 2px;
    min-height: 28px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_BORDER};
    width: 16px;
    border: none;
}}

/* ── ComboBox ── */
QComboBox {{
    background-color: {BG_CARD};
    border: 1px solid {BG_BORDER};
    color: {TEXT_MAIN};
    font-size: 11px;
    padding: 5px 8px;
    border-radius: 2px;
    min-height: 28px;
    min-width: 160px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BG_BORDER};
    color: {TEXT_MAIN};
    selection-background-color: {darken_hex(ACCENT)};
}}

/* ── ScrollArea ── */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BG_BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Toggle switch ── */
QCheckBox {{
    color: {TEXT_MAIN};
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 38px;
    height: 20px;
    border-radius: 10px;
    background-color: {BG_BORDER};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
}}

/* ── Separator ── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {BG_BORDER};
}}

/* ── MessageBox ── */
QMessageBox {{
    background-color: {BG_CARD};
}}
QMessageBox QLabel {{
    color: {TEXT_MAIN};
    font-size: 12px;
}}
"""
