from __future__ import annotations

# Semantic colors used directly in code (e.g. QColor() calls, dynamic stylesheets)
SUCCESS   = "#22c55e"
DANGER    = "#ef4444"
WARNING   = "#f59e0b"

# Neutral values kept for backward compatibility with calibration.py
BG_DEEP   = "#333333"   # used as QColor for RmsBar zone markers
BG_PANEL  = "transparent"
BG_CARD   = "transparent"
BG_BORDER = "#aaaaaa"   # neutral gray, used as QColor in RmsBar background
TEXT_DIM  = "gray"
TEXT_MAIN = "palette(window-text)"  # valid Qt CSS, adapts to system theme
ACCENT    = "#1976d2"               # kept for any direct QColor usage


def darken_hex(color: str, factor: float = 0.18) -> str:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = max(0, min(255, int(r * factor + (1 - factor) * 25)))
    g = max(0, min(255, int(g * factor + (1 - factor) * 25)))
    b = max(0, min(255, int(b * factor + (1 - factor) * 25)))
    return f"#{r:02x}{g:02x}{b:02x}"


# Minimal stylesheet: only semantic/functional rules, no background/font overrides.
# The native Qt platform style (Breeze on KDE, Fusion elsewhere) handles everything else.
GLOBAL_STYLESHEET = """
QWidget#navbar {
    min-height: 56px;
    max-height: 56px;
    border-bottom: 1px solid palette(mid);
}

QLabel#dim      { color: gray; font-size: 11px; }
QLabel#success  { color: #22c55e; }
QLabel#warning  { color: #f59e0b; }
QLabel#danger   { color: #ef4444; }
QLabel#section  { font-weight: bold; font-size: 12px; }
QLabel#nav-title { font-size: 14px; font-weight: bold; }
QLabel#title    { font-size: 36px; font-weight: bold; }
QLabel#subtitle { font-size: 13px; color: gray; }

QPushButton#success { color: #22c55e; }
QPushButton#danger  { color: #ef4444; }
QPushButton#warning { color: #f59e0b; }
"""
