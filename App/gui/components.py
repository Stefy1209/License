from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout,
    QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QPainter, QColor


from gui.styles import ACCENT, TEXT_MAIN, BG_BORDER, FONT_FAMILY


class StyledButton(QPushButton):
    """Outlined button. Pass variant='success'|'danger'|'warning' for alternate colors."""

    def __init__(self, text: str, variant: str = "accent", parent: QWidget = None):
        super().__init__(text, parent)
        if variant != "accent":
            self.setObjectName(variant)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class NavBar(QWidget):
    """Top navigation bar with back button, title, and optional right-side widget."""

    back_clicked = pyqtSignal()

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("navbar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        self._back_btn = StyledButton("BACK")
        self._back_btn.setFixedWidth(110)
        self._back_btn.clicked.connect(self.back_clicked)
        layout.addWidget(self._back_btn)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("nav-title")
        layout.addWidget(title_lbl)

        layout.addStretch()

        self._right_slot = QHBoxLayout()
        self._right_slot.setSpacing(8)
        layout.addLayout(self._right_slot)

    def add_right_widget(self, widget: QWidget):
        self._right_slot.addWidget(widget)


class SectionHeader(QWidget):
    """Bold section label + horizontal rule."""

    def __init__(self, text: str, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 4)
        layout.setSpacing(4)

        lbl = QLabel(text)
        lbl.setObjectName("section")
        layout.addWidget(lbl)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)


class StatusBadge(QLabel):
    """Small colored label used to show run state or file presence."""

    def __init__(self, text: str = "", variant: str = "dim", parent: QWidget = None):
        super().__init__(text, parent)
        self.setObjectName(variant)
        self.setFont(QFont(FONT_FAMILY, 10))

    def set_variant(self, variant: str, text: str):
        self.setObjectName(variant)
        self.setText(text)
        self.style().unpolish(self)
        self.style().polish(self)


class ToggleSwitch(QWidget):
    """Animated sliding toggle switch."""

    toggled = pyqtSignal(bool)

    def __init__(self, label: str = "", checked: bool = True, parent: QWidget = None):
        super().__init__(parent)
        self._checked  = checked
        self._label    = label
        self._offset   = 1.0 if checked else 0.0
        self._anim     = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, val: float):
        self._offset = val
        self.update()

    offset = pyqtProperty(float, _get_offset, _set_offset)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, state: bool):
        if state != self._checked:
            self._checked = state
            self._animate(state)
            self.toggled.emit(state)

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        text_w = self.fontMetrics().horizontalAdvance(self._label)
        return QSize(46 + (12 + text_w if self._label else 0), 28)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        track_w, track_h = 42, 22
        track_x, track_y = 0, (self.height() - track_h) // 2
        radius = track_h / 2

        # Track
        track_color = QColor(ACCENT) if self._checked else QColor(BG_BORDER)
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(track_x, track_y, track_w, track_h, radius, radius)

        # Thumb
        thumb_d    = track_h - 4
        travel     = track_w - thumb_d - 4
        thumb_x    = track_x + 2 + int(self._offset * travel)
        thumb_y    = track_y + 2
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(thumb_x, thumb_y, thumb_d, thumb_d)

        # Label
        if self._label:
            p.setPen(QColor(TEXT_MAIN))
            p.setFont(self.font())
            p.drawText(track_w + 10, 0, self.width() - track_w - 10,
                       self.height(), Qt.AlignmentFlag.AlignVCenter, self._label)

        p.end()

    def mousePressEvent(self, event):
        self.setChecked(not self._checked)

    def _animate(self, state: bool):
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if state else 0.0)
        self._anim.start()
