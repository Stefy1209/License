from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal

from gui.components import StyledButton
from gui.styles import BG_BORDER


class MainMenuWidget(QWidget):
    """Landing screen: Calibrate / Run / Settings / Exit."""

    calibrate_requested = pyqtSignal()
    run_requested       = pyqtSignal()
    settings_requested  = pyqtSignal()
    exit_requested      = pyqtSignal()

    def __init__(self, config_path: str, hw_profile: str, parent: QWidget = None):
        super().__init__(parent)
        self._build(config_path, hw_profile)

    def _build(self, config_path: str, hw_profile: str):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.setSpacing(0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.setSpacing(4)

        title = QLabel("DEPTH VISION")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("ground detection & path planning system")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedWidth(380)
        sep.setStyleSheet(f"color: {BG_BORDER};")
        header_layout.addSpacing(16)
        header_layout.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(2)
        root.addWidget(header)
        root.addSpacing(48)

        btn_col = QWidget()
        btn_col.setFixedWidth(340)
        btn_layout = QVBoxLayout(btn_col)
        btn_layout.setSpacing(14)

        calibrate_btn = StyledButton("CALIBRATE CAMERA")
        run_btn       = StyledButton("RUN SYSTEM",  variant="success")
        settings_btn  = StyledButton("SETTINGS",    variant="warning")
        exit_btn      = StyledButton("EXIT",        variant="danger")

        for btn in (calibrate_btn, run_btn, settings_btn, exit_btn):
            btn.setMinimumHeight(48)
            btn_layout.addWidget(btn)

        calibrate_btn.clicked.connect(self.calibrate_requested)
        run_btn.clicked.connect(self.run_requested)
        settings_btn.clicked.connect(self.settings_requested)
        exit_btn.clicked.connect(self.exit_requested)

        root.addWidget(btn_col, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addStretch(2)

        footer = QLabel(f"config: {config_path}  |  profile: {hw_profile}")
        footer.setObjectName("dim")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)
        root.addSpacing(20)
