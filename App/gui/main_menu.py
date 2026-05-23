from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal

from gui.components import StyledButton


class MainMenuWidget(QWidget):  
    calibrate_requested = pyqtSignal()
    calibration_library_requested = pyqtSignal()
    run_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    exit_requested = pyqtSignal()

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

        title = QLabel("Depth Vision")
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
        header_layout.addSpacing(16)
        header_layout.addWidget(sep, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(2)
        root.addWidget(header)
        root.addSpacing(48)

        btn_col = QWidget()
        btn_col.setFixedWidth(340)
        btn_layout = QVBoxLayout(btn_col)
        btn_layout.setSpacing(14)

        calibrate_btn = StyledButton("Calibrate Camera", icon_name="camera-photo")
        library_btn = StyledButton("Calibration Library", icon_name="view-list-details")
        run_btn = StyledButton("Run System", icon_name="media-playback-start")
        settings_btn = StyledButton("Settings", icon_name="configure")
        exit_btn = StyledButton("Exit", icon_name="application-exit")

        for btn in (calibrate_btn, library_btn, run_btn, settings_btn, exit_btn):
            btn.setMinimumHeight(48)
            btn_layout.addWidget(btn)

        calibrate_btn.clicked.connect(self.calibrate_requested)
        library_btn.clicked.connect(self.calibration_library_requested)
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
