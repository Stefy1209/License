from __future__ import annotations

import sys
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QApplication

from gui.styles import GLOBAL_STYLESHEET
from gui.main_menu import MainMenuWidget
from gui.calibration import CalibrationWidget
from gui.settings import SettingsWidget
from gui.system_view import SystemViewWidget


class MainWindow(QMainWindow):
    """
    Root window.

    Uses a QStackedWidget to swap between the four screens without
    destroying and recreating widgets unnecessarily.
    """

    def __init__(self, cfg, config_path: str):
        super().__init__()
        self._cfg         = cfg
        self._config_path = config_path

        self.setWindowTitle("DEPTH VISION")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._show_main()

    def _clear_stack(self):
        while self._stack.count():
            w = self._stack.widget(0)
            if hasattr(w, "on_close"):
                w.on_close()
            self._stack.removeWidget(w)
            w.deleteLater()

    def _push(self, widget):
        self._clear_stack()
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)

    # Screen factories

    def _show_main(self):
        w = MainMenuWidget(self._config_path, self._cfg.hardware.profile)
        w.calibrate_requested.connect(self._show_calibration)
        w.run_requested.connect(self._show_system)
        w.settings_requested.connect(self._show_settings)
        w.exit_requested.connect(self.close)
        self._push(w)

    def _show_calibration(self):
        w = CalibrationWidget(self._cfg)
        w.back_requested.connect(self._show_main)
        self._push(w)

    def _show_settings(self):
        w = SettingsWidget(self._cfg, self._config_path)
        w.back_requested.connect(self._show_main)
        w.saved.connect(self._reload_config)
        self._push(w)

    def _show_system(self):
        w = SystemViewWidget(self._cfg)
        w.back_requested.connect(self._show_main)
        self._push(w)

    def _reload_config(self):
        from config import AppConfig
        self._cfg = AppConfig.load(self._config_path)

    def closeEvent(self, event):
        current = self._stack.currentWidget()
        if hasattr(current, "on_close"):
            current.on_close()
        event.accept()


def run(config_path: str = "config.toml") -> None:
    """Create the QApplication, apply the stylesheet, and launch the window."""
    from config import AppConfig

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)

    cfg = AppConfig.load(config_path)
    window = MainWindow(cfg, config_path)
    window.show()

    sys.exit(app.exec())
