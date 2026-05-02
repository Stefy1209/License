from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QScrollArea, QMessageBox
from PyQt6.QtCore import QTimer, pyqtSignal, Qt

from gui.components import NavBar, StyledButton, SectionHeader


class SettingsWidget(QWidget):
    """
    Editable form for config.toml.

    Every section of the config is shown as a labeled group.
    SAVE writes the values back to disk and signals the caller to reload.
    """

    back_requested = pyqtSignal()
    saved = pyqtSignal()

    def __init__(self, cfg, config_path: str, parent: QWidget = None):
        super().__init__(parent)
        self._cfg  = cfg
        self._config_path = config_path
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        navbar = NavBar("SETTINGS")
        navbar.back_clicked.connect(self.back_requested)
        root.addWidget(navbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setObjectName("root")
        form = QVBoxLayout(inner)
        form.setContentsMargins(60, 20, 60, 40)
        form.setSpacing(0)

        cfg = self._cfg

        form.addWidget(SectionHeader("HARDWARE"))
        self._profile = _DropdownRow(form, "Profile", cfg.hardware.profile, ["nvidia", "rpi"])
        self._depth_mode = _DropdownRow(form, "Depth mode", cfg.hardware.depth_mode, ["metric", "relative"])


        form.addWidget(SectionHeader("CAMERA"))
        self._cam_id = _FieldRow(form, "Camera ID", str(cfg.camera.id), tip="device index")
        self._cam_width = _FieldRow(form, "Width (px)", str(cfg.camera.width))
        self._cam_height = _FieldRow(form, "Height (px)", str(cfg.camera.height))
        self._cam_retries = _FieldRow(form, "Max retries", str(cfg.camera.max_read_retries))

        form.addWidget(SectionHeader("CALIBRATION"))
        self._cal_file = _FieldRow(form, "Output file", cfg.calibration.file, tip=".npz path")
        self._cal_cols = _FieldRow(form, "Cols", str(cfg.calibration.cols), tip="inner corners")
        self._cal_rows = _FieldRow(form, "Rows", str(cfg.calibration.rows), tip="inner corners")
        self._cal_square = _FieldRow(form, "Square (mm)", str(cfg.calibration.square_mm))
        self._cal_frames = _FieldRow(form, "Min frames", str(cfg.calibration.min_frames))

        form.addWidget(SectionHeader("GROUND DETECTION"))
        self._gnd_seed = _FieldRow(form, "Seed region", str(cfg.ground.seed_region), tip="0-1 fraction from bottom")
        self._gnd_iter = _FieldRow(form, "RANSAC iterations", str(cfg.ground.ransac_iterations))
        self._gnd_smooth = _FieldRow(form, "Plane smoothing", str(cfg.ground.plane_smoothing), tip="EMA weight 0-1")
        self._gnd_normal = _FieldRow(form, "Normal threshold", str(cfg.ground.normal_threshold), tip="min |cos(angle)|")
        self._gnd_thr_m = _FieldRow(form, "Threshold metric", str(cfg.ground.ransac_threshold_metric), tip="metres")
        self._gnd_thr_r = _FieldRow(form, "Threshold relative", str(cfg.ground.ransac_threshold_relative), tip="fraction of range")

        form.addWidget(SectionHeader("MODEL (NVIDIA)"))
        self._mdl_id = _FieldRow(form, "Model ID", cfg.model.id, tip="HuggingFace model ID")

        form.addWidget(SectionHeader("RPi / HAILO"))
        self._rpi_hef = _FieldRow(form, "HEF path", cfg.rpi.hef_path)
        self._rpi_w = _FieldRow(form, "Model input width", str(cfg.rpi.model_input_width))
        self._rpi_h = _FieldRow(form, "Model input height", str(cfg.rpi.model_input_height))

        form.addSpacing(32)
        save_row = QWidget()
        save_layout = QHBoxLayout(save_row)
        save_layout.setContentsMargins(0, 0, 0, 0)

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("success")
        save_layout.addWidget(self._status_lbl)
        save_layout.addStretch()

        save_btn = StyledButton("SAVE SETTINGS", variant="success")
        save_btn.setMinimumWidth(200)
        save_btn.clicked.connect(self._save)
        save_layout.addWidget(save_btn)

        form.addWidget(save_row)
        form.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll)

    def _save(self):
        try:
            data = self._collect()
        except ValueError as exc:
            QMessageBox.critical(self, "Validation error", str(exc))
            return

        self._write_toml(data)
        self._status_lbl.setText("Saved!")
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.setInterval(3000)
        self._clear_timer.timeout.connect(lambda: self._status_lbl.setText("") if not self._status_lbl.isHidden() else None)
        self._clear_timer.start()
        self.saved.emit()

    def _collect(self) -> dict:
        def _int(field, name):
            try:
                return int(field.text())
            except ValueError:
                raise ValueError(f"'{name}' must be an integer.")

        def _float(field, name):
            try:
                return float(field.text())
            except ValueError:
                raise ValueError(f"'{name}' must be a number.")

        cfg = self._cfg
        return {
            "hardware": {
                "profile":    self._profile.value(),
                "depth_mode": self._depth_mode.value(),
            },
            "camera": {
                "id":               _int(self._cam_id,      "Camera ID"),
                "width":            _int(self._cam_width,    "Width"),
                "height":           _int(self._cam_height,   "Height"),
                "max_read_retries": _int(self._cam_retries,  "Max retries"),
            },
            "calibration": {
                "file":       self._cal_file.text(),
                "cols":       _int(self._cal_cols,    "Cols"),
                "rows":       _int(self._cal_rows,    "Rows"),
                "square_mm":  _float(self._cal_square, "Square mm"),
                "min_frames": _int(self._cal_frames,  "Min frames"),
            },
            "model": {
                "id": self._mdl_id.text(),
            },
            "rpi": {
                "hef_path":           self._rpi_hef.text(),
                "model_input_width":  _int(self._rpi_w, "RPi model width"),
                "model_input_height": _int(self._rpi_h, "RPi model height"),
            },
            "depth": {
                "depth_map_save_location": cfg.depth.depth_map_save_location,
            },
            "ground": {
                "seed_region":               _float(self._gnd_seed,   "Seed region"),
                "ransac_iterations":         _int(self._gnd_iter,     "RANSAC iterations"),
                "plane_smoothing":           _float(self._gnd_smooth, "Plane smoothing"),
                "normal_threshold":          _float(self._gnd_normal, "Normal threshold"),
                "ransac_threshold_metric":   _float(self._gnd_thr_m,  "Threshold metric"),
                "ransac_threshold_relative": _float(self._gnd_thr_r,  "Threshold relative"),
                "ground_map_save_location":  cfg.ground.ground_map_save_location,
            },
            "visualization": {
                "window_title":         cfg.visualization.window_title,
                "ground_overlay_alpha": cfg.visualization.ground_overlay_alpha,
                "ground_colour_bgr":    list(cfg.visualization.ground_colour_bgr),
                "colorbar_width":       cfg.visualization.colorbar_width,
            },
        }

    def _write_toml(self, data: dict):
        lines = []
        for section, fields in data.items():
            lines.append(f"\n[{section}]")
            for key, val in fields.items():
                if isinstance(val, str):
                    lines.append(f'{key} = "{val}"')
                elif isinstance(val, list):
                    lines.append(f"{key} = {val}")
                else:
                    lines.append(f"{key} = {val}")
        with open(self._config_path, "w") as f:
            f.write("\n".join(lines).lstrip() + "\n")


class _FieldRow:
    """A label + QLineEdit row, added directly to a QVBoxLayout."""

    def __init__(self, layout: QVBoxLayout, label: str, value: str, tip: str = ""):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)
        row_layout.setSpacing(16)

        lbl = QLabel(label)
        lbl.setObjectName("dim")
        lbl.setFixedWidth(200)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(lbl)

        self._edit = QLineEdit(value)
        self._edit.setFixedWidth(280)
        row_layout.addWidget(self._edit)

        if tip:
            tip_lbl = QLabel(tip)
            tip_lbl.setObjectName("dim")
            row_layout.addWidget(tip_lbl)

        row_layout.addStretch()
        layout.addWidget(row)

    def text(self) -> str:
        return self._edit.text()


class _DropdownRow:
    """A label + QComboBox row, added directly to a QVBoxLayout."""

    def __init__(self, layout: QVBoxLayout, label: str, current: str, options: list[str]):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 4, 0, 4)
        row_layout.setSpacing(16)

        lbl = QLabel(label)
        lbl.setObjectName("dim")
        lbl.setFixedWidth(200)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.addItems(options)
        idx = self._combo.findText(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        row_layout.addWidget(self._combo)
        row_layout.addStretch()

        layout.addWidget(row)

    def value(self) -> str:
        return self._combo.currentText()
