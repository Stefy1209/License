from __future__ import annotations

import time
import threading
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from gui.components import NavBar, StyledButton, ToggleSwitch


class SystemViewWidget(QWidget):
    """
    Live system view.

    Inference runs in a dedicated background thread. Results are delivered
    to the GUI thread via a Qt signal so the UI never freezes.
    """

    back_requested = pyqtSignal()
    _result_signal = pyqtSignal(object)

    _FPS_SMOOTH = 0.2

    def __init__(self, cfg, parent: QWidget = None):
        super().__init__(parent)
        self._cfg = cfg
        self._pipeline = None
        self._running = False
        self._last_result = None
        self._fps: Optional[float] = None
        self._last_tick: Optional[float] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._result_signal.connect(self._on_result)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        navbar = NavBar("LIVE SYSTEM")
        navbar.back_clicked.connect(self._on_back)

        self._start_btn = StyledButton("START", variant="success")
        self._stop_btn  = StyledButton("STOP", variant="danger")
        self._start_btn.setFixedWidth(120)
        self._stop_btn.setFixedWidth(120)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        navbar.add_right_widget(self._start_btn)
        navbar.add_right_widget(self._stop_btn)

        root.addWidget(navbar)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        content_layout.addWidget(self._build_sidebar())
        content_layout.addWidget(self._build_canvas(), stretch=1)

        root.addWidget(content)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("panel")
        sidebar.setFixedWidth(320)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        view_hdr = QLabel("VIEW OPTIONS")
        view_hdr.setObjectName("section")
        layout.addWidget(view_hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addSpacing(4)

        self._depth_toggle = ToggleSwitch("Depth Map", checked=True)
        self._ground_toggle = ToggleSwitch("Show ground overlay", checked=True)
        self._path_toggle = ToggleSwitch("Show path", checked=True)

        layout.addWidget(self._depth_toggle)
        layout.addWidget(self._ground_toggle)
        layout.addWidget(self._path_toggle)

        layout.addSpacing(12)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        layout.addSpacing(8)

        self._status_lbl = QLabel("IDLE")
        self._status_lbl.setObjectName("dim")
        layout.addWidget(self._status_lbl)

        self._plane_lbl = QLabel("plane: —")
        self._plane_lbl.setObjectName("dim")
        self._plane_lbl.setWordWrap(True)
        layout.addWidget(self._plane_lbl)

        self._fps_lbl = QLabel("fps: —")
        self._fps_lbl.setObjectName("dim")
        layout.addWidget(self._fps_lbl)

        layout.addStretch()
        return sidebar

    def _build_canvas(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("card")
        wrapper.setStyleSheet("background-color: #000000;")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)

        self._frame_lbl = QLabel()
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._frame_lbl)

        self._placeholder = QLabel("Press  START  to begin")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setObjectName("dim")
        self._placeholder.setStyleSheet("font-size: 16px;")
        layout.addWidget(self._placeholder)

        return wrapper

    def _start(self):
        if self._running:
            return
        self._placeholder.hide()
        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_lbl.setText("LOADING...")
        self._status_lbl.setObjectName("warning")
        self._refresh_label_style(self._status_lbl)

        from hardware import HardwareProfile
        from pipeline import DepthPipeline

        hw = HardwareProfile.from_config(self._cfg)
        self._pipeline = DepthPipeline(self._cfg, hw)

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        self._inference_thread.start()

    def _stop(self):
        if not self._running:
            return
        self._running = False
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._inference_thread:
            self._inference_thread.join(timeout=3.0)
            self._inference_thread = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._frame_lbl.clear()
        self._placeholder.show()
        self._status_lbl.setText("IDLE")
        self._status_lbl.setObjectName("dim")
        self._refresh_label_style(self._status_lbl)
        self._plane_lbl.setText("plane: —")
        self._fps_lbl.setText("fps: —")

    def _on_back(self):
        self._stop()
        self.back_requested.emit()

    def _inference_loop(self):
        """Runs entirely in a background thread — loads model, then loops frames."""
        pipeline = self._pipeline
        if pipeline is None:
            return

        pipeline.load_calibration()
        pipeline.load_model()
        pipeline.start_capture()

        while self._running and pipeline is not None and not pipeline.is_stopped():
            result = pipeline.process_next_frame(timeout=0.5)
            if result is not None:
                self._result_signal.emit(result)

    def _on_result(self, result):
        """Called on the GUI thread when a new frame result is ready."""
        if not self._running:
            return

        self._last_result = result
        self._update_fps()
        self._render(result)
        self._update_sidebar(result)

    def _update_fps(self):
        now = time.perf_counter()
        if self._last_tick is not None:
            dt = now - self._last_tick
            instant = 1.0 / dt if dt > 0 else 0.0
            self._fps = (instant if self._fps is None
                         else self._FPS_SMOOTH * instant + (1 - self._FPS_SMOOTH) * self._fps)
        self._last_tick = now

    def _render(self, result):
        from visualization import visualize_depth

        show_depth = self._depth_toggle.isChecked()
        show_ground = self._ground_toggle.isChecked()
        show_path = self._path_toggle.isChecked()
        cfg = self._cfg

        if show_depth and result.depth_map is not None:
            frame, _, _ = visualize_depth(result.depth_map)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = cv2.cvtColor(result.rgb_frame, cv2.COLOR_BGR2RGB)

        if show_ground and result.ground_mask is not None:
            colour_rgb = tuple(reversed(cfg.visualization.ground_colour_bgr))
            frame = _overlay_ground(frame, result.ground_mask, colour_rgb,
                                    cfg.visualization.ground_overlay_alpha)

        if show_path and result.path is not None and len(result.path) >= 2:
            frame = _overlay_path(frame, result.path, result.start_point, result.end_point)

        _draw_status_bar(frame, result.plane)

        available = self._frame_lbl.size()
        ih, iw = frame.shape[:2]
        aw, ah = available.width(), available.height()
        if aw > 1 and ah > 1:
            scale = min(aw / iw, ah / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._frame_lbl.setPixmap(QPixmap.fromImage(qt_image))

    def _update_sidebar(self, result):
        self._status_lbl.setText("RUNNING")
        self._status_lbl.setObjectName("success")
        self._refresh_label_style(self._status_lbl)

        if result.plane is not None:
            a, b, c, d = result.plane
            self._plane_lbl.setText(f"plane:\n[{a:+.2f}, {b:+.2f},\n {c:+.2f}, {d:+.2f}]")
            self._plane_lbl.setObjectName("success")
        else:
            self._plane_lbl.setText("plane: —")
            self._plane_lbl.setObjectName("dim")
        self._refresh_label_style(self._plane_lbl)

        if self._fps is not None:
            self._fps_lbl.setText(f"fps: {self._fps:.1f}")

    @staticmethod
    def _refresh_label_style(lbl: QLabel):
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

    def on_close(self):
        self._stop()


def _overlay_ground(frame, mask, colour_rgb, alpha):
    overlay = frame.copy()
    overlay[mask] = colour_rgb
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, colour_rgb, thickness=2)
    return blended


def _overlay_path(frame, path, start, end):
    out = frame.copy()
    if len(path) >= 2:
        for i in range(len(path) - 1):
            cv2.line(out,
                     (int(path[i, 1]),     int(path[i, 0])),
                     (int(path[i + 1, 1]), int(path[i + 1, 0])),
                     (255, 255, 0), 2, cv2.LINE_AA)
    if start is not None:
        cv2.circle(out, (int(start[1]), int(start[0])), 6, (0, 255, 0), -1)
        cv2.circle(out, (int(start[1]), int(start[0])), 6, (0,   0, 0),  1)
    if end is not None:
        cv2.circle(out, (int(end[1]), int(end[0])), 6, (255, 0, 0), -1)
        cv2.circle(out, (int(end[1]), int(end[0])), 6, (0,   0, 0),  1)
    return out


def _draw_status_bar(frame, plane):
    if plane is None:
        text, colour = "Ground plane: NOT DETECTED", (220, 40, 40)
    else:
        a, b, c, d = plane
        text, colour = f"Ground: [{a:+.2f}, {b:+.2f}, {c:+.2f}, {d:+.2f}]", (40, 220, 80)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(frame, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)