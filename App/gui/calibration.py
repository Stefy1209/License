from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QDialog, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor

from gui.components import NavBar, StyledButton, StatusBadge
from gui.styles import SUCCESS, WARNING, DANGER, TEXT_DIM, BG_DEEP, BG_PANEL, BG_BORDER, TEXT_MAIN


def _rms_verdict(rms: float) -> tuple[str, str]:
    """Return (label, object_name) describing the RMS quality."""
    if rms < 0.5:
        return "Excellent — ready to use.", "success"
    if rms < 1.0:
        return "Acceptable — good enough for most uses.", "warning"
    return "Poor — consider recalibrating with more varied poses.", "danger"


class CalibrationWidget(QWidget):
    """
    Camera calibration screen.

    Shows whether a calibration file already exists, displays the RMS
    score from the last run, and lets the user launch a new capture.
    """

    back_requested = pyqtSignal()

    def __init__(self, cfg, parent: QWidget = None):
        super().__init__(parent)
        self._cfg = cfg

        # Root layout is created once and never recreated.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._navbar = NavBar("CAMERA CALIBRATION")
        self._navbar.back_clicked.connect(self.back_requested)
        root.addWidget(self._navbar)

        # _body is a plain container whose contents are swapped on refresh.
        self._body = QWidget()
        root.addWidget(self._body)

        self._refresh()

    def _refresh(self):
        """Replace the body widget entirely to reflect the current calibration state."""
        root_layout = self.layout()

        # Remove and delete the old body widget.
        if self._body is not None:
            root_layout.removeWidget(self._body)
            self._body.deleteLater()

        # Create a fresh body widget.
        self._body = QWidget()
        root_layout.addWidget(self._body)

        # Update the navbar badge.
        file_exists = os.path.exists(self._cfg.calibration.file)
        status_text = "Calibration file present" if file_exists else "No calibration file found"
        status_var  = "success" if file_exists else "warning"

        while self._navbar._right_slot.count():
            item = self._navbar._right_slot.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._navbar.add_right_widget(StatusBadge(status_text, variant=status_var))

        # Populate the new body.
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(60, 40, 60, 40)
        body_layout.setSpacing(16)

        if file_exists:
            header = QLabel("A calibration file already exists.")
            header.setObjectName("success")
            header.setStyleSheet("font-size: 15px; font-weight: bold;")
            body_layout.addWidget(header)

            note = QLabel("You can recalibrate at any time. The new file will overwrite the existing one.")
            note.setObjectName("dim")
            body_layout.addWidget(note)

            rms = self._load_rms()
            if rms is not None:
                body_layout.addWidget(_RmsCard(rms))
        else:
            note = QLabel(
                "No calibration file was found at the configured path.\n"
                "Run calibration below to generate one before using the system."
            )
            note.setObjectName("warning")
            body_layout.addWidget(note)

        body_layout.addSpacing(8)

        body_layout.addWidget(_InfoGrid([
            ("Calibration file", self._cfg.calibration.file),
            ("Checkerboard cols", f"{self._cfg.calibration.cols} inner corners"),
            ("Checkerboard rows", f"{self._cfg.calibration.rows} inner corners"),
            ("Square length", f"{self._cfg.calibration.square_mm} mm"),
            ("Frames required", str(self._cfg.calibration.min_frames)),
            ("Camera index", str(self._cfg.camera.id)),
        ]))

        body_layout.addSpacing(16)

        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)

        launch_btn = StyledButton("START CALIBRATION")
        launch_btn.setMinimumWidth(220)
        launch_btn.clicked.connect(self._launch)
        ctrl_layout.addWidget(launch_btn)
        ctrl_layout.addStretch()

        body_layout.addWidget(ctrl)
        body_layout.addStretch()

    def _load_rms(self) -> Optional[float]:
        try:
            with np.load(self._cfg.calibration.file) as data:
                rms = data.get("rms")
                return float(rms) if rms is not None else None
        except Exception:
            return None

    def _launch(self):
        cfg = self._cfg
        dialog = _CalibrationDialog(
            camera_id  = cfg.camera.id,
            out_path = cfg.calibration.file,
            cols = cfg.calibration.cols,
            rows = cfg.calibration.rows,
            square_mm = cfg.calibration.square_mm,
            min_frames = cfg.calibration.min_frames,
            parent = self,
        )
        dialog.exec()
        self._refresh()


class _RmsCard(QWidget):
    """Displays the RMS reprojection error with a quality verdict and bar."""

    def __init__(self, rms: float, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(10)

        verdict_text, verdict_style = _rms_verdict(rms)

        title = QLabel("Last calibration result")
        title.setObjectName("section")
        layout.addWidget(title)

        score_row = QWidget()
        score_layout = QHBoxLayout(score_row)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(16)

        rms_lbl = QLabel(f"RMS reprojection error:  {rms:.4f} px")
        rms_lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_MAIN};")
        score_layout.addWidget(rms_lbl)
        score_layout.addStretch()

        layout.addWidget(score_row)

        bar = _RmsBar(rms)
        layout.addWidget(bar)

        verdict = QLabel(verdict_text)
        verdict.setObjectName(verdict_style)
        layout.addWidget(verdict)

        if rms >= 1.0:
            tip = QLabel(
                "Tips: vary the board angle more, cover corners and edges of the frame,\n"
                "ensure the board is flat and well-lit, and avoid motion blur."
            )
            tip.setObjectName("dim")
            tip.setWordWrap(True)
            layout.addWidget(tip)


class _RmsBar(QWidget):
    """Visual 0-2 px progress bar coloured by RMS quality zones."""

    _MAX = 2.0

    def __init__(self, rms: float, parent: QWidget = None):
        super().__init__(parent)
        self._rms = min(rms, self._MAX)
        self.setFixedHeight(20)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        r = 4

        # Background track
        p.setBrush(QColor(BG_BORDER))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, w, h, r, r)

        # Filled portion
        fill_w = int((self._rms / self._MAX) * w)
        if fill_w > 0:
            if self._rms < 0.5:
                color = QColor(SUCCESS)
            elif self._rms < 1.0:
                color = QColor(WARNING)
            else:
                color = QColor(DANGER)
            p.setBrush(color)
            p.drawRoundedRect(0, 0, fill_w, h, r, r)

        # Zone markers at 0.5 and 1.0
        p.setPen(QColor(BG_DEEP))
        for threshold in (0.5, 1.0):
            x = int((threshold / self._MAX) * w)
            p.drawLine(x, 0, x, h)

        p.end()


class _CalibrationDialog(QDialog):
    """
    Modal dialog: live camera feed -> checkerboard capture -> solve -> show result.
    """

    _frame_ready = pyqtSignal(np.ndarray, int, int)
    _result_ready = pyqtSignal(float)
    _aborted = pyqtSignal()

    def __init__(
        self,
        camera_id: int,
        out_path: str,
        cols: int,
        rows: int,
        square_mm: float,
        min_frames: int,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self._camera_id = camera_id
        self._out_path = out_path
        self._cols = cols
        self._rows = rows
        self._square_mm = square_mm
        self._min_frames = min_frames

        self._stop_event = threading.Event()
        self._capture_thread = None

        self.setWindowTitle("Camera Calibration")
        self.setMinimumSize(800, 620)
        self.setStyleSheet(f"background-color: {BG_DEEP};")

        self._build()
        self._frame_ready.connect(self._on_frame)
        self._result_ready.connect(self._on_result)
        self._aborted.connect(self.close)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Preview area 
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setStyleSheet("background-color: #000000;")
        root.addWidget(self._preview, stretch=1)

        # Result card
        self._result_card = QWidget()
        self._result_card.setStyleSheet(f"background-color: {BG_PANEL};")
        result_layout = QVBoxLayout(self._result_card)
        result_layout.setContentsMargins(30, 20, 30, 20)
        result_layout.setSpacing(10)

        self._result_title = QLabel()
        self._result_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        result_layout.addWidget(self._result_title)

        self._result_bar_holder = QVBoxLayout()
        result_layout.addLayout(self._result_bar_holder)

        self._result_verdict = QLabel()
        self._result_verdict.setStyleSheet("font-size: 12px; font-weight: bold;")
        result_layout.addWidget(self._result_verdict)

        self._result_tip = QLabel()
        self._result_tip.setObjectName("dim")
        self._result_tip.setWordWrap(True)
        self._result_tip.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        result_layout.addWidget(self._result_tip)

        self._result_card.hide()
        root.addWidget(self._result_card)

        # Status bar
        bar = QWidget()
        bar.setStyleSheet(f"background-color: {BG_PANEL}; border-top: 1px solid {BG_BORDER};")
        bar.setFixedHeight(52)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(20, 0, 20, 0)
        bar_layout.setSpacing(20)

        self._progress_lbl = QLabel(f"Captured: 0 / {self._min_frames}")
        self._progress_lbl.setStyleSheet(f"color: {TEXT_MAIN}; font-size: 13px; font-weight: bold;")
        bar_layout.addWidget(self._progress_lbl)

        self._hint_lbl = QLabel("Move the checkerboard to fill the frame from all angles.")
        self._hint_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        bar_layout.addWidget(self._hint_lbl, stretch=1)

        self._action_btn = QPushButton("STOP")
        self._action_btn.setObjectName("danger")
        self._action_btn.setFixedWidth(110)
        self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._action_btn.clicked.connect(self._abort)
        bar_layout.addWidget(self._action_btn)

        root.addWidget(bar)

    def showEvent(self, event):
        super().showEvent(event)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="calibration-capture"
        )
        self._capture_thread.start()

    def closeEvent(self, event):
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        event.accept()

    def _abort(self):
        self._stop_event.set()
        self.close()

    def _capture_loop(self):
        pattern  = (self._cols, self._rows)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        obj_pts = np.zeros((self._cols * self._rows, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:self._cols, 0:self._rows].T.reshape(-1, 2)
        obj_pts *= self._square_mm

        real_world_pts, image_pts = [], []

        cap = cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            self._aborted.emit()
            return

        cooldown, last_capture, n_captured = 3.0, -1.0, 0
        gray = None

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            now  = cv2.getTickCount() / cv2.getTickFrequency()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern, None)

            display = frame.copy()
            if found:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, pattern, corners_sub, found)
                if (now - last_capture) >= cooldown:
                    real_world_pts.append(obj_pts.copy())
                    image_pts.append(corners_sub)
                    last_capture = now
                    n_captured  += 1

            self._frame_ready.emit(
                cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                n_captured,
                self._min_frames,
            )

            if n_captured >= self._min_frames:
                break

        cap.release()

        if n_captured >= self._min_frames and gray is not None and not self._stop_event.is_set():
            rms, cam_mtx, dist_coeffs, _, _ = cv2.calibrateCamera(
                real_world_pts, image_pts, gray.shape[::-1], None, None
            )
            np.savez(
                self._out_path,
                camera_matrix=cam_mtx,
                distortion_coefficients=dist_coeffs,
                rms=np.float32(rms),
            )
            self._result_ready.emit(float(rms))
        else:
            if not self._stop_event.is_set():
                self._aborted.emit()

    def _on_frame(self, frame: np.ndarray, captured: int, total: int):
        self._progress_lbl.setText(f"Captured: {captured} / {total}")

        available = self._preview.size()
        ih, iw = frame.shape[:2]
        aw, ah = available.width(), available.height()
        if aw > 1 and ah > 1:
            scale = min(aw / iw, ah / ih)
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        h, w, ch = frame.shape
        self._preview.setPixmap(
            QPixmap.fromImage(QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888))
        )

    def _on_result(self, rms: float):
        """Called on the GUI thread once calibration is solved and saved."""
        verdict_text, verdict_style = _rms_verdict(rms)
        color_map = {"success": SUCCESS, "warning": WARNING, "danger": DANGER}
        color = color_map[verdict_style]

        self._preview.hide()

        self._result_title.setText(f"Calibration complete  —  RMS error: {rms:.4f} px")
        self._result_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {TEXT_MAIN};")

        bar = _RmsBar(rms)
        bar.setFixedHeight(22)
        self._result_bar_holder.addWidget(bar)

        self._result_verdict.setText(verdict_text)
        self._result_verdict.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {color};")

        if rms >= 1.0:
            self._result_tip.setText(
                "Tips: vary the board angle more, cover corners and edges of the frame,\n"
                "ensure the board is flat and well-lit, and avoid motion blur."
            )
        elif rms < 0.5:
            self._result_tip.setText("The calibration is of high quality. No action needed.")
        else:
            self._result_tip.setText(
                "The calibration is usable. For better accuracy, recalibrate with\n"
                "more varied board poses covering the full frame."
            )

        self._result_card.show()

        self._progress_lbl.setText(f"Captured: {self._min_frames} / {self._min_frames}")
        self._hint_lbl.setText("Calibration saved.")
        self._action_btn.setText("CLOSE")
        self._action_btn.setObjectName("success")
        self._action_btn.style().unpolish(self._action_btn)
        self._action_btn.style().polish(self._action_btn)
        self._action_btn.clicked.disconnect()
        self._action_btn.clicked.connect(self.close)


class _InfoGrid(QWidget):
    """Simple two-column key/value grid."""

    def __init__(self, rows: list, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for label, value in rows:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(20)

            lbl = QLabel(label)
            lbl.setObjectName("dim")
            lbl.setFixedWidth(180)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row_layout.addWidget(lbl)

            val = QLabel(value)
            row_layout.addWidget(val)
            row_layout.addStretch()

            layout.addWidget(row)