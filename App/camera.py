from __future__ import annotations

import sys
import numpy as np
import cv2
from typing import Tuple, Union

from hardware import HardwareProfile
from config import AppConfig


Camera = Union[cv2.VideoCapture, "Picamera2Wrapper"]


class Picamera2Wrapper:
    """Wraps picamera2.Picamera2 to expose the same read() / release() interface as cv2.VideoCapture."""

    def __init__(self, camera_id: int, width: int, height: int) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "picamera2 not found. On the Raspberry Pi run:\n"
                "    sudo apt install -y python3-picamera2\n"
                f"Original error: {exc}"
            ) from exc

        self._cam = Picamera2(camera_id)
        config    = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        self._w, self._h = width, height
        print(f"  [RPi] picamera2 opened (camera {camera_id}, {width}x{height}).")

    def isOpened(self) -> bool:
        return True

    def read(self) -> Tuple[bool, np.ndarray]:
        try:
            rgb = self._cam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return True, bgr
        except Exception as exc:
            print(f"  [RPi] picamera2 capture failed: {exc}")
            return False, None

    def release(self) -> None:
        self._cam.stop()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        return 0.0


class CameraFactory:
    """Creates the appropriate camera backend based on the hardware profile."""

    @staticmethod
    def open(cfg: AppConfig, hw: HardwareProfile) -> Camera:
        camera_id = cfg.camera.id
        width     = cfg.camera.width
        height    = cfg.camera.height

        if hw.is_rpi:
            try:
                return Picamera2Wrapper(camera_id, width, height)
            except ImportError:
                print("  [RPi] picamera2 not available — falling back to OpenCV VideoCapture.")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            sys.exit(f"ERROR: Cannot open camera {camera_id}")
        if hw.is_rpi:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap


def open_camera(camera_id: int, hw: HardwareProfile, width: int = 640, height: int = 480) -> Camera:
    """Open the camera most appropriate for the active hardware profile."""
    if hw.is_rpi:
        try:
            return Picamera2Wrapper(camera_id, width, height)
        except ImportError:
            print("  [RPi] picamera2 not available — falling back to OpenCV VideoCapture.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera {camera_id}")
    if hw.is_rpi:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def build_undistort_maps(
    mtx: np.ndarray, dist: np.ndarray, frame_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute remap arrays (faster than per-frame cv2.undistort)."""
    w, h = frame_size
    map1, map2 = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def build_fallback_intrinsics(w: int, h: int) -> np.ndarray:
    """Estimate a camera matrix assuming ~70 deg horizontal FOV."""
    f = w / (2.0 * np.tan(np.radians(35)))
    return np.array(
        [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64
    )


def undistort(frame: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
