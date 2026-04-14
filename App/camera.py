import sys
import numpy as np
import cv2
from typing import Optional, Tuple


def open_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera {camera_id}")
    return cap


def build_undistort_maps(
    mtx: np.ndarray, dist: np.ndarray, frame_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute remap arrays (faster than per-frame cv2.undistort)."""
    w, h = frame_size
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_16SC2)
    return map1, map2


def build_fallback_intrinsics(w: int, h: int) -> np.ndarray:
    """Estimate a camera matrix assuming ~70° horizontal FOV."""
    f = w / (2.0 * np.tan(np.radians(35)))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def undistort(frame: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
