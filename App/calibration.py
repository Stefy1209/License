import sys
import argparse
import numpy as np
import cv2 as cv
from typing import Optional, Tuple

from config_loader import load_config


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (camera_matrix, distortion_coefficients) from an .npz file."""
    with np.load(path) as data:
        mtx  = data.get("mtx")  or data.get("K")  or data.get("camera_matrix")
        dist = data.get("dist") or data.get("D")  or data.get("distortion_coefficients")
    if mtx is None or dist is None:
        raise ValueError("Calibration file missing 'mtx'/'K' or 'dist'/'D' keys.")
    return mtx, dist


def run_calibration(
    camera_id: int,
    out_path: str,
    cols: int,
    rows: int,
    square_mm: float,
    min_frames: int,
) -> None:
    pattern  = (cols, rows)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_pts = np.zeros((cols * rows, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_pts *= square_mm

    real_world_points, image_points = [], []

    camera = cv.VideoCapture(camera_id)
    if not camera.isOpened():
        sys.exit(f"ERROR: Cannot open camera {camera_id}")

    cooldown_sec, last_capture, n_captured = 1.0, -1.0, 0
    gray = None

    while True:
        if n_captured >= min_frames:
            rms, cam_mtx, dist_coeffs, _, _ = cv.calibrateCamera(
                real_world_points, image_points, gray.shape[::-1], None, None
            )
            print(f"RMS: {rms:.4f}" + ("  — consider recalibrating." if rms > 1.0 else ""))
            np.savez(out_path, camera_matrix=cam_mtx,
                     distortion_coefficients=dist_coeffs, rms=np.float32(rms))
            print(f"Saved to '{out_path}'.")
            break

        ok, frame = camera.read()
        if not ok:
            sys.exit("ERROR: Lost camera feed.")

        now  = cv.getTickCount() / cv.getTickFrequency()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, pattern, None)

        display = frame.copy()
        if found:
            corners_sub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(display, pattern, corners_sub, found)
            if (now - last_capture) >= cooldown_sec:
                real_world_points.append(obj_pts.copy())
                image_points.append(corners_sub)
                last_capture = now
                n_captured  += 1

        cv.imshow("Camera Calibration", display)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV checkerboard camera calibration",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--camera", type=int,   default=None)
    parser.add_argument("--out",    type=str,   default=None)
    parser.add_argument("--cols",   type=int,   default=None)
    parser.add_argument("--rows",   type=int,   default=None)
    parser.add_argument("--square", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_calibration(
        camera_id = args.camera if args.camera is not None else cfg["camera"]["id"],
        out_path  = args.out    if args.out    is not None else cfg["calibration"]["file"],
        cols      = args.cols   if args.cols   is not None else cfg["calibration"]["cols"],
        rows      = args.rows   if args.rows   is not None else cfg["calibration"]["rows"],
        square_mm = args.square if args.square is not None else cfg["calibration"]["square_mm"],
        min_frames= cfg["calibration"]["min_frames"],
    )
