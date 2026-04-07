import numpy as np
import cv2 as cv
import sys
import argparse

from config_loader import load_config

def run_calibration(camera_id: int, out_path: str, cols: int, rows: int, square_mm: float, min_frames: int) -> None:
    pattern = (cols, rows)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = np.zeros((cols*rows,3), np.float32)
    object_points[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= square_mm

    # Arrays to store object points and image points from all the images.
    real_world_points = [] # 3d point in real world space
    image_points = [] # 2d points in image plane.

    camera = cv.VideoCapture(camera_id)
    if not camera.isOpened():
        sys.exit(f"ERROR: Cannot open camera {camera_id}")

    cooldown_sec = 1.0          
    last_capture = -cooldown_sec
    number_of_captured_frames = 0

    while True:
        if number_of_captured_frames >= min_frames:
            rms, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.calibrateCamera(real_world_points, image_points, gray.shape[::-1], None, None) 

            print(f"RMS Values: {rms}.")

            if rms > 1.0:
                print("Consider recalibrating.")

            np.savez(out_path, camera_matrix = camera_matrix, distortion_coefficients = distortion_coefficients, rms = np.float32(rms))
            break

        camera_found, frame = camera.read()
        if not camera_found:
            sys.exit("ERROR: Lost camera feed.")

        now = cv.getTickCount() / cv.getTickFrequency()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners_found, corners = cv.findChessboardCorners(gray, pattern, None)

        display = frame.copy()
        if corners_found == True:
            corners_sub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(display, pattern, corners_sub, corners_found)
            
            cooldown_left = cooldown_sec - (now - last_capture)
            if cooldown_left <= 0:
                real_world_points.append(object_points.copy())
                image_points.append(corners_sub)
                last_capture = now
                number_of_captured_frames += 1

        cv.imshow("Camera Calibration", display)

        key = cv.waitKey(1) & 0xFF
 
        if key == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV checkerboard camera calibration", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--config", type=str,   default="config.toml", help="Path to TOML configuration file")
    parser.add_argument("--camera", type=int,   default=None, help="Camera index (overrides config)")
    parser.add_argument("--out",    type=str,   default=None, help="Output .npz file path (overrides config)")
    parser.add_argument("--cols",   type=int,   default=None, help="Inner corners — columns (overrides config)")
    parser.add_argument("--rows",   type=int,   default=None, help="Inner corners — rows (overrides config)")
    parser.add_argument("--square", type=float, default=None, help="Square size in mm (overrides config)")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
 
    run_calibration(
        camera_id  = args.camera    if args.camera is not None else cfg["camera"]["id"],
        out_path   = args.out       if args.out    is not None else cfg["calibration"]["file"],
        cols       = args.cols      if args.cols   is not None else cfg["calibration"]["cols"],
        rows       = args.rows      if args.rows   is not None else cfg["calibration"]["rows"],
        square_mm  = args.square    if args.square is not None else cfg["calibration"]["square_mm"],
        min_frames = cfg["calibration"]["min_frames"],
    )