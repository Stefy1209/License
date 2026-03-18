import numpy as np
import cv2 as cv
import sys
import argparse

from config_loader import load_config

def run_calibration(camera_id: int, out_path: str, cols: int, rows: int, square_mm: float, min_frames: int) -> None:
    # pattern
    pattern = (cols, rows)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_points = np.zeros((cols*rows,3), np.float32)
    object_points[:, :2] = np.array(
        [(c * square_mm, r * square_mm) for r in range(rows) for c in range(cols)],
        dtype=np.float32,
    )

    # Arrays to store object points and image points from all the images.
    real_world_points = [] # 3d point in real world space
    image_points = [] # 2d points in image plane.

    camera = cv.VideoCapture(camera_id)
    if not camera.isOpened():
        sys.exit(f"ERROR: Cannot open camera {camera_id}")

    print(f"\n=== CAMERA CALIBRATION ===")
    print(f"Pattern      : {cols}x{rows} inner corners")
    print(f"Output       : {out_path}")
    print(f"Frames needed: at least {min_frames}\n")
    print("Controls:")
    print("  c      — compute & save calibration")
    print("  q      — quit without saving\n")

    COOLDOWN_SEC = 2.0          
    last_capture = -COOLDOWN_SEC

    while True:
        camera_found, frame = camera.read()
        if not camera_found:
            print("ERROR: Lost camera feed.")
            break

        now = cv.getTickCount() / cv.getTickFrequency()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners_found, corners = cv.findChessboardCorners(gray, pattern, None)

        display = frame.copy()
        if corners_found == True:
            corners_sub = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(display, pattern, corners_sub, corners_found)
            
            cooldown_left = COOLDOWN_SEC - (now - last_capture)
            if cooldown_left <= 0:
                real_world_points.append(object_points.copy())
                image_points.append(corners_sub)
                last_capture = now
                print(f"  Captured frame {len(real_world_points)}")
                status = f"CAPTURED  |  Total: {len(real_world_points)}"
                color  = (0, 255, 120)
            else:
                status = f"FOUND  |  Captured: {len(real_world_points)}  |  Next in {cooldown_left:.1f}s"
                color  = (0, 220, 0)
        else:
            corners_sub = None
            status = f"Searching...  |  Captured: {len(real_world_points)}"
            color  = (0, 100, 255)

        cv.putText(display, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv.putText(display, "SPACE=capture   c=calibrate   q=quit", (10, display.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv.imshow("Camera Calibration", display)

        key = cv.waitKey(1) & 0xFF
 
        if key == ord('q'):
            print("Quit — calibration not saved.")
            break
 
        elif key == ord('c'):
            n = len(real_world_points)
            if n < min_frames:
                print(f"  Need at least {min_frames} frames, have {n}. Keep capturing.")
                continue
 
            print(f"\nComputing calibration from {n} frames ...")
            rms, K, dist, _, _ = cv.calibrateCamera(real_world_points, image_points, gray.shape[::-1], None, None)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
 
            print(f"  RMS reprojection error : {rms:.4f} px")
            print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
            if rms > 1.0:
                print("  WARNING: RMS > 1.0 px — consider re-calibrating.")
                print("           Tips: more frames, vary angles, fill the frame.")
 
            np.savez(out_path, K=K, dist=dist, rms=np.float32(rms))
            print(f"  Saved to '{out_path}'")
            break

    camera.release()
    cv.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
 
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
 
    # CLI wins over config file when explicitly provided
    run_calibration(
        camera_id  = args.camera    if args.camera is not None else cfg["camera"]["id"],
        out_path   = args.out       if args.out    is not None else cfg["calibration"]["file"],
        cols       = args.cols      if args.cols   is not None else cfg["calibration"]["cols"],
        rows       = args.rows      if args.rows   is not None else cfg["calibration"]["rows"],
        square_mm  = args.square    if args.square is not None else cfg["calibration"]["square_mm"],
        min_frames = cfg["calibration"]["min_frames"],
    )