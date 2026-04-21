"""
main.py — Entry point for the depth-based ground detection and path planning system.

Hardware profiles
-----------------
Set  [hardware] profile = "nvidia"  or  "rpi"  in config.toml.

  nvidia : NVIDIA GPU host — PyTorch + CUDA inference via DepthAnything3.
  rpi    : Raspberry Pi 5 + AI HAT+ (Hailo-8L NPU) — hailort inference, picamera2 camera capture.

Calibration
-----------
If the calibration file is missing at startup, the user is prompted:
  * Run calibration now   -> runs run_calibration() then continues.
  * Skip (use fallback)   -> continues with an estimated camera matrix.
  * Quit                  -> exits cleanly.
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import cv2

from config_loader  import load_config
from hardware       import get_profile, HardwareProfile
from calibration    import load_calibration, run_calibration
from camera         import open_camera, build_undistort_maps, build_fallback_intrinsics, undistort
from model          import load_model, estimate_depth
from ground         import detect_ground_mask
from path           import find_starting_point, find_ending_point, find_path
from visualization  import (
    visualize_depth, overlay_ground, overlay_path,
    add_status_bar, save_depth_map, save_ground_mask,
)


# ---------------------------------------------------------------------------
# Calibration prompt
# ---------------------------------------------------------------------------

def _prompt_calibration(cal_cfg: dict, cam_cfg: dict) -> None:
    """
    If the calibration file is absent, ask the user what to do.

    Options
    -------
    c — run interactive checkerboard calibration then continue.
    s — skip calibration (main loop will use a fallback intrinsics estimate).
    q — quit the program.
    """
    cal_file = cal_cfg["file"]
    if os.path.exists(cal_file):
        return  # Nothing to do — file is present.

    print(
        f"\nCalibration file '{cal_file}' not found.\n"
        "  [c] Run camera calibration now\n"
        "  [s] Skip (use fallback intrinsics — accuracy may be reduced)\n"
        "  [q] Quit\n"
    )

    while True:
        choice = input("Your choice [c/s/q]: ").strip().lower()
        if choice == "q":
            sys.exit("Exiting.")
        if choice == "s":
            print("Skipping calibration. Continuing with fallback intrinsics.")
            return
        if choice == "c":
            print("Starting calibration…  (press 'q' inside the window to abort)\n")
            run_calibration(
                camera_id  = cam_cfg["id"],
                out_path   = cal_file,
                cols       = cal_cfg["cols"],
                rows       = cal_cfg["rows"],
                square_mm  = cal_cfg["square_mm"],
                min_frames = cal_cfg["min_frames"],
            )
            if os.path.exists(cal_file):
                print("Calibration complete.\n")
            else:
                print("Calibration was not saved (window closed early). Using fallback.\n")
            return
        print("Please enter 'c', 's', or 'q'.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    cam_cfg = cfg["camera"]
    cal_cfg = cfg["calibration"]
    mdl_cfg = cfg["model"]
    dep_cfg = cfg["depth"]
    gnd_cfg = cfg["ground"]
    vis_cfg = cfg["visualization"]

    ground_colour = tuple(vis_cfg["ground_colour_bgr"])

    # ------------------------------------------------------------------ #
    #  Hardware profile                                                    #
    # ------------------------------------------------------------------ #
    hw     = get_profile(cfg)
    device = hw.torch_device()
    print(f"Hardware profile : {hw.profile}  |  device : {device}")

    # ------------------------------------------------------------------ #
    #  Calibration                                                         #
    # ------------------------------------------------------------------ #
    _prompt_calibration(cal_cfg, cam_cfg)

    mtx = dist = undistort_maps = None
    try:
        mtx, dist = load_calibration(cal_cfg["file"])
        print("Calibration loaded.")
    except Exception as exc:
        print(f"No calibration: {exc}. Running with fallback intrinsics.")

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    print(f"Loading depth model ({hw.profile} backend)…")
    model = load_model(mdl_cfg["id"], device, hw, cfg)

    # ------------------------------------------------------------------ #
    #  Camera                                                              #
    # ------------------------------------------------------------------ #
    cap = open_camera(
        cam_cfg["id"],
        hw,
        width  = cam_cfg.get("width",  640),
        height = cam_cfg.get("height", 480),
    )

    print("Press 'q' to quit, 's' to save the current depth map and ground mask.")

    consecutive_failures = 0
    prev_plane           = None
    fallback_mtx         = None
    path                 = np.empty((0, 2), dtype=int)
    start_point          = None
    end_point            = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= cam_cfg["max_read_retries"]:
                    print("Too many camera failures. Exiting.")
                    break
                continue
            consecutive_failures = 0

            h, w = frame.shape[:2]

            # Build helpers once frame size is known
            if mtx is not None and dist is not None and undistort_maps is None:
                undistort_maps = build_undistort_maps(mtx, dist, (w, h))
            if mtx is None and fallback_mtx is None:
                fallback_mtx = build_fallback_intrinsics(w, h)

            if undistort_maps is not None:
                frame = undistort(frame, undistort_maps)

            rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            effective_mtx = mtx if mtx is not None else fallback_mtx

            try:
                # Suppress noisy stdout from model libraries
                sys.stdout = open(os.devnull, "w")
                raw_depth = estimate_depth(rgb_frame, model, mtx, device, hw)
                sys.stdout = sys.__stdout__

                depth_map = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

                if effective_mtx is not None:
                    ground_mask, prev_plane = detect_ground_mask(
                        depth_map, effective_mtx,
                        seed_region       = gnd_cfg["seed_region"],
                        ransac_threshold  = gnd_cfg["ransac_threshold"],
                        ransac_iterations = gnd_cfg["ransac_iterations"],
                        plane_smoothing   = gnd_cfg["plane_smoothing"],
                        normal_threshold  = gnd_cfg["normal_threshold"],
                        prev_plane        = prev_plane,
                    )

                    # Path planning
                    try:
                        start_point = find_starting_point(ground_mask)
                        end_point   = find_ending_point(ground_mask)
                        path        = find_path(ground_mask, start_point, end_point)
                    except ValueError:
                        path        = np.empty((0, 2), dtype=int)
                        start_point = None
                        end_point   = None

                else:
                    ground_mask = np.zeros((h, w), dtype=bool)
                    path        = np.empty((0, 2), dtype=int)
                    start_point = None
                    end_point   = None

                depth_color, _, _ = visualize_depth(depth_map)

                frame_view = overlay_ground(frame,       ground_mask, ground_colour, vis_cfg["ground_overlay_alpha"])
                depth_view = overlay_ground(depth_color, ground_mask, ground_colour, vis_cfg["ground_overlay_alpha"])

                frame_view = overlay_path(frame_view, path, start_point, end_point)
                depth_view = overlay_path(depth_view, path, start_point, end_point)

                add_status_bar(frame_view, prev_plane)

                cv2.imshow(vis_cfg["window_title"], np.hstack((frame_view, depth_view)))

            except RuntimeError as exc:
                sys.stdout = sys.__stdout__
                print(f"Inference error: {exc}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_depth_map(depth_map, dep_cfg["depth_map_save_location"])
                save_ground_mask(ground_mask, gnd_cfg["ground_map_save_location"])

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth-based ground detection and path planning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.toml", help="Path to TOML configuration file.")
    args = parser.parse_args()
    run(load_config(args.config))
