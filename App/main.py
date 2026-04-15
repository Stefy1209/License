import os
import sys
import argparse
import numpy as np
import cv2
import torch

from config_loader   import load_config
from calibration     import load_calibration
from camera          import open_camera, build_undistort_maps, build_fallback_intrinsics, undistort
from model           import load_model, estimate_depth
from ground          import detect_ground_mask
from visualization   import visualize_depth, make_colorbar, overlay_ground, add_status_bar, save_depth_map, save_ground_mask


def run(cfg: dict) -> None:
    cam_cfg   = cfg["camera"]
    cal_cfg   = cfg["calibration"]
    mdl_cfg   = cfg["model"]
    dep_cfg   = cfg["depth"]
    gnd_cfg   = cfg["ground"]
    vis_cfg   = cfg["visualization"]

    ground_colour = tuple(vis_cfg["ground_colour_bgr"])

    # --- Calibration (optional) ---
    mtx = dist = undistort_maps = None
    try:
        mtx, dist = load_calibration(cal_cfg["file"])
        print("Calibration loaded.")
    except Exception as e:
        print(f"No calibration: {e}. Running uncalibrated.")

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}…")
    model = load_model(mdl_cfg["id"], device)

    # --- Camera ---
    cap = open_camera(cam_cfg["id"])

    print("Press 'q' to quit, 's' to save the depth map.")
    consecutive_failures = 0
    prev_plane   = None
    fallback_mtx = None

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

            # Build helpers once the frame size is known
            if mtx is not None and dist is not None and undistort_maps is None:
                undistort_maps = build_undistort_maps(mtx, dist, (w, h))
            if mtx is None and fallback_mtx is None:
                fallback_mtx = build_fallback_intrinsics(w, h)

            if undistort_maps is not None:
                frame = undistort(frame, undistort_maps)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            effective_mtx = mtx if mtx is not None else fallback_mtx

            try:
                sys.stdout = open(os.devnull, "w")
                raw_depth = estimate_depth(rgb_frame, model, mtx, device)
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
                else:
                    ground_mask = np.zeros((h, w), dtype=bool)

                depth_color, min_d, max_d = visualize_depth(depth_map)
                # colorbar = make_colorbar(h, vis_cfg["colorbar_width"], min_d, max_d)

                frame_view = overlay_ground(frame, ground_mask, ground_colour, vis_cfg["ground_overlay_alpha"])
                add_status_bar(frame_view, prev_plane)

                depth_view = overlay_ground(depth_color, ground_mask, ground_colour, vis_cfg["ground_overlay_alpha"])

                cv2.imshow(vis_cfg["window_title"], np.hstack((frame_view, depth_view))) # colorbar can be put as parameter

            except RuntimeError as e:
                sys.stdout = sys.__stdout__
                print(f"Inference error: {e}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    run(load_config(args.config))
