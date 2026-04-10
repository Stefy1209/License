import os
import cv2
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from depth_anything_3.api import DepthAnything3

# Help PyTorch find CUDA libraries in Conda environments
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    os.environ["LD_LIBRARY_PATH"] = conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")


@dataclass
class DepthConfig:
    calibration_file: str = "calibration.npz"
    model_id: str = "depth-anything/da3metric-large"
    camera_index: int = 0
    window_title: str = "DA3 Calibrated Live Feed"
    max_read_retries: int = 5
    depth_map_save_location: str = "depth_map.npy"


def load_calibration(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads camera intrinsic matrix and distortion coefficients."""
    with np.load(path) as data:
        mtx = data.get("mtx") or data.get("K") or data.get("camera_matrix")
        dist = data.get("dist") or data.get("D") or data.get("distortion_coefficients")
    if mtx is None or dist is None:
        raise ValueError("Calibration file missing 'mtx'/'K' or 'dist'/'D' keys.")
    return mtx, dist


def build_undistort_maps(
    mtx: np.ndarray, dist: np.ndarray, frame_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute undistortion maps (much faster than per-frame cv2.undistort)."""
    w, h = frame_size
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_16SC2)
    return map1, map2


def load_model(model_id: str, device: str) -> torch.nn.Module:
    """Load and optionally compile the depth model."""
    # Disable JIT fusion to avoid NVRTC failures on some CUDA setups
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()

    # torch.compile gives a significant speed-up on PyTorch >= 2.0
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    return model


@torch.inference_mode()
def estimate_depth(
    rgb_frame: np.ndarray,
    model: torch.nn.Module,
    intrinsics: Optional[np.ndarray],
    device: str,
) -> np.ndarray:
    """Run DA3 inference; returns a float32 depth map at model resolution."""
    autocast_ctx = torch.amp.autocast("cuda") if device == "cuda" else torch.no_grad()
    with autocast_ctx:
        prediction = model.inference(
            image=[rgb_frame],
            intrinsics=[intrinsics] if intrinsics is not None else None,
        )
    return prediction.depth[0]  # numpy float32, model resolution


def make_colorbar(height: int, width: int, min_depth: float, max_depth: float) -> np.ndarray:
    """Create a vertical INFERNO colorbar with depth labels."""
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    gradient = np.tile(gradient, (1, width))
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    num_ticks = 6

    for i in range(num_ticks):
        frac = i / (num_ticks - 1)
        depth_val = max_depth - frac * (max_depth - min_depth)  # top = far
        y = max(int(frac * (height - 1)), 10)
        label = f"{depth_val:.2f}m"
        # Black outline for readability, white text on top
        cv2.putText(bar, label, (4, y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar, label, (4, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return bar


def visualize_depth(depth_map: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Convert a float32 depth map to a colorized image; returns (image, min_depth, max_depth)."""
    min_depth, max_depth = float(depth_map.min()), float(depth_map.max())
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO), min_depth, max_depth

def save_depth_map(depth_map: np.ndarray, save_location: str):
    try:
        print(f'Trying to save the depth map in location: {save_location}')
        np.save(save_location, depth_map)
        print('Depth map saves successfully!')
    except RuntimeError as e:
        print(f"Inference error: {e}")


def run(cfg: DepthConfig) -> None:
    # --- Calibration ---
    mtx, dist = None, None
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
    try:
        mtx, dist = load_calibration(cfg.calibration_file)
        print("Calibration data loaded successfully.")
    except Exception as e:
        print(f"Calibration failed: {e}. Running uncalibrated.")

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Depth Anything V3 on {device}...")
    model = load_model(cfg.model_id, device)

    # --- Camera ---
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("Starting feed. Press 'q' to exit.")
    consecutive_failures = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= cfg.max_read_retries:
                    print("Camera read failed too many times. Exiting.")
                    break
                continue
            consecutive_failures = 0

            orig_h, orig_w = frame.shape[:2]

            # Build undistort maps on first valid frame (size now known)
            if mtx is not None and dist is not None and undistort_maps is None:
                undistort_maps = build_undistort_maps(mtx, dist, (orig_w, orig_h))

            # Undistort using precomputed maps
            if undistort_maps is not None:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)

            # BGR → RGB once per frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                raw_depth = estimate_depth(rgb_frame, model, mtx, device)

                # Resize depth map back to camera resolution
                depth_map = cv2.resize(raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                depth_color, min_d, max_d = visualize_depth(depth_map)
                colorbar = make_colorbar(orig_h, width=70, min_depth=min_d, max_depth=max_d)
                combined_view = np.hstack((frame, depth_color, colorbar))
                cv2.imshow(cfg.window_title, combined_view)

            except RuntimeError as e:
                print(f"Inference error: {e}")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) & 0xFF == ord("s"):
                save_depth_map(depth_map=depth_map, save_location=cfg.depth_map_save_location)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(DepthConfig())