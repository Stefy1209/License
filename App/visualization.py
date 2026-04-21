import numpy as np
import cv2
from typing import Optional, Tuple


def visualize_depth(depth_map: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Colorize a float32 depth map with INFERNO. Returns (image, min_d, max_d)."""
    min_d, max_d = float(depth_map.min()), float(depth_map.max())
    norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), min_d, max_d


def make_colorbar(height: int, width: int, min_depth: float, max_depth: float,
                  depth_mode: str = "metric") -> np.ndarray:
    """Vertical INFERNO colorbar with depth tick labels.

    Labels show metres when depth_mode="metric", or a unitless 0-1 scale
    when depth_mode="relative".
    """
    gradient = np.tile(np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1), (1, width))
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)

    unit = "m" if depth_mode == "metric" else ""
    for i in range(6):
        frac  = i / 5
        y     = max(int(frac * (height - 1)), 10)
        value = max_depth - frac * (max_depth - min_depth)
        label = f"{value:.2f}{unit}"
        cv2.putText(bar, label, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar, label, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return bar


def overlay_ground(
    frame: np.ndarray,
    ground_mask: np.ndarray,
    colour: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Blend a semi-transparent coloured overlay onto ground pixels."""
    overlay = frame.copy()
    overlay[ground_mask] = colour
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    mask_u8 = ground_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, colour, thickness=2)
    return blended


def overlay_path(image: np.ndarray, path: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    """Overlay the A* path, start, and end markers onto an image (in-place copy)."""
    out = image.copy()

    if len(path) >= 2:
        for i in range(len(path) - 1):
            pt1 = (int(path[i,     1]), int(path[i,     0]))  # (col, row) → (x, y)
            pt2 = (int(path[i + 1, 1]), int(path[i + 1, 0]))
            cv2.line(out, pt1, pt2, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Start marker — green circle
    if start is not None:
        cv2.circle(out, (int(start[1]), int(start[0])), radius=6, color=(0, 255, 0),  thickness=-1)
        cv2.circle(out, (int(start[1]), int(start[0])), radius=6, color=(0, 0, 0),    thickness=1)

    # End marker — red circle
    if end is not None:
        cv2.circle(out, (int(end[1]),   int(end[0])),   radius=6, color=(0, 0, 255),  thickness=-1)
        cv2.circle(out, (int(end[1]),   int(end[0])),   radius=6, color=(0, 0, 0),    thickness=1)

    return out


def add_status_bar(image: np.ndarray, plane: Optional[np.ndarray]) -> None:
    """Draw a HUD strip at the top of *image* (in-place)."""
    if plane is None:
        text, colour = "Ground plane: NOT DETECTED", (0, 0, 220)
    else:
        a, b, c, d = plane
        text, colour = f"Ground: [{a:+.2f}, {b:+.2f}, {c:+.2f}, {d:+.2f}]", (0, 220, 80)

    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(image, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def save_depth_map(depth_map: np.ndarray, path: str) -> None:
    try:
        np.save(path, depth_map)
        print(f"Depth map saved to '{path}'.")
    except Exception as e:
        print(f"Could not save depth map: {e}")

def save_ground_mask(ground_mask: np.ndarray, path: str) -> None:
    try:
        np.save(path, ground_mask)
        print(f"Ground mask saved to '{path}'.")
    except Exception as e:
        print(f"Could not save ground mask: {e}")

