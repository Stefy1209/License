import numpy as np
import cv2
from typing import Optional, Tuple


def visualize_depth(depth_map: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Colorize a float32 depth map with INFERNO. Returns (image, min_d, max_d)."""
    min_d, max_d = float(depth_map.min()), float(depth_map.max())
    norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), min_d, max_d


def make_colorbar(height: int, width: int, min_depth: float, max_depth: float) -> np.ndarray:
    """Vertical INFERNO colorbar with depth tick labels."""
    gradient = np.tile(np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1), (1, width))
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)

    for i in range(6):
        frac  = i / 5
        y     = max(int(frac * (height - 1)), 10)
        label = f"{max_depth - frac * (max_depth - min_depth):.2f}m"
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
