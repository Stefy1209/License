import numpy as np
from typing import Optional, Tuple


def depth_to_pointcloud(depth_map: np.ndarray, mtx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project every pixel to 3-D (camera space).

    Returns
    -------
    points : (N, 3) float32  — XYZ
    pixels : (N, 2) int      — (row, col)
    """
    h, w = depth_map.shape
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    cols_g, rows_g = np.meshgrid(np.arange(w), np.arange(h))
    X = (cols_g - cx) * depth_map / fx
    Y = (rows_g - cy) * depth_map / fy

    points = np.stack([X, Y, depth_map], axis=-1).reshape(-1, 3).astype(np.float32)
    pixels = np.stack([rows_g, cols_g], axis=-1).reshape(-1, 2)
    return points, pixels


def ransac_plane(points: np.ndarray, n_iter: int, threshold: float) -> Optional[np.ndarray]:
    """Fit aX+bY+cZ+d=0 to *points* via RANSAC; returns (4,) or None."""
    N = len(points)
    if N < 3:
        return None

    rng = np.random.default_rng(seed=0)
    best_plane, best_count = None, 0

    for _ in range(n_iter):
        p1, p2, p3 = points[rng.choice(N, size=3, replace=False)]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal /= norm_len
        d = -normal.dot(p1)

        count = int((np.abs(points @ normal + d) < threshold).sum())
        if count > best_count:
            best_count = count
            best_plane = np.append(normal, d)

    # Least-squares refinement over inliers
    if best_plane is not None:
        dist = np.abs(points @ best_plane[:3] + best_plane[3])
        inliers = points[dist < threshold]
        if len(inliers) >= 3:
            A = np.hstack([inliers, np.ones((len(inliers), 1), dtype=np.float32)])
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            plane = Vt[-1]
            plane /= np.linalg.norm(plane[:3]) + 1e-9
            best_plane = plane

    return best_plane


def _resolve_threshold(depth_map: np.ndarray, depth_mode: str,
                        threshold_metric: float, threshold_relative: float) -> float:
    """
    Return the RANSAC inlier threshold appropriate for the active depth mode.

    metric   : return threshold_metric unchanged (absolute, in metres).
    relative : return threshold_relative * depth_range, so the tolerance scales with the actual spread of depth values in the frame.
    """
    if depth_mode == "metric":
        return threshold_metric
    # relative mode: scale by the depth range of the current frame
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_range = d_max - d_min
    if depth_range < 1e-6:
        return threshold_relative        # degenerate frame — use raw value
    return threshold_relative * depth_range


def detect_ground_mask(
    depth_map: np.ndarray,
    mtx: np.ndarray,
    seed_region: float,
    ransac_threshold_metric: float,
    ransac_threshold_relative: float,
    ransac_iterations: int,
    plane_smoothing: float,
    normal_threshold: float,
    prev_plane: Optional[np.ndarray],
    depth_mode: str = "metric",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (binary ground mask, updated plane coefficients).

    Parameters
    ----------
    ransac_threshold_metric   : inlier tolerance in metres (metric mode).
    ransac_threshold_relative : inlier tolerance as a fraction of the depth range (relative mode).
    depth_mode                : "metric" | "relative" — selects which threshold to use and how to interpret depth.
    """
    h = depth_map.shape[0]

    threshold = _resolve_threshold(
        depth_map, depth_mode,
        ransac_threshold_metric, ransac_threshold_relative,
    )

    all_points, all_pixels = depth_to_pointcloud(depth_map, mtx)

    # Seed RANSAC from the bottom portion of the image
    seed_row  = int(h * (1.0 - seed_region))
    seed_mask = all_pixels[:, 0] >= seed_row
    seed_pts  = all_points[seed_mask]

    raw_plane = ransac_plane(seed_pts, ransac_iterations, threshold)

    # Reject planes not roughly horizontal (normal ≈ Y axis)
    if raw_plane is not None and abs(raw_plane[1]) < normal_threshold:
        raw_plane = None

    # Temporal smoothing (EMA)
    if raw_plane is not None and prev_plane is not None:
        smoothed_plane = plane_smoothing * prev_plane + (1.0 - plane_smoothing) * raw_plane
        smoothed_plane /= np.linalg.norm(smoothed_plane[:3]) + 1e-9
    else:
        smoothed_plane = raw_plane if raw_plane is not None else prev_plane

    ground_mask = np.zeros(depth_map.shape, dtype=bool)
    if smoothed_plane is not None:
        dist = np.abs(all_points @ smoothed_plane[:3] + smoothed_plane[3])
        inlier_pixels = all_pixels[dist < threshold]
        ground_mask[inlier_pixels[:, 0], inlier_pixels[:, 1]] = True

    return ground_mask, smoothed_plane
