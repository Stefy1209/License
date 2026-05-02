import numpy as np
from typing import Optional, Tuple


#  1. Subsampled RANSAC seed (~1500 points instead of ~38000).
#  2. Vectorized RANSAC: all iterations evaluated in parallel via numpy.
#  3. Stride downsampling of the bottom region (every 4th pixel).
#  4. Mask built directly in 2D image space (no full point-cloud array).


def _resolve_threshold(depth_map: np.ndarray, depth_mode: str,
                       threshold_metric: float, threshold_relative: float) -> float:
    if depth_mode == "metric":
        return threshold_metric
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_range = d_max - d_min
    if depth_range < 1e-6:
        return threshold_relative
    return threshold_relative * depth_range


def _seed_points(
    depth_map: np.ndarray,
    mtx: np.ndarray,
    seed_region: float,
    stride: int,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Back-project a stride-sampled subset of the bottom region to 3D."""
    h, w = depth_map.shape
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    seed_row = int(h * (1.0 - seed_region))

    rows = np.arange(seed_row, h, stride)
    cols = np.arange(0, w, stride)
    cc, rr = np.meshgrid(cols, rows)
    z = depth_map[rr, cc].astype(np.float32)

    valid = (z > 1e-3) & np.isfinite(z)
    cc, rr, z = cc[valid], rr[valid], z[valid]

    if z.size > max_points:
        idx = rng.choice(z.size, size=max_points, replace=False)
        cc, rr, z = cc[idx], rr[idx], z[idx]

    x = (cc - cx) * z / fx
    y = (rr - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _vectorized_ransac(
    pts: np.ndarray,
    n_iter: int,
    threshold: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Vectorized plane RANSAC: sample n_iter triplets in parallel."""
    N = pts.shape[0]
    if N < 3:
        return None

    idx = rng.integers(0, N, size=(n_iter, 3))
    p1 = pts[idx[:, 0]]
    p2 = pts[idx[:, 1]]
    p3 = pts[idx[:, 2]]

    normals = np.cross(p2 - p1, p3 - p1)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return None

    normals = normals[valid] / lengths[valid, None]
    d_vals = -np.einsum("ij,ij->i", normals, p1[valid])

    dists = np.abs(normals @ pts.T + d_vals[:, None])
    counts = (dists < threshold).sum(axis=1)
    best = int(np.argmax(counts))

    plane = np.append(normals[best], d_vals[best]).astype(np.float32)

    inlier_dist = np.abs(pts @ plane[:3] + plane[3])
    inliers = pts[inlier_dist < threshold]
    if inliers.shape[0] >= 3:
        A = np.hstack([inliers, np.ones((inliers.shape[0], 1), dtype=np.float32)])
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        refined = Vt[-1].astype(np.float32)
        n_len = np.linalg.norm(refined[:3]) + 1e-9
        plane = refined / n_len

    return plane


def _build_mask(depth_map: np.ndarray, mtx: np.ndarray, plane: np.ndarray, threshold: float) -> np.ndarray:
    """Build mask in 2D without materializing a full point cloud array."""
    h, w = depth_map.shape
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    a, b, c, d = plane

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    coeff = a * (uu - cx) / fx + b * (vv - cy) / fy + c
    dist = np.abs(coeff * depth_map + d)
    return dist < threshold


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
    rng = np.random.default_rng(seed=0)

    threshold = _resolve_threshold(
        depth_map, depth_mode,
        ransac_threshold_metric, ransac_threshold_relative,
    )

    seed_pts = _seed_points(
        depth_map, mtx, seed_region,
        stride=4,
        max_points=1500,
        rng=rng,
    )

    raw_plane = _vectorized_ransac(seed_pts, ransac_iterations, threshold, rng)

    if raw_plane is not None and abs(raw_plane[1]) < normal_threshold:
        raw_plane = None

    if raw_plane is not None and prev_plane is not None:
        smoothed = plane_smoothing * prev_plane + (1.0 - plane_smoothing) * raw_plane
        smoothed /= np.linalg.norm(smoothed[:3]) + 1e-9
        smoothed = smoothed.astype(np.float32)
    else:
        smoothed = raw_plane if raw_plane is not None else prev_plane

    if smoothed is None:
        return np.zeros(depth_map.shape, dtype=bool), prev_plane

    mask = _build_mask(depth_map, mtx, smoothed, threshold)
    return mask, smoothed