import numpy as np
from functools import lru_cache
from typing import Optional, Tuple


#  1. Subsampled RANSAC seed (~1500 points instead of ~38000).
#  2. Vectorized RANSAC: all iterations evaluated in parallel via numpy.
#  3. Stride downsampling of the bottom region (every 4th pixel).
#  4. Mask built directly in 2D image space (no full point-cloud array).
#  5. Pixel-coordinate grids depend only on resolution + intrinsics (fixed
#     for a camera session), so they are precomputed once and cached across
#     frames instead of being rebuilt on every call.


def _resolve_threshold(depth_map: np.ndarray, depth_mode: str,
                       threshold_metric: float, threshold_relative: float) -> float:
    if depth_mode == "metric":
        return threshold_metric
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    depth_range = d_max - d_min
    if depth_range < 1e-6:
        return threshold_relative
    return threshold_relative * depth_range


@lru_cache(maxsize=8)
def _seed_grid(
    h: int, w: int,
    fx: float, fy: float, cx: float, cy: float,
    seed_region: float, stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the seed-region pixel indices and normalized camera rays.

    These depend only on the frame resolution, intrinsics and sampling
    parameters — all fixed for a camera session — so the result is cached
    and reused across frames (one cache entry per unique geometry).
    """
    seed_row = int(h * (1.0 - seed_region))
    rows = np.arange(seed_row, h, stride)
    cols = np.arange(0, w, stride)
    cc, rr = np.meshgrid(cols, rows)
    rr = rr.ravel()
    cc = cc.ravel()
    norm_x = (cc - cx) / fx
    norm_y = (rr - cy) / fy
    return rr, cc, norm_x, norm_y


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
    rr, cc, norm_x, norm_y = _seed_grid(
        h, w,
        float(mtx[0, 0]), float(mtx[1, 1]),
        float(mtx[0, 2]), float(mtx[1, 2]),
        seed_region, stride,
    )

    z = depth_map[rr, cc].astype(np.float32)
    valid = (z > 1e-3) & np.isfinite(z)
    z, nx, ny = z[valid], norm_x[valid], norm_y[valid]

    if z.size > max_points:
        idx = rng.choice(z.size, size=max_points, replace=False)
        z, nx, ny = z[idx], nx[idx], ny[idx]

    x = nx * z
    y = ny * z
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


@lru_cache(maxsize=8)
def _mask_grid(
    h: int, w: int,
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute the per-pixel normalized camera rays for the full frame.

    Constant for a given resolution + intrinsics, so it is built once and
    reused across frames instead of rebuilding the meshgrid on every call.
    """
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy
    return x_norm, y_norm


def _build_mask(depth_map: np.ndarray, mtx: np.ndarray, plane: np.ndarray, threshold: float) -> np.ndarray:
    """Build mask in 2D without materializing a full point cloud array."""
    h, w = depth_map.shape
    a, b, c, d = plane

    x_norm, y_norm = _mask_grid(
        h, w,
        float(mtx[0, 0]), float(mtx[1, 1]),
        float(mtx[0, 2]), float(mtx[1, 2]),
    )

    coeff = a * x_norm + b * y_norm + c
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