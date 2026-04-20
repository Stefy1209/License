import numpy as np
import heapq
from typing import Tuple


def find_starting_point(ground_mask: np.ndarray) -> Tuple[int, int]:
    """
    Find the starting point at the bottom-center of the ground mask.
    Scans upward from the bottom row to find the first valid ground pixel
    near the horizontal center.

    Returns
    -------
    (row, col) tuple
    """
    h, w = ground_mask.shape
    center_col = w // 2

    # Search upward from the bottom, looking for ground near center
    for row in range(h - 1, -1, -1):
        # Scan outward from center column to find nearest ground pixel
        for offset in range(w // 2 + 1):
            for col in [center_col - offset, center_col + offset]:
                if 0 <= col < w and ground_mask[row, col]:
                    return (row, col)

    # Fallback: return any ground pixel in bottom half
    bottom_half = ground_mask[h // 2 :, :]
    ys, xs = np.where(bottom_half)
    if len(ys) > 0:
        idx = np.argmin(np.abs(xs - center_col))
        return (ys[idx] + h // 2, xs[idx])

    raise ValueError("No ground pixels found in the mask.")


def find_ending_point(ground_mask: np.ndarray) -> Tuple[int, int]:
    """
    Find the ending point at the top-center of the ground mask.
    Scans downward from the top row to find the first valid ground pixel
    near the horizontal center.

    Returns
    -------
    (row, col) tuple
    """
    h, w = ground_mask.shape
    center_col = w // 2

    # Search downward from the top, looking for ground near center
    for row in range(h):
        for offset in range(w // 2 + 1):
            for col in [center_col - offset, center_col + offset]:
                if 0 <= col < w and ground_mask[row, col]:
                    return (row, col)

    # Fallback: return any ground pixel in top half
    top_half = ground_mask[: h // 2, :]
    ys, xs = np.where(top_half)
    if len(ys) > 0:
        idx = np.argmin(np.abs(xs - center_col))
        return (ys[idx], xs[idx])

    raise ValueError("No ground pixels found in the mask.")


def find_path(ground_mask: np.ndarray, starting_point: Tuple[int, int], ending_point: Tuple[int, int]) -> np.ndarray:
    """
    Find the shortest path between starting_point and ending_point on the
    ground mask using the A* algorithm with 8-connectivity.

    Parameters
    ----------
    ground_mask    : (H, W) bool array — True where ground is traversable
    starting_point : (row, col) start position (must be on ground)
    ending_point   : (row, col) goal position (must be on ground)

    Returns
    -------
    path : (N, 2) int array of (row, col) waypoints from start to end,
           or empty array if no path exists.
    """
    h, w = ground_mask.shape
    start = tuple(starting_point)
    goal = tuple(ending_point)

    if start == goal:
        return np.array([start], dtype=int)

    # 8-directional neighbours: (drow, dcol, cost)
    NEIGHBORS = [
        (-1,  0, 1.0),       # up
        ( 1,  0, 1.0),       # down                                
        ( 0, -1, 1.0),       # left             
        ( 0,  1, 1.0),       # right             
        (-1, -1, 1.4142),    # down-left
        (-1,  1, 1.4142),    # down-right
        ( 1, -1, 1.4142),    # up-left
        ( 1,  1, 1.4142),    # up-right
    ]

    def heuristic(a, b):
        # Octile distance — admissible for 8-connectivity
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)

    # g_score[row, col] = best known cost from start
    g_score = np.full((h, w), np.inf, dtype=np.float32)
    g_score[start] = 0.0

    came_from = {}  # (row, col) -> (row, col)

    # Min-heap: (f_score, row, col)
    open_heap = [(heuristic(start, goal), start[0], start[1])]

    while open_heap:
        f, row, col = heapq.heappop(open_heap)
        current = (row, col)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return np.array(path, dtype=int)

        # Skip if we already found a better route to this node
        if f > g_score[current] + heuristic(current, goal) + 1e-6:
            continue

        for dr, dc, move_cost in NEIGHBORS:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not ground_mask[nr, nc]:
                continue

            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_new, nr, nc))

    # No path found
    return np.empty((0, 2), dtype=int)