"""Geometry helpers for simplex-based credal set plotting."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _sort_vertices_by_angle(pts: np.ndarray) -> np.ndarray:
    """Sort a set of 3-simplex points by polar angle around their centroid.

    Args:
        pts: Array of shape (n, 3) with points on the simplex.

    Returns:
        The same points reordered by ascending polar angle for polygon winding.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def _compute_convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Compute the ordered convex hull boundary from a set of 3-simplex vertices.

    For fewer than 3 points the hull degenerates, so the points are returned
    sorted by angle instead. For 3 or more points scipy ``ConvexHull`` is used
    on the first two coordinates (the third is redundant on the simplex), with
    a fallback to angle sorting if the hull computation fails.

    Args:
        points: Array of shape (n, 3) with points on the 3-simplex.

    Returns:
        Array of shape (m, 3) with hull vertices in polygon winding order.
    """
    if len(points) < 3:
        return _sort_vertices_by_angle(points)

    pts_2d = points[:, :2]
    try:
        hull = ConvexHull(pts_2d)
        return points[hull.vertices]
    except QhullError:
        return _sort_vertices_by_angle(points)


def _compute_interval_vertices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Compute vertices of the feasible region on the 3-simplex from probability intervals.

    Enumerates all combinations of lower/upper bounds for pairs of dimensions,
    derives the third via the simplex constraint (sum = 1), and keeps only
    vertices that satisfy all bounds. Returns vertices sorted by polar angle
    for correct polygon winding.

    Args:
        lower: Lower probability bounds, shape (3,).
        upper: Upper probability bounds, shape (3,).

    Returns:
        Array of shape (n_vertices, 3) with feasible vertices sorted for polygon rendering.

    Raises:
        ValueError: If no feasible vertices are found.
    """
    vertices: list[list[float]] = []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (0, 2, 1)]:
        for x in [lower[i], upper[i]]:
            for y in [lower[j], upper[j]]:
                z = 1.0 - x - y
                if lower[k] - 1e-9 <= z <= upper[k] + 1e-9:
                    p = [0.0, 0.0, 0.0]
                    p[i] = x
                    p[j] = y
                    p[k] = z
                    vertices.append(p)

    if not vertices:
        msg = "No feasible vertices found. Check that the probability intervals are valid and overlap the simplex."
        raise ValueError(msg)

    pts = np.array(vertices)
    pts = np.unique(np.round(pts, decimals=10), axis=0)  # remove near-duplicate vertices

    return _sort_vertices_by_angle(pts)
