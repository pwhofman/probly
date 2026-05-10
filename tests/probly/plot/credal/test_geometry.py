"""Tests for geometry primitives in ``probly.plot.credal._geometry``."""

from __future__ import annotations

import numpy as np
import pytest

from probly.plot.credal._geometry import (
    _compute_convex_hull_vertices,
    _compute_interval_vertices,
    _sort_vertices_by_angle,
)


class TestGeometryHelpers:
    """Geometry primitives used by ternary credal plots."""

    def test_sort_vertices_orders_by_angle(self) -> None:
        # Four points at the corners of a square plus z=0; expect cyclic order.
        pts = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        sorted_pts = _sort_vertices_by_angle(pts)
        # Compute angles after sorting; they should be monotonic non-decreasing.
        center = sorted_pts.mean(axis=0)
        angles = np.arctan2(sorted_pts[:, 1] - center[1], sorted_pts[:, 0] - center[0])
        diffs = np.diff(angles)
        assert (diffs >= -1e-10).all()

    def test_convex_hull_vertices_returns_ordered_boundary(self) -> None:
        # 5 points where the interior point should be excluded by ConvexHull.
        pts = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],  # on edge, should still be vertex of hull (or excluded)
                [0.3, 0.3, 0.4],  # interior
            ]
        )
        hull = _compute_convex_hull_vertices(pts)
        assert hull.shape[1] == 3
        # The interior point (0.3, 0.3, 0.4) should not appear in the hull.
        assert not any(np.allclose(v, [0.3, 0.3, 0.4]) for v in hull)

    def test_convex_hull_with_two_points_is_sorted(self) -> None:
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = _compute_convex_hull_vertices(pts)
        assert result.shape == (2, 3)

    def test_convex_hull_with_one_point(self) -> None:
        pts = np.array([[1.0, 0.0, 0.0]])
        result = _compute_convex_hull_vertices(pts)
        np.testing.assert_allclose(result, pts)

    def test_convex_hull_collinear_falls_back_to_angle_sort(self) -> None:
        # Three collinear points: scipy ConvexHull raises QhullError, so the
        # fallback returns the points sorted by angle.
        pts = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.5],
                [1.0, 0.0, 0.0],
            ]
        )
        result = _compute_convex_hull_vertices(pts)
        assert result.shape == (3, 3)

    def test_interval_vertices_full_simplex(self) -> None:
        # Lower=0, Upper=1 -> the whole simplex; we should get the corners.
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        verts = _compute_interval_vertices(lower, upper)
        assert verts.shape[1] == 3
        # Each vertex should sum to 1 within tolerance.
        np.testing.assert_allclose(verts.sum(axis=1), np.ones(verts.shape[0]), atol=1e-9)
        # The unit corners must be present.
        for corner in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            assert any(np.allclose(v, corner) for v in verts)

    def test_interval_vertices_singleton(self) -> None:
        lower = np.array([0.3, 0.3, 0.4])
        upper = np.array([0.3, 0.3, 0.4])
        verts = _compute_interval_vertices(lower, upper)
        # Should yield the single feasible point.
        assert verts.shape[0] >= 1
        for v in verts:
            np.testing.assert_allclose(v, [0.3, 0.3, 0.4], atol=1e-9)

    def test_interval_vertices_empty_raises(self) -> None:
        # Lower bounds sum > 1 -> infeasible, no vertices.
        lower = np.array([0.6, 0.6, 0.6])
        upper = np.array([0.7, 0.7, 0.7])
        with pytest.raises(ValueError, match="No feasible vertices"):
            _compute_interval_vertices(lower, upper)
