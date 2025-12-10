"""Test for visualization for three classes."""

from __future__ import annotations

import numpy as np
import pytest

from probly.visualization.plot_3d import TernaryVisualizer


def test_ternary_plot_uses_custom_labels_for_vertices() -> None:
    """Test if custom labels are used."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])

    labels = ["A", "B", "C"]
    ax = viz.ternary_plot(probs, labels=labels)

    texts = [t.get_text() for t in ax.texts]

    assert "A" in texts  # noqa: S101
    assert "B" in texts  # noqa: S101
    assert "C" in texts  # noqa: S101


def test_ternary_plot_raises_if_label_count_mismatch() -> None:
    """Testing if lable mismatch is throwing error."""
    viz = TernaryVisualizer()
    probs = np.array([[0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match=r"Lables don't match."):
        viz.ternary_plot(probs, labels=["C1", "C2"])


def test_probs_to_coords_3d_maps_vertices_to_triangle_corners() -> None:
    """Test if edge cases work."""
    viz = TernaryVisualizer()

    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])

    x1, y1 = viz.probs_to_coords_3d(p1)
    x2, y2 = viz.probs_to_coords_3d(p2)
    x3, y3 = viz.probs_to_coords_3d(p3)

    assert x1 == pytest.approx(0.0)  # noqa: S101
    assert y1 == pytest.approx(0.0)  # noqa: S101

    assert x2 == pytest.approx(1.0)  # noqa: S101
    assert y2 == pytest.approx(0.0)  # noqa: S101

    assert x3 == pytest.approx(0.5)  # noqa: S101
    assert y3 == pytest.approx(np.sqrt(3) / 2)  # noqa: S101
