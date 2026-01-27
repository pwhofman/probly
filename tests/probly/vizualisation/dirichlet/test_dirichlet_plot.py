from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.dirichlet.plot_dirichlet import DirichletTernaryVisualizer


def test_triangle_corners_returns_correct_shape_and_values() -> None:
    """triangle_corners returns the three fixed simplex corners."""
    viz = DirichletTernaryVisualizer()
    corners = viz.triangle_corners()

    assert corners.shape == (3, 2)
    assert np.allclose(corners[0], [0.0, 0.0])
    assert np.allclose(corners[1], [1.0, 0.0])
    assert np.allclose(corners[2], [0.5, np.sqrt(3) / 2])


def test_xy_to_barycentric_returns_valid_coordinates() -> None:
    """xy_to_barycentric returns valid barycentric coordinates."""
    viz = DirichletTernaryVisualizer()

    point = np.array([0.4, 0.2])
    bary = viz.xy_to_barycentric(point)

    assert bary.shape == (3,)
    assert np.all(bary > 0.0)
    assert np.all(bary < 1.0)
    assert np.isclose(bary.sum(), 1.0)


def test_dirichlet_pdf_is_positive() -> None:
    """Dirichlet pdf returns a positive value for valid input."""
    alpha = np.array([2.0, 3.0, 4.0])
    dist = DirichletTernaryVisualizer.Dirichlet(alpha)

    x = np.array([0.2, 0.3, 0.5])
    value = dist.pdf(x)

    assert value > 0.0


def test_label_corners_and_vertices_adds_texts() -> None:
    """label_corners_and_vertices adds text labels to the axes."""
    viz = DirichletTernaryVisualizer()
    fig, ax = plt.subplots()

    viz.label_corners_and_vertices(ax, ["A", "B", "C"])

    assert len(ax.texts) > 0
    plt.close(fig)


def test_dirichlet_plot_returns_axes_and_sets_title() -> None:
    """dirichlet_plot returns Axes and sets the given title."""
    viz = DirichletTernaryVisualizer()
    alpha = np.array([2.0, 2.0, 2.0])

    ax = viz.dirichlet_plot(alpha=alpha, labels=["A", "B", "C"], title="Test Dirichlet Plot")

    assert isinstance(ax, Axes)
    assert ax.get_title() == "Test Dirichlet Plot"
    assert len(ax.collections) > 0
    plt.close(ax.figure)
