"""One method for the user to call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


from probly.visualization.dirichlet.plot_dirichlet import DirichletTernaryVisualizer


def create_dirichlet_plot(
    alpha: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    *,
    show: bool = True,
) -> Axes | None:
    """Create a ternary Dirichlet distribution plot.

    Args:
    alpha: Dirichlet concentration parameters.
    labels: List of labels corresponding to the simplex corners.
    title: Custom plot title.
    show: Enables the user to decide whether to show the plot or not.
    """
    alpha = np.asarray(alpha)
    if alpha.shape != (3,):
        msg = "Dirichlet plot requires exactly three alpha values."
        raise ValueError(msg)

    if labels is None:
        labels = ["θ₁", "θ₂", "θ₃"]

    if title is None:
        title = f"Dirichlet Distribution (α = {alpha.tolist()})"  # noqa: RUF001

    visualizer = DirichletTernaryVisualizer()
    ax = visualizer.dirichlet_plot(
        alpha=alpha,
        labels=labels,
        title=title,
    )

    if show:
        plt.show()

    return ax
