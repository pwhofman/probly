import matplotlib.pyplot as plt
import numpy as np


def simplex_plot(probs: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Plot probability distributions on the simplex.

    Args:
        probs: numpy.ndarray of shape (n_instances, n_classes)

    Returns:
        fig: matplotlib figure
        ax: matplotlib axes

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="ternary")
    ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2])
    return fig, ax
