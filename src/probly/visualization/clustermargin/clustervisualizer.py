"""Visualizing the uncertainty between two 2D clusters. Derived from margin-based confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from matplotlib.colors import Colormap  # noqa: TC002
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

import probly.visualization.config as cfg

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check of input shape.

    Args:
        input_data: Input data with shape (n_samples, 2).
    """
    msg_type = "Input must be a NumPy Array."
    msg_empty = "Input must not be empty."
    msg_shape = "Input must have shape (n_samples, 2)."
    if not isinstance(input_data, np.ndarray):
        raise TypeError(msg_type)
    if input_data.size == 0:
        raise ValueError(msg_empty)
    if input_data.ndim != 2 or input_data.shape[1] != 2:
        raise ValueError(msg_shape)
    return input_data


def _2_cluster_to_y(cluster1: np.ndarray, cluster2: np.ndarray) -> np.ndarray:
    """Helper method to convert 2 clusters into one array with labels for SVM.

    Args:
        cluster1: 2D NumPy array with shape (n_samples, 2).
        cluster2: 2D NumPy array with shape (n_samples, 2).

    Returns:
        One 1D NumPy array with shape (n_labels, ) only consisting of 0s and 1s.
    """
    input1 = _check_shape(cluster1)
    input2 = _check_shape(cluster2)
    y = np.concatenate(
        (
            np.zeros(len(input1), dtype=int),
            np.ones(len(input2), dtype=int),
        )
    )
    return cast("np.ndarray", y)


def _2_cluster_to_x(cluster1: np.ndarray, cluster2: np.ndarray) -> np.ndarray:
    """Helper method to convert 2 clusters into one cluster with all samples.

    Args:
        cluster1: 2D NumPy array with shape (n_samples, 2).
        cluster2: 2D NumPy array with shape (n_samples, 2).

    Returns:
        One 2D NumPy array with shape (n_samples, 2).
    """
    input1 = _check_shape(cluster1)
    input2 = _check_shape(cluster2)
    stacked_cluster = np.vstack((input1, input2))
    return stacked_cluster


def _plot_svm_beam(ax: Axes, clf: SVC, X: np.ndarray, cmap: Colormap) -> None:  # noqa: N803
    """Helper method with SVM logic to depict margin-based confidence, adjusted to uncertainty between two 2D clusters.

    Args:
        ax: Matplotlib Axes to draw the uncertainty contour on.
        clf: Trained SVC classifier.
        X: 2D array of input samples with shape (n_samples, 2).
        cmap: Matplotlib colormap used to visualize uncertainty.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    margin = np.abs(clf.decision_function(grid)).reshape(xx.shape)

    uncertainty = np.exp(-margin)
    umin = float(uncertainty.min())
    umax = float(uncertainty.max())
    uncertainty = (uncertainty - umin) / (umax - umin + 1e-12)
    contour = ax.contourf(xx, yy, uncertainty, levels=200, cmap=cmap, alpha=0.8)

    cbar = ax.figure.colorbar(contour, ax=ax)
    cfg.style_colorbar(cbar, label="Uncertainty")


def plot_uncertainty(
    input_1: np.ndarray,
    input_2: np.ndarray,
    ax: Axes | None = None,
    title: str = "Uncertainty",
    x_label: str = "Feature 1",
    y_label: str = "Feature 2",
    class_labels: tuple[str, str] | None = None,
    kernel: Literal["linear", "rbf", "sigmoid"] = "rbf",
    C: float = 0.5,  # noqa: N803
    gamma: float | Literal["auto", "scale"] = "scale",
    show: bool = True,
) -> Axes:
    """Method to plot uncertainty between two 2D clusters.

    Args:
        input_1: First 2D NumPy array with shape (n_samples, 2).
        input_2: Second 2D NumPy array with shape (n_samples, 2).
        ax: Matplotlib Axes to draw the plot on. If None, a new figure and axes are created.
        title: Title of plot, defaults to "Uncertainty".
        x_label: Name of x-axis, defaults to "Feature 1".
        y_label: Name of y-axis, defaults to "Feature 2".
        class_labels: Optional names for the two classes. Defaults to ("Class 1", "Class 2").
        kernel: SVM kernel type, one of {"linear", "rbf", "sigmoid"}. Default is "rbf".
        C: SVM regularization parameter. Must be > 0.0. Lower values tolerate more outliers.
        gamma: Kernel coefficient controlling the influence radius of samples.
               Must be >= 0.0, or one of {"auto", "scale"}. Higher values create more local decision boundaries.
        show: Whether to display the plot immediately.

    Returns:
        The Matplotlib Axes containing the uncertainty plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    X = _2_cluster_to_x(input_1, input_2)  # noqa: N806
    y = _2_cluster_to_y(input_1, input_2)

    X = _check_shape(X)  # noqa: N806
    msg_wrong_gamma = "gamma has to be >= 0.0 or one of {'auto', 'scale'}"

    if isinstance(gamma, (int, float)):
        if gamma < 0.0:
            raise ValueError(msg_wrong_gamma)
    elif isinstance(gamma, str):
        if gamma not in {"auto", "scale"}:
            raise ValueError(msg_wrong_gamma)
    else:
        raise TypeError(msg_wrong_gamma)
    msg_wrong_c = "C has to be > 0.0"
    if C < 0.0:
        raise ValueError(msg_wrong_c)
    msg_labels = "Number of labels must match number of samples."
    if len(y) != len(X):
        raise ValueError(msg_labels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(f"{title} (kernel: {kernel}, C: {C})")

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    default_names = tuple(f"Class {i + 1}" for i in range(n_classes))
    legend_names = default_names if class_labels is None else tuple(class_labels)

    cmap = cfg.PROBLY_CMAP

    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=20, color=colors[i], alpha=0.8, zorder=10, label=legend_names[i])

    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X, y)

    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.12), borderaxespad=0.0, frameon=True)
    ax.figure.subplots_adjust(right=0.75, top=0.90)

    _plot_svm_beam(ax, clf, X, cmap)

    if show:
        plt.show()
    return ax
