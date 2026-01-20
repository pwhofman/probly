"""Visualizing the uncertainty between two 2D clusters."""

from __future__ import annotations

from typing import Literal

from matplotlib.colors import Colormap  # noqa: TC002
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.svm import SVC


def _check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check of input shape.

    Args:
        input_data: input data with shape (n_samples, 2).
    """
    msg_type = "Input must be a numpy array."
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
        cluster1: 2D numpy array with shape (n_samples, 2).
        cluster2: 2D numpy array with shape (n_samples, 2).

    Returns:
        One 1D numpy array with shape (n_samples, ) only consisting of 0s and 1s.
    """
    input1 = _check_shape(cluster1)
    input2 = _check_shape(cluster2)
    y = np.concatenate(
        (
            np.zeros(len(input1), dtype=int),
            np.ones(len(input2), dtype=int),
        )
    )
    return y


def _2_cluster_to_x(cluster1: np.ndarray, cluster2: np.ndarray) -> np.ndarray:
    """Helper method to convert 2 clusters to one cluster with all samples.

    Args:
        cluster1: 2D numpy array with shape (n_samples, 2).
        cluster2: 2D numpy array with shape (n_samples, 2).

    Returns:
        One 2D numpy array with shape (n_samples, 2).
    """
    input1 = _check_shape(cluster1)
    input2 = _check_shape(cluster2)
    stacked_cluster = np.vstack((input1, input2))
    return stacked_cluster


def _plot_svm_beam(ax: plt.Axes, clf: SVC, X: np.ndarray, cmap: Colormap) -> None:  # noqa: N803
    """Helper method that contains SVM logic to depict uncertainty between two 2D clusters.

    Args:
        ax: matplotlib axes object.
        clf: SVC classifier.
        X: 2D numpy array with shape (n_samples, 2).
        cmap: matplotlib colormap object.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    margin = clf.decision_function(grid)
    margin = np.abs(margin)
    margin = margin.reshape(xx.shape)
    margin /= margin.max()

    margin_probs = np.exp(-margin)
    contour = ax.contourf(xx, yy, margin_probs, levels=200, cmap=cmap, alpha=0.8)

    cbar = plt.colorbar(contour, ax=ax, label="Uncertainty")
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()


def plot_uncertainty(
    input_1: np.ndarray,
    input_2: np.ndarray,
    ax: plt.Axes = None,
    title: str = "Uncertainty",
    x_label: str = "Feature 1",
    y_label: str = "Feature 2",
    class_labels: tuple[str, str] | None = None,
    cmap_name: str = "coolwarm",
    kernel: Literal["linear", "rbf", "sigmoid", "polynomial"] = "rbf",
    C: float = 0.5,  # noqa: N803
    gamma: float | Literal["auto", "scale"] = "scale",
    show: bool = True,
) -> plt.Axes:
    """Method to plot uncertainty between two 2D clusters.

    Args:
        input_1: First 2D numpy array with shape (n_samples, 2).
        input_2: Second 2D numpy array with shape (n_samples, 2).
        ax: Matplotlib axes object.
        title: Title of plot, defaults to "Uncertainty".
        x_label: Name of x-axis, defaults to "Feature 1".
        y_label: Name of y-axis, defaults to "Feature 2".
        class_labels: Names of classes for legend. Defaults to Class [i], where i is number of class.
        cmap_name: Colormap name, defaults to "coolwarm".
        kernel: Defaults to "rbf". Otherwise, choose "linearr", "polynomial", "sigmoid".
        C: Regularization parameter, defaults to 0.5. The lower, the more tolerant to outliers. Cannot be below 0.0.
        gamma:  Kernel coefficient controlling the influence radius of samples.
                Higher values lead to more local decision boundaries.
        show: Flag to show the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    X = _2_cluster_to_x(input_1, input_2)  # noqa: N806
    y = _2_cluster_to_y(input_1, input_2)

    X = _check_shape(X)  # noqa: N806
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

    cmap = plt.get_cmap(cmap_name)

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
