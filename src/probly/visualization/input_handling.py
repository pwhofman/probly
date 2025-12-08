"""Input Handling here."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from .plot_2d import IntervalVisualizer
    from .plot_3d import TernaryVisualizer
    #from spider import radar_factory  # Importing the factory function from spider.py
    #from .geometry import CredalVisualizer
except ImportError as e:
    print(  # noqa: T201
        f"Warning: Modules could not be imported. Please ensure all plotting files are present. Error: {e}",
    )


def check_num_classes(input_data: np.ndarray) -> int:
    """Checks number of classes and refers to respective function.

    Args:
    input_data: array with last dimension equal to the number of classes.

    Returns:
    Number of classes.
    """
    n_classes = input_data.shape[-1]
    return n_classes

def check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check.

    Args:
    input_data: 3D tensor.
    """
    msg1 = "Input must be a NumPy Array."
    msg2 = "There must be at least 2 classes."
    msg3 = "The probabilities of each class must sum to 1."
    msg4 = "All probabilities must be positive."
    msg5 = "Input_data must be at least a 2D NumPy Array."
    if not isinstance(input_data, np.ndarray):
        raise ValueError(msg1)
    if input_data.shape[2] <= 1:
        raise ValueError(msg2)
    if not np.allclose(input_data.sum(axis=-1), 1):
        raise ValueError(msg3)
    if (input_data < 0).any():
        raise ValueError(msg4)
    if input_data.ndim < 2:
        raise ValueError(msg5)
    return input_data

def normalize_input(input_data: np.ndarray) -> np.ndarray:
    """Normalize input data.
    Args:
    """
    if input_data.ndim >= 3:
        input_data = input_data.reshape(-1, input_data.shape[-1])
        return input_data
    else:
        return input_data

def dispatch_plot(input_data: np.ndarray, labels: list[str] | None = None) -> None:
    """Selects and executes the correct plotting function based on class count.

    Args:
        input_data: Probabilities array.
        labels: Optional list of class names (only relevant for Spider plot).
    """
    # 1. Validate Input
    input_data = check_shape(input_data)

    # 2. Flatten if 3D: (Models, Samples, Classes) -> (Total Samples, Classes)
    points = normalize_input(input_data)

    n_classes = check_num_classes(points)

    print(f"Detected {n_classes} classes. Selecting visualizer...")  # noqa: T201

    # 3. Dispatch Logic
    if n_classes == 2:
        viz = IntervalVisualizer()
        viz.interval_plot()
        plt.show()

    elif n_classes == 3:
        ter = TernaryVisualizer()
        ax = ter.ternary_plot(points, s=30, alpha=0.6)

        # Draw Convex Hull on top
        ter.plot_convex_hull(points, ax=ax, facecolor="lightgreen", alpha=0.2)
        plt.show()

    else:
        _plot_spider_custom(points, n_classes, labels)


def _plot_spider_custom(points: np.ndarray, n_classes: int, labels: list[str] | None = None) -> None:
    """Internal helper to handle the Spider plot logic using spider.py utils."""
    # If no labels are provided, generate generic ones (C1, C2, ...)
    if labels is None:
        labels = [f"C{i + 1}" for i in range(n_classes)]

    if len(labels) != n_classes:
        msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
        raise ValueError(msg)

    # Calculate Mean Prediction
    mean_probs = np.mean(points, axis=0)

    # Use the factory from spider.py
    theta = radar_factory(n_classes, frame="polygon")

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "radar"})

    # Setup Axis
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_varlabels(labels)

    # Plot the Mean Prediction
    ax.plot(theta, mean_probs, color="b", linewidth=2, label="Mean Prediction")
    ax.fill(theta, mean_probs, facecolor="b", alpha=0.25)

    # Calculate Min/Max for the "Credal Set" area
    min_probs = np.min(points, axis=0)
    max_probs = np.max(points, axis=0)

    # Fill the area between Min and Max (Uncertainty)
    ax.fill_between(theta, min_probs, max_probs, color="green", alpha=0.3, label="Credal Set Range")

    plt.title(f"Spider Plot ({n_classes} Classes)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
