"""Coverage-Efficiency visualization for ID vs. OOD data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.efficiency.plot_coverage_efficiency import CoverageEfficiencyVisualizer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _validate_shapes(
    probs: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray | None = None,
) -> None:
    """Validate that inputs have compatible shapes.

    Args:
        probs: Array of shape (N, C) with class probabilities.
        targets: Array of shape (N,) with integer class indices.
        labels: Optional array of shape (N,) with ID/OOD markers.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if probs.ndim != 2:
        msg = "probs must be a 2D array of shape (N, C)."
        raise ValueError(msg)

    if targets.ndim != 1:
        msg = "targets must be a 1D array of shape (N,)."
        raise ValueError(msg)

    if probs.shape[0] != targets.shape[0]:
        msg = "probs and targets must agree on the first dimension N."
        raise ValueError(msg)

    if labels is not None:
        if labels.ndim != 1:
            msg = "labels must be a 1D array of shape (N,)."
            raise ValueError(msg)
        if labels.shape[0] != probs.shape[0]:
            msg = "labels must have the same length N as probs/targets."
            raise ValueError(msg)


def _scores_from_probs(probs: np.ndarray) -> np.ndarray:
    """Compute a per-sample score from class probabilities (max confidence).

    Args:
        probs: Array of shape (N, C) with class probabilities.

    Returns:
        Array of shape (N,) with max probability per sample.
    """
    return cast("np.ndarray", np.max(probs, axis=1))


def plot_coverage_efficiency_id_ood(
    probs_id: np.ndarray,
    targets_id: np.ndarray,
    probs_ood: np.ndarray,
    targets_ood: np.ndarray,
    *,
    title_id: str = "Coverage vs. Efficiency (ID)",
    title_ood: str = "Coverage vs. Efficiency (OOD)",
    figsize: tuple[float, float] = (14.0, 5.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot Coverage-Efficiency for ID and OOD data side by side.

    This function does not compute ID/OOD splits. It assumes inputs are already split.

    Args:
        probs_id: (N_id, C) class probabilities for in-distribution samples.
        targets_id: (N_id,) integer class labels for in-distribution samples.
        probs_ood: (N_ood, C) class probabilities for out-of-distribution samples.
        targets_ood: (N_ood,) integer class labels for out-of-distribution samples.
            Note: For some OOD datasets "true" targets may be undefined. In that case,
            consider passing proxy labels (e.g. argmax) and interpreting OOD coverage carefully.
        title_id: Title for the ID subplot.
        title_ood: Title for the OOD subplot.
        figsize: Figure size.

    Returns:
        (fig, (ax_id, ax_ood)): Matplotlib Figure and the two Axes objects.
    """
    probs_id = np.asarray(probs_id)
    targets_id = np.asarray(targets_id)
    probs_ood = np.asarray(probs_ood)
    targets_ood = np.asarray(targets_ood)

    _validate_shapes(probs_id, targets_id)
    _validate_shapes(probs_ood, targets_ood)

    viz = CoverageEfficiencyVisualizer()

    fig, axes_arr = plt.subplots(1, 2, figsize=figsize)
    ax_id: Axes = axes_arr[0]
    ax_ood: Axes = axes_arr[1]

    viz.plot_coverage_efficiency(probs_id, targets_id, title=title_id, ax=ax_id)
    viz.plot_coverage_efficiency(probs_ood, targets_ood, title=title_ood, ax=ax_ood)

    fig.tight_layout()
    return fig, (ax_id, ax_ood)


def plot_coverage_efficiency_from_ood_labels(
    probs: np.ndarray | list[list[float]],
    targets: np.ndarray | list[int],
    ood_labels: np.ndarray | list[int] | list[float],
    *,
    id_label: int = 0,
    ood_label: int = 1,
    title_id: str = "Coverage vs. Efficiency (ID)",
    title_ood: str = "Coverage vs. Efficiency (OOD)",
    figsize: tuple[float, float] = (14.0, 5.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Bridge function: plot Coverage-Efficiency using OOD-style label convention.

    This is the main connection point between the OOD evaluation pipeline and the
    coverage-efficiency visualization.

    The OOD API constructs labels like:
        labels = [0 ... 0, 1 ... 1]
    where:
        0 = ID, 1 = OOD

    This function accepts such `ood_labels` and splits `probs`/`targets` accordingly.

    Args:
        probs: (N, C) class probabilities for all samples.
        targets: (N,) integer class labels for all samples.
        ood_labels: (N,) markers indicating ID vs OOD membership.
        id_label: Value in `ood_labels` that marks in-distribution samples (default: 0).
        ood_label: Value in `ood_labels` that marks out-of-distribution samples (default: 1).
        title_id: Title for the ID subplot.
        title_ood: Title for the OOD subplot.
        figsize: Figure size.

    Returns:
        (fig, (ax_id, ax_ood)): Matplotlib Figure and the two Axes objects.

    Raises:
        ValueError: If shapes do not match, or if no samples are found for ID/OOD.
    """
    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)
    labels_arr = np.asarray(ood_labels)

    _validate_shapes(probs_arr, targets_arr, labels_arr)

    id_mask = labels_arr == id_label
    ood_mask = labels_arr == ood_label

    if not np.any(id_mask):
        msg = "No ID samples found."
        raise ValueError(msg)

    if not np.any(ood_mask):
        msg = "No OOD samples found."
        raise ValueError(msg)

    return plot_coverage_efficiency_id_ood(
        probs_arr[id_mask],
        targets_arr[id_mask],
        probs_arr[ood_mask],
        targets_arr[ood_mask],
        title_id=title_id,
        title_ood=title_ood,
        figsize=figsize,
    )


__all__ = [
    "plot_coverage_efficiency_from_ood_labels",
    "plot_coverage_efficiency_id_ood",
]
