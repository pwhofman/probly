"""Implementation for Reliability Diagrams."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np


def compute_reliability_diagram(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Calculates the bins for a reliability diagram.

    Args:
        probabilities: The probabilities from the model
        labels: The labels corresponding to the probabilities
        n_bins: The number of bins the intervall [0, 1] should be divided into

    Returns:
        diagram: An dict object of the form ["n_bins"]["bin_accuracies"]["bin_confidences"]["bin_counts"]
        where
            - ["n_bins contains"] the number of bins
            - ["bin_accuracies"] contains the mean accuracies for all bins
            - ["bin_confidences"] contains the mean confidences for all bins
            - ["bin_counts"] contains the number of predictions in every bin
    """
    diagram = {"n_bins": n_bins, "bin_accuracies": [], "bin_confidences": [], "bin_counts": []}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    for b in range(n_bins):
        indices = np.where(bin_indices == b)[0]

        if len(indices) == 0:
            cast("list", diagram["bin_accuracies"]).append(np.float64(0.0))
            cast("list", diagram["bin_confidences"]).append(np.float64(0.0))
            cast("list", diagram["bin_counts"]).append(0)
            continue

        accuracy = np.mean(predictions[indices] == labels[indices])
        confidence = np.mean(confidences[indices])

        cast("list", diagram["bin_accuracies"]).append(accuracy)
        cast("list", diagram["bin_confidences"]).append(confidence)
        cast("list", diagram["bin_counts"]).append(len(indices))

    return diagram


def plot_reliability_diagram(diagram: dict, title: str = "Model Calibration") -> tuple[plt.figure, plt.axes]:
    """Plots the diagram calculated with `compute_reliability_diagram`.

    Args:
        diagram: A dictionary containing the fields ["n_bins"]["bin_accuracies"]["bin_confidences"]["bin_counts"]
        title: The title the diagram should have

    Returns:
        fig: The matplotlib figure
        ax: The matplotlib axis
    """
    n_bins = diagram["n_bins"]
    accs = np.array(diagram["bin_accuracies"], dtype=float)
    confs = np.array(diagram["bin_confidences"], dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    lefts = edges[:-1]
    width = edges[1] - edges[0]

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    # Blue bars
    ax.bar(
        lefts,
        accs,
        width=width,
        align="edge",
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.2,
        label="Outputs",
    )

    gap = np.abs(accs - confs)
    bottom = np.minimum(accs, confs)

    # Red bars
    ax.bar(
        lefts,
        gap,
        bottom=bottom,
        width=width,
        align="edge",
        facecolor="red",
        edgecolor="red",
        linewidth=1.0,
        hatch="/",
        alpha=0.30,
        label="Calibration gap",
    )

    ax.legend()
    return fig, ax
