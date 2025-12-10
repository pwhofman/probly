"""Implementation for Reliability Diagrams."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def compute_reliability_diagram(probabilities: np.ndarray, labels: np.ndarray, n_bins: int =10) -> dict:
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

    bins = np.linspace(0.0, 1.0, n_bins+1)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins-1)

    for b in range(n_bins):
        indices = np.where(bin_indices == b)[0]

        if(len(indices) == 0):
            diagram["bin_accuracies"].append(np.float64(0.0))
            diagram["bin_confidences"].append(np.float64(0.0))
            diagram["bin_counts"].append(0)
            continue

        accuracy = np.mean(predictions[indices] == labels[indices])
        confidence = np.mean(confidences[indices])

        diagram["bin_accuracies"].append(accuracy)
        diagram["bin_confidences"].append(confidence)
        diagram["bin_counts"].append(len(indices))

    return diagram


def plot_reliability_diagram(diagram: dict, title: str ="Model Calibration") -> tuple[plt.figure, plt.axes]:
    """Plots the diagram calculated with `compute_reliability_diagram`.

    Args:
        diagram: A dictionary containing the fields ["n_bins"]["bin_accuracies"]["bin_confidences"]["bin_counts"]
        title: The title the diagram should have

    Returns:
        fig: The matplotlib figure
        axis: The matplotlib axis
    """
    n_bins = diagram["n_bins"]
    x = np.linspace(0.0, 1.0, n_bins+1)

    fig, axis = plt.subplots()

    axis.set_title(title)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_axisbelow(True)
    axis.grid(True, linestyle="--")
    axis.plot(x, x, "k--")

    accuracies = diagram["bin_accuracies"]
    for b, acc in zip(x, accuracies, strict=False):
        axis.bar(b, acc, 1/n_bins, align="edge", linewidth=1.0, edgecolor="#004f8f", color="#008cff")

    return fig, axis
