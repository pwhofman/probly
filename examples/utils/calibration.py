from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score, summed over classes and averaged over samples."""
    one_hot = np.eye(probs.shape[-1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean top-label confidence and accuracy; empty bins are NaN."""
    confidence = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        if mask.any():
            bin_conf[b] = confidence[mask].mean()
            bin_acc[b] = correct[mask].mean()
    return bin_conf, bin_acc


def plot_reliability_diagram(
    curves: dict[str, np.ndarray],
    labels: np.ndarray,
    title: str,
    n_bins: int = 15,
) -> Figure:
    """Draw top-label reliability curves, one per ``legend label -> probabilities`` entry."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for (name, probs), marker in zip(curves.items(), ("o-", "s-", "^-", "D-"), strict=False):
        conf, acc = reliability_curve(probs, labels, n_bins)
        ax.plot(conf, acc, marker, label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def binary_to_two_class(prob_positive: np.ndarray) -> np.ndarray:
    """Stack ``P(y=1)`` into two-column class probabilities ``[P(y=0), P(y=1)]``."""
    prob_positive = np.asarray(prob_positive).reshape(-1)
    return np.column_stack([1.0 - prob_positive, prob_positive])
