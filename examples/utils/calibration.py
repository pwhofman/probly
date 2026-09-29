from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from probly.metrics import expected_calibration_error
from probly.plot import PlotConfig


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class."""
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score, summed over classes and averaged over samples."""
    one_hot = np.eye(probs.shape[-1])[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean top-label confidence, accuracy, and sample count; empty bins are NaN."""
    confidence = probs.max(-1)
    correct = (probs.argmax(-1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (confidence > edges[b]) & (confidence <= edges[b + 1])
        bin_count[b] = mask.sum()
        if mask.any():
            bin_conf[b] = confidence[mask].mean()
            bin_acc[b] = correct[mask].mean()
    return bin_conf, bin_acc, bin_count


def plot_reliability_diagram(
    curves: dict[str, np.ndarray],
    labels: np.ndarray,
    title: str,
    n_bins: int = 15,
) -> Figure:
    """Draw Guo et al. (2017) style reliability diagrams, one column per ``method name -> probabilities`` entry.

    The top row is the confidence histogram (share of samples per bin, with the accuracy and the
    average confidence marked), the bottom row the reliability diagram: per-bin accuracy bars
    ("Outputs") and the gap to the bin's mean confidence ("Gap"), annotated with the ECE in percent.
    """
    config = PlotConfig()
    blue, red = config.categorical_palette[:2]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    width = 1.0 / n_bins
    fig, axes = plt.subplots(
        2,
        len(curves),
        figsize=(3.6 * len(curves), 5.6),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for col, (name, probs) in enumerate(curves.items()):
        ax_hist, ax_rel = axes[0, col], axes[1, col]
        conf, acc, count = reliability_curve(probs, labels, n_bins)
        confidence = probs.max(-1)
        accuracy = float((probs.argmax(-1) == labels).mean())
        ece = float(expected_calibration_error(probs, labels, num_bins=n_bins))

        ax_hist.bar(edges[:-1], 100 * count / count.sum(), width=width, align="edge", color=blue, edgecolor="black")
        ax_hist.axvline(accuracy, color=red, linestyle="--", label="Accuracy")
        ax_hist.axvline(float(confidence.mean()), color=config.color_neutral, linestyle="--", label="Avg. confidence")
        ax_hist.set_title(name)
        ax_hist.legend(loc="upper left", fontsize="small")

        filled = count > 0
        ax_rel.bar(
            edges[:-1][filled], acc[filled], width=width, align="edge", color=blue, edgecolor="black", label="Outputs"
        )
        ax_rel.bar(
            edges[:-1][filled],
            (conf - acc)[filled],
            bottom=acc[filled],
            width=width,
            align="edge",
            color=red,
            alpha=config.fill_alpha,
            edgecolor=red,
            hatch="//",
            label="Gap",
        )
        ax_rel.plot([0, 1], [0, 1], color=config.color_neutral, linestyle="--")
        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_aspect("equal")
        ax_rel.set_xlabel("Confidence")
        ax_rel.text(
            0.95,
            0.05,
            f"ECE={100 * ece:.2f}",
            ha="right",
            va="bottom",
            transform=ax_rel.transAxes,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax_rel.legend(loc="upper left")

    axes[0, 0].set_ylabel("% of Samples")
    axes[1, 0].set_ylabel("Accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def binary_to_two_class(prob_positive: np.ndarray) -> np.ndarray:
    """Stack ``P(y=1)`` into two-column class probabilities ``[P(y=0), P(y=1)]``."""
    prob_positive = np.asarray(prob_positive).reshape(-1)
    return np.column_stack([1.0 - prob_positive, prob_positive])
