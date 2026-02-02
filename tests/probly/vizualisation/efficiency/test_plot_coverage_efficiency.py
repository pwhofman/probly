"""Tests for plot_coverage_efficiency."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

mpl.use("Agg")

from probly.visualization.efficiency.plot_coverage_efficiency import CoverageEfficiencyVisualizer


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Fixture providing valid probability distributions and targets."""
    n_samples = 50
    n_classes = 5
    rng = np.random.default_rng()
    probs = rng.random((n_samples, n_classes))
    probs /= probs.sum(axis=1, keepdims=True)
    targets = rng.integers(0, n_classes, size=n_samples)
    return probs, targets


def test_visualizer_initialization() -> None:
    """Test that the visualizer class can be instantiated."""
    viz = CoverageEfficiencyVisualizer()
    if not isinstance(viz, CoverageEfficiencyVisualizer):
        msg = "Failed to instantiate visualizer."
        raise AssertionError(msg)  # noqa: TRY004


def test_plot_coverage_efficiency_execution(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test standard execution returns a matplotlib Axes object."""
    probs, targets = sample_data
    viz = CoverageEfficiencyVisualizer()

    ax = viz.plot_coverage_efficiency(probs, targets)

    if not isinstance(ax, Axes):
        msg = "Expected matplotlib Axes object."
        raise AssertionError(msg)  # noqa: TRY004

    if ax.get_title() != "Coverage vs. Efficiency":
        msg = "Title mismatch."
        raise AssertionError(msg)

    if "Empirical Coverage" not in ax.get_ylabel():
        msg = "Y-label mismatch."
        raise AssertionError(msg)
    if "Target Confidence Level" not in ax.get_xlabel():
        msg = "X-label mismatch."
        raise AssertionError(msg)


def test_plot_coverage_efficiency_custom_title(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that custom title is applied."""
    probs, targets = sample_data
    viz = CoverageEfficiencyVisualizer()

    custom_title = "My Custom Metric Plot"
    ax = viz.plot_coverage_efficiency(probs, targets, title=custom_title)

    if ax.get_title() != custom_title:
        msg = f"Title mismatch. Got: {ax.get_title()}"
        raise AssertionError(msg)


def test_plot_coverage_efficiency_existing_ax(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test plotting onto an existing Axes object."""
    probs, targets = sample_data
    viz = CoverageEfficiencyVisualizer()

    _fig, ax_input = plt.subplots()
    ax_output = viz.plot_coverage_efficiency(probs, targets, ax=ax_input)

    if ax_output is not ax_input:
        msg = "Output axes should be the same object as input axes."
        raise AssertionError(msg)

    if len(ax_output.get_lines()) == 0:
        msg = "No lines were plotted."
        raise AssertionError(msg)


def test_compute_metrics_logic(sample_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Integration test for internal metric computation logic."""
    probs, targets = sample_data
    viz = CoverageEfficiencyVisualizer()
    alphas = np.linspace(0.1, 0.9, 5)

    coverages, efficiencies = viz._compute_metrics(probs, targets, alphas)  # noqa: SLF001

    if len(coverages) != len(alphas):
        msg = "Coverages length mismatch."
        raise AssertionError(msg)
    if len(efficiencies) != len(alphas):
        msg = "Efficiencies length mismatch."
        raise AssertionError(msg)

    if not (np.all(coverages >= 0.0) and np.all(coverages <= 1.0)):
        msg = "Coverages out of bounds [0, 1]."
        raise AssertionError(msg)

    n_classes = probs.shape[1]
    if not (np.all(efficiencies >= 1.0) and np.all(efficiencies <= n_classes)):
        msg = f"Efficiencies out of bounds [1, {n_classes}]."
        raise AssertionError(msg)
