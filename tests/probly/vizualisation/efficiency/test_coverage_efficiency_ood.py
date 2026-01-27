"""Tests for coverage_efficiency_ood."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pytest

mpl.use("Agg")

from probly.visualization.efficiency.coverage_efficiency_ood import (
    plot_coverage_efficiency_from_ood_labels,
    plot_coverage_efficiency_id_ood,
)


@pytest.fixture
def id_ood_data() -> dict[str, np.ndarray]:
    """Fixture providing ID and OOD data arrays."""
    n_classes = 3
    rng = np.random.default_rng()

    probs_id = rng.random((20, n_classes))
    probs_id /= probs_id.sum(axis=1, keepdims=True)
    targets_id = rng.integers(0, n_classes, size=20)

    probs_ood = rng.random((10, n_classes))
    probs_ood /= probs_ood.sum(axis=1, keepdims=True)
    targets_ood = rng.integers(0, n_classes, size=10)

    return {
        "probs_id": probs_id,
        "targets_id": targets_id,
        "probs_ood": probs_ood,
        "targets_ood": targets_ood,
    }


def test_plot_id_ood_execution(id_ood_data: dict[str, np.ndarray]) -> None:
    """Test plot_coverage_efficiency_id_ood returns Figure and Axes tuple."""
    fig, (ax1, ax2) = plot_coverage_efficiency_id_ood(
        probs_id=id_ood_data["probs_id"],
        targets_id=id_ood_data["targets_id"],
        probs_ood=id_ood_data["probs_ood"],
        targets_ood=id_ood_data["targets_ood"],
        title_id="ID Test",
        title_ood="OOD Test",
    )

    if not isinstance(fig, Figure):
        msg = "Expected Figure object."
        raise AssertionError(msg)  # noqa: TRY004
    if not isinstance(ax1, Axes) or not isinstance(ax2, Axes):
        msg = "Expected Axes objects."
        raise AssertionError(msg)  # noqa: TRY004

    if ax1.get_title() != "ID Test":
        msg = "ID Title mismatch."
        raise AssertionError(msg)
    if ax2.get_title() != "OOD Test":
        msg = "OOD Title mismatch."
        raise AssertionError(msg)


def test_plot_from_ood_labels_execution() -> None:
    """Test splitting logic in plot_coverage_efficiency_from_ood_labels."""
    n_classes = 3
    rng = np.random.default_rng()
    probs = rng.random((10, n_classes))
    probs /= probs.sum(axis=1, keepdims=True)
    targets = rng.integers(0, n_classes, size=10)
    ood_labels = np.array([0] * 5 + [1] * 5)

    fig, axes = plot_coverage_efficiency_from_ood_labels(
        probs=probs, targets=targets, ood_labels=ood_labels, id_label=0, ood_label=1
    )

    if not isinstance(fig, Figure):
        msg = "Expected Figure object."
        raise AssertionError(msg)  # noqa: TRY004
    if len(axes) != 2:
        msg = "Expected 2 Axes."
        raise AssertionError(msg)


def test_validation_raises_on_shape_mismatch() -> None:
    """Test that shape validation raises ValueError."""
    rng = np.random.default_rng()
    probs = rng.random((10, 3))
    targets = rng.integers(0, 3, size=5)

    with pytest.raises(ValueError, match="must agree on the first dimension"):
        plot_coverage_efficiency_id_ood(probs_id=probs, targets_id=targets, probs_ood=probs, targets_ood=targets)


def test_validation_raises_on_1d_probs() -> None:
    """Test Error if probs is not 2D."""
    with pytest.raises(ValueError, match="probs must be a 2D array"):
        plot_coverage_efficiency_id_ood(
            probs_id=np.array([0.1, 0.9]),
            targets_id=np.array([0]),
            probs_ood=np.zeros((1, 2)),
            targets_ood=np.array([0]),
        )


def test_raises_if_no_id_samples() -> None:
    """Test ValueError if all samples are OOD."""
    rng = np.random.default_rng()
    probs = rng.random((5, 3))
    targets = np.zeros(5)
    ood_labels = np.ones(5)

    with pytest.raises(ValueError, match="No ID samples found"):
        plot_coverage_efficiency_from_ood_labels(probs=probs, targets=targets, ood_labels=ood_labels, id_label=0)


def test_raises_if_no_ood_samples() -> None:
    """Test ValueError if all samples are ID."""
    rng = np.random.default_rng()
    probs = rng.random((5, 3))
    targets = np.zeros(5)
    ood_labels = np.zeros(5)

    with pytest.raises(ValueError, match="No OOD samples found"):
        plot_coverage_efficiency_from_ood_labels(probs=probs, targets=targets, ood_labels=ood_labels, ood_label=1)
