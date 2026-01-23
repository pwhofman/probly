"""Tests main API for clustermargin."""

from __future__ import annotations

import re
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from probly.visualization.clustermargin.clustervisualizer import plot_uncertainty

# Use a non-interactive backend so tests do not open GUI windows.
mpl.use("Agg")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def create_cluster1(rng):
    mu = [0, 0]
    sigma = [[0.05, 0], [0, 0.5]]

    return rng.multivariate_normal(mu, sigma, size=10)


@pytest.fixture
def create_cluster2(rng):
    mu = [1, 1]
    sigma = [[0.05, 0], [0, 0.5]]

    return rng.multivariate_normal(mu, sigma, size=10)


def test_plot_uncertainty_show_false(create_cluster1, create_cluster2):
    """Basic integration test for main API, when show is True."""
    ax = plot_uncertainty(create_cluster1, create_cluster2, show=False)

    assert isinstance(ax, plt.Axes)

    assert "kernel:" in ax.get_title()
    assert "C:" in ax.get_title()


def test_plot_uncertainty_show_true(create_cluster1, create_cluster2):
    """Basic integration test for main API, when show is False."""
    with patch("matplotlib.pyplot.show") as mock_show:
        ax = plot_uncertainty(create_cluster1, create_cluster2, show=True)

        mock_show.assert_called_once()

    assert isinstance(ax, plt.Axes)


def test_plot_uncertainty_default_title(create_cluster1, create_cluster2):
    """Tests the default title."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        show=False,
        kernel="rbf",
        C=0.5,
    )

    assert ax.get_title() == "Uncertainty (kernel: rbf, C: 0.5)"


def test_plot_uncertainty_custom_title(create_cluster1, create_cluster2):
    """Tests the custom title."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        title="Custom title",
        show=False,
        kernel="rbf",
        C=0.5,
    )
    test_title = "Custom title (kernel: rbf, C: 0.5)"
    assert ax.get_title() == test_title


def test_plot_uncertainty_default_x_label(create_cluster1, create_cluster2):
    """Tests the default x label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        show=False,
    )
    assert ax.get_xlabel() == "Feature 1"


def test_plot_uncertainty_custom_x_label(create_cluster1, create_cluster2):
    """Tests the custom x label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        x_label="Custom Feature",
        show=False,
    )
    test_label = "Custom Feature"
    assert ax.get_xlabel() == test_label


def test_plot_uncertainty_default_y_label(create_cluster1, create_cluster2):
    """Tests the default y label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        show=False,
    )
    assert ax.get_ylabel() == "Feature 2"


def test_plot_uncertainty_custom_y_label(create_cluster1, create_cluster2):
    """Tests the custom y label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        y_label="Custom Feature",
        show=False,
    )
    test_label = "Custom Feature"
    assert ax.get_ylabel() == test_label


def test_plot_uncertainty_default_class_labels(create_cluster1, create_cluster2):
    """Tests the default class label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None

    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Class 1", "Class 2"]


def test_plot_uncertainty_custom_class_labels(create_cluster1, create_cluster2):
    """Tests the custom class label."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        class_labels=("Test Class A", "Test Class B"),
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None

    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Test Class A", "Test Class B"]


@pytest.mark.parametrize("kernel", ["linear", "rbf", "sigmoid"])
def test_plot_uncertainty_supported_kernels(create_cluster1, create_cluster2, kernel):
    """Tests the supported kernels."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        kernel=kernel,
        show=False,
    )

    assert ax is not None


def test_plot_uncertainty_valid_C(create_cluster1, create_cluster2):  # noqa: N802
    """Tests the valid C input."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        C=1.0,
        show=False,
    )

    assert ax is not None


def test_plot_uncertainty_invalid_C(create_cluster1, create_cluster2):  # noqa: N802
    """Tests that invalid C input return ValueError."""
    with pytest.raises(ValueError, match=re.escape("C has to be > 0.0")):
        plot_uncertainty(create_cluster1, create_cluster2, C=-1.0, show=False)


@pytest.mark.parametrize("gamma", ["scale", "auto"])
def test_plot_uncertainty_valid_gamma_string(create_cluster1, create_cluster2, gamma):
    """Tests the supported gamma input. Only the strings."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        gamma=gamma,
        show=False,
    )

    assert ax is not None


@pytest.mark.parametrize("gamma", [0.1, 1.0, 10.0])
def test_plot_uncertainty_gamma_float(create_cluster1, create_cluster2, gamma):
    """Tests the supported gamma input. Exemplary float input."""
    ax = plot_uncertainty(
        create_cluster1,
        create_cluster2,
        gamma=gamma,
        show=False,
    )

    assert ax is not None


def test_plot_uncertainty_invalid_gamma_float(create_cluster1, create_cluster2):
    """Raises ValueError if gamma is < 0.0."""
    with pytest.raises(ValueError, match=re.escape("gamma has to be >= 0.0 or one of {'auto', 'scale'}")):
        plot_uncertainty(create_cluster1, create_cluster2, gamma=-0.1, show=False)


def test_plot_uncertainty_invalid_gamma_string(create_cluster1, create_cluster2):
    """Raises ValueError if gamma is neither auto nor scale."""
    with pytest.raises(ValueError, match=re.escape("gamma has to be >= 0.0 or one of {'auto', 'scale'}")):
        plot_uncertainty(create_cluster1, create_cluster2, gamma="wrong_string", show=False)
