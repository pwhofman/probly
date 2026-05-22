"""Torch-backend credal plotting tests (binary and ternary).

The array-backend equivalents live in ``test_binary.py`` and ``test_ternary.py``.
Torch is an optional dependency, so this whole module is skipped when it is not
installed; the array tests stay backend-free and keep running regardless.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.plot import plot_credal_set  # noqa: E402
from probly.representation.credal_set.torch import (  # noqa: E402
    TorchConvexCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import (  # noqa: E402
    TorchProbabilityCategoricalDistribution,
)


@pytest.mark.usefixtures("_close_figures")
class TestTorchBinaryPlot:
    """Binary (2-class) torch credal plotting."""

    def test_intervals_binary_default_labels(self) -> None:
        data = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.3]]),
            upper_bounds=torch.tensor([[0.6, 0.8]]),
        )
        ax = plot_credal_set(data)
        assert ax.get_xlabel() == "Probability of Class 1"

    def test_intervals_binary_zero_width_renders_point(self) -> None:
        data = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.5, 0.5]]),
            upper_bounds=torch.tensor([[0.5, 0.5]]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_distance_based_binary(self) -> None:
        data = TorchDistanceBasedCredalSet(
            nominal=TorchProbabilityCategoricalDistribution(torch.tensor([[0.4, 0.6]])),
            radius=torch.tensor([0.1]),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_convex_binary(self) -> None:
        data = TorchConvexCredalSet(
            tensor=TorchProbabilityCategoricalDistribution(torch.tensor([[[0.3, 0.7], [0.5, 0.5], [0.4, 0.6]]])),
        )
        ax = plot_credal_set(data)
        assert ax is not None

    def test_series_labels_show_legend(self) -> None:
        data = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.3], [0.0, 0.4]]),
            upper_bounds=torch.tensor([[0.6, 0.9], [0.5, 1.0]]),
        )
        ax = plot_credal_set(data, series_labels=["A", "B"])
        legend = ax.get_legend()
        assert legend is not None


@pytest.mark.usefixtures("_close_figures")
class TestTorchTernaryPlot:
    """Ternary (3-class) torch credal plotting."""

    def test_intervals_ternary(self) -> None:
        data = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.1, 0.1, 0.1]]),
            upper_bounds=torch.tensor([[0.7, 0.7, 0.7]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_intervals_ternary_zero_width_renders_point(self) -> None:
        data = TorchProbabilityIntervalsCredalSet(
            lower_bounds=torch.tensor([[0.4, 0.3, 0.3]]),
            upper_bounds=torch.tensor([[0.4, 0.3, 0.3]]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_distance_based_ternary(self) -> None:
        data = TorchDistanceBasedCredalSet(
            nominal=TorchProbabilityCategoricalDistribution(torch.tensor([[0.4, 0.3, 0.3]])),
            radius=torch.tensor([0.1]),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary(self) -> None:
        data = TorchConvexCredalSet(
            tensor=TorchProbabilityCategoricalDistribution(
                torch.tensor([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]])
            ),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary_two_points(self) -> None:
        data = TorchConvexCredalSet(
            tensor=TorchProbabilityCategoricalDistribution(torch.tensor([[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]])),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name

    def test_convex_ternary_single_point(self) -> None:
        data = TorchConvexCredalSet(
            tensor=TorchProbabilityCategoricalDistribution(torch.tensor([[[0.6, 0.2, 0.2]]])),
        )
        ax = plot_credal_set(data)
        assert "ternary" in ax.name
