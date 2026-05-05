"""Tests for the maximin categorical decider (torch backend)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.decider import categorical_from_maximin
from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)


def test_maximin_picks_argmax_of_lower_probability_for_torch_convex_credal_set() -> None:
    vertices = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]], dtype=torch.float64)
    credal_set = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))

    decision = categorical_from_maximin(credal_set)

    assert isinstance(decision, TorchCategoricalDistribution)
    assert torch.allclose(decision.probabilities, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_maximin_handles_batched_torch_convex_credal_set() -> None:
    vertices = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]],
            [[0.1, 0.7, 0.2], [0.2, 0.3, 0.5]],
        ],
        dtype=torch.float64,
    )
    credal_set = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))

    decision = categorical_from_maximin(credal_set)

    assert isinstance(decision, TorchCategoricalDistribution)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(decision.probabilities, expected)


def test_maximin_breaks_ties_by_picking_first_index_for_torch_convex_credal_set() -> None:
    vertices = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
    credal_set = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))

    decision = categorical_from_maximin(credal_set)

    assert torch.allclose(decision.probabilities, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))


def test_maximin_returns_one_hot_distribution_for_torch_convex_credal_set() -> None:
    vertices = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]],
            [[0.1, 0.7, 0.2], [0.2, 0.3, 0.5]],
        ],
        dtype=torch.float64,
    )
    credal_set = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))

    decision = categorical_from_maximin(credal_set)

    assert torch.allclose(decision.probabilities.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert float(decision.probabilities.max()) == pytest.approx(1.0)
