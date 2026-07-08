"""Tests for torch Mahalanobis representations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.decider import categorical_from_mean  # noqa: E402
from probly.method.mahalanobis import MahalanobisRepresentation, create_mahalanobis_representation  # noqa: E402
from probly.method.mahalanobis.torch import TorchMahalanobisRepresentation  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution  # noqa: E402


def test_torch_mahalanobis_representation_holds_softmax_scores_and_combiner() -> None:
    softmax = TorchProbabilityCategoricalDistribution(torch.tensor([[0.2, 0.8]]))
    layer_scores = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([-1.0, -1.0])
    bias = torch.tensor(0.0)

    representation = TorchMahalanobisRepresentation(softmax, layer_scores, weight, bias)

    assert isinstance(representation, MahalanobisRepresentation)
    assert representation.softmax is softmax
    assert representation.layer_scores is layer_scores
    assert representation.weight is weight
    assert representation.bias is bias


def test_torch_mahalanobis_factory_creates_representation() -> None:
    softmax = TorchProbabilityCategoricalDistribution(torch.tensor([[0.3, 0.7]]))
    layer_scores = torch.tensor([[2.0, 1.0]])
    weight = torch.tensor([-1.0, -1.0])
    bias = torch.tensor(0.5)

    representation = create_mahalanobis_representation(softmax, layer_scores, weight, bias)

    assert isinstance(representation, TorchMahalanobisRepresentation)
    assert representation.softmax is softmax
    assert representation.layer_scores is layer_scores
    assert representation.weight is weight
    assert representation.bias is bias


def test_categorical_from_mean_reduces_torch_mahalanobis_to_softmax_distribution() -> None:
    softmax = TorchProbabilityCategoricalDistribution(torch.tensor([[0.4, 0.6]]))
    representation = TorchMahalanobisRepresentation(
        softmax,
        torch.tensor([[3.0, 1.0]]),
        torch.tensor([-1.0, -1.0]),
        torch.tensor(0.0),
    )

    assert categorical_from_mean(representation) is softmax
