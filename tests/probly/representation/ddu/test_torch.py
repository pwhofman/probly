"""Tests for torch DDU representations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.decider import categorical_from_mean  # noqa: E402
from probly.representation.ddu import DDURepresentation, create_ddu_representation  # noqa: E402
from probly.representation.ddu.torch import TorchDDURepresentation  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402


def test_torch_ddu_representation_holds_softmax_and_densities() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.2, 0.8]]))
    densities = torch.tensor([[1.0, 2.0]])

    representation = TorchDDURepresentation(softmax, densities)

    assert isinstance(representation, DDURepresentation)
    assert representation.softmax is softmax
    assert representation.densities is densities


def test_torch_ddu_factory_creates_representation() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.3, 0.7]]))
    densities = torch.tensor([[2.0, 1.0]])

    representation = create_ddu_representation(softmax, densities)

    assert isinstance(representation, TorchDDURepresentation)
    assert representation.softmax is softmax
    assert representation.densities is densities


def test_categorical_from_mean_reduces_torch_ddu_to_softmax_distribution() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.4, 0.6]]))
    representation = TorchDDURepresentation(softmax, torch.tensor([[3.0, 1.0]]))

    assert categorical_from_mean(representation) is softmax
