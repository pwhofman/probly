"""Tests for torch DDU representations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.predictor import to_single_prediction  # noqa: E402
from probly.representation import CanonicalRepresentation  # noqa: E402
from probly.representation.ddu import DDURepresentation, create_ddu_representation  # noqa: E402
from probly.representation.ddu.torch import TorchDDURepresentation  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402


def test_torch_ddu_representation_is_canonical_with_softmax_element() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.2, 0.8]]))
    densities = torch.tensor([[1.0, 2.0]])

    representation = TorchDDURepresentation(softmax, densities)

    assert isinstance(representation, DDURepresentation)
    assert isinstance(representation, CanonicalRepresentation)
    assert representation.canonical_element is softmax
    assert representation.densities is densities


def test_torch_ddu_factory_creates_canonical_representation() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.3, 0.7]]))
    densities = torch.tensor([[2.0, 1.0]])

    representation = create_ddu_representation(softmax, densities)

    assert isinstance(representation, TorchDDURepresentation)
    assert representation.canonical_element is softmax
    assert representation.densities is densities


def test_to_single_prediction_reduces_torch_ddu_to_softmax_distribution() -> None:
    softmax = TorchCategoricalDistribution(torch.tensor([[0.4, 0.6]]))
    representation = TorchDDURepresentation(softmax, torch.tensor([[3.0, 1.0]]))

    assert to_single_prediction(representation) is softmax
