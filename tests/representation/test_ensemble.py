"""Test for the ensemble module."""

from __future__ import annotations

import pytest
import torch

from probly.representation import Ensemble


@pytest.fixture
def ensemble(conv_linear_model: torch.nn.Module) -> Ensemble:
    return Ensemble(conv_linear_model, 5)


def test_forward_shape(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble(x)
    assert output.shape == (2, 2)


def test_predict_pointwise_shape(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_pointwise(x)
    assert output.shape == (2, 2)


def test_predict_pointwise_logits_shape(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_pointwise(x, logits=True)
    assert output.shape == (2, 2)


def test_predict_pointwise_probabilities(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_pointwise(x)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))


def test_predict_representation_shape(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_representation(x)
    assert output.shape == (2, 5, 2)


def test_predict_representation_logits_shape(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_representation(x, logits=True)
    assert output.shape == (2, 5, 2)


def test_predict_representation_probabilities(ensemble: Ensemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = ensemble.predict_representation(x)
    assert torch.allclose(output.sum(dim=2), torch.ones_like(output[:, :, 0]))


def test_models_independent(ensemble: Ensemble) -> None:
    def flatten_params(model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([param.flatten() for param in model.parameters()])

    weights = [flatten_params(model) for model in ensemble.models]
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            assert not torch.equal(weights[i], weights[j])
