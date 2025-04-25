"""Test for the subensemble module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.representation import SubEnsemble


@pytest.fixture
def subensemble() -> SubEnsemble:
    base = nn.Sequential(nn.Conv2d(3, 5, 5), nn.ReLU(), nn.Flatten())
    head = nn.Linear(5, 2)
    return SubEnsemble(base, 5, head)


def test_forward_shape(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble(x)
    assert output.shape == (2, 2)


def test_predict_pointwise_shape(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_pointwise(x)
    assert output.shape == (2, 2)


def test_predict_pointwise_logits_shape(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_pointwise(x, logits=True)
    assert output.shape == (2, 2)


def test_predict_pointwise_probabilities(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_pointwise(x)
    assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))


def test_predict_representation_shape(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_representation(x)
    assert output.shape == (2, 5, 2)


def test_predict_representation_logits_shape(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_representation(x, logits=True)
    assert output.shape == (2, 5, 2)


def test_predict_representation_probabilities(subensemble: SubEnsemble) -> None:
    x = torch.randn(2, 3, 5, 5)
    output = subensemble.predict_representation(x)
    assert torch.allclose(output.sum(dim=2), torch.ones_like(output[:, :, 0]))


def test_models_independent(subensemble: SubEnsemble) -> None:
    def flatten_params(model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([param.flatten() for param in model.parameters()])

    weights = [flatten_params(model) for model in subensemble.models]
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            assert not torch.equal(weights[i], weights[j])


def test_base_frozen_and_head_not(subensemble: SubEnsemble) -> None:
    for model in subensemble.models:
        for param in model[0].parameters():
            assert not param.requires_grad
        for param in model[1].parameters():
            assert param.requires_grad
