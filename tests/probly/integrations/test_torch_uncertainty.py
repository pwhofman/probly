"""Tests for optional torch-uncertainty bindings."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_uncertainty")
import torch
from torch import nn
import torch_uncertainty.models.wrappers as tu_wrappers

from probly.predictor import predict
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample
from probly.representer import representer


def _linear_model() -> nn.Module:
    model = nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.zero_()
    return model


def test_torch_uncertainty_deep_ensemble_predict_returns_sample() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    sample = predict(model, x)

    assert isinstance(sample, TorchSample)
    assert sample.sample_dim == 0
    assert sample.tensor.shape == (2, 4, 3)


def test_torch_uncertainty_deep_ensemble_logit_representer() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    representation = representer(model, kind="logits")(x)

    assert isinstance(representation, TorchCategoricalDistributionSample)
    assert isinstance(representation.tensor, TorchLogitCategoricalDistribution)
    assert representation.sample_dim == 0
    assert representation.tensor.logits.shape == (2, 4, 3)


def test_torch_uncertainty_deep_ensemble_value_representer() -> None:
    model = tu_wrappers.deep_ensembles([_linear_model(), _linear_model()])
    x = torch.ones(4, 2)

    representation = representer(model)(x)

    assert isinstance(representation, TorchSample)
    assert representation.tensor.shape == (2, 4, 3)
