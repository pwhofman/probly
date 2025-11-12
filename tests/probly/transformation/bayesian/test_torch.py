# tests/probly/transformation/bayesian/test_torch.py
from __future__ import annotations

import pytest
import torch
from torch import nn

from probly.layers.torch import BayesLinear
from probly.transformation.bayesian import bayesian


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        """Simple linear model for testing Bayesian transform."""
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def model():
    return SimpleNet()


def test_bayesian_transform_replaces_linear_but_returns_mean_only(model):
    bayesian_model = bayesian(model)
    assert isinstance(bayesian_model.linear, BayesLinear), "Linear not replaced"

    x = torch.randn(5, 2)
    output = bayesian_model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 1)
    assert not torch.isnan(output).any()
