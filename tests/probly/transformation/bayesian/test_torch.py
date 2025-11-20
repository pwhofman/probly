# tests/probly/transformation/bayesian/test_torch.py
from __future__ import annotations

import torch
from torch import nn

from probly.layers.torch import BayesLinear
from probly.transformation.bayesian import bayesian


def test_bayesian_transform_replaces_linear_but_returns_mean_only(torch_model_small_2d_2d):
    model = torch_model_small_2d_2d
    bayesian_model = bayesian(model)

    for name, module in bayesian_model.named_modules():
        if isinstance(module, nn.Linear):
            assert isinstance(module, BayesLinear), f"{name} not converted!"

    x = torch.randn(5, 2)
    output = bayesian_model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
