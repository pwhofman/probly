"""Tests for utils.torch functions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch.utils.data import DataLoader, TensorDataset

from probly.utils.torch import (
    dirichlet_entropy,
    temperature_softmax,
    torch_collect_outputs,
    torch_head_dimension,
    torch_reset_all_parameters,
)


def test_torch_reset_all_parameters(torch_conv_linear_model: torch.nn.Module) -> None:
    def flatten_params(model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([param.flatten() for param in model.parameters()])

    before = flatten_params(torch_conv_linear_model)
    torch_reset_all_parameters(torch_conv_linear_model)
    after = flatten_params(torch_conv_linear_model)
    assert not torch.equal(before, after)


def test_torch_collect_outputs(torch_conv_linear_model: torch.nn.Module) -> None:
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 5, 5),
            torch.randn(
                2,
            ),
        ),
    )
    outputs, targets = torch_collect_outputs(torch_conv_linear_model, loader, torch.device("cpu"))
    assert outputs.shape == (2, 2)
    assert targets.shape == (2,)


def test_temperature_softmax() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(temperature_softmax(x, 2.0), torch.softmax(x / 2.0, dim=1))
    assert torch.equal(temperature_softmax(x, torch.tensor(1.0)), torch.softmax(x, dim=1))


def test_dirichlet_entropy() -> None:
    alphas = torch.tensor([[10.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    entropy = dirichlet_entropy(alphas)
    torch.testing.assert_close(entropy, torch.distributions.Dirichlet(alphas).entropy())
    assert entropy[1] > entropy[0]


def test_head_dimension_accepts_custom_module_and_integer_like_values() -> None:
    class CustomHead(torch.nn.Module):
        out_features = np.int64(7)

    head = CustomHead()
    assert torch_head_dimension(head, "out_features") == 7
    with pytest.raises(TypeError, match="in_features"):
        torch_head_dimension(head, "in_features")


def test_reset_skips_non_callable_reset_parameters() -> None:
    class CustomModule(torch.nn.Module):
        reset_parameters = None

    torch_reset_all_parameters(CustomModule())
