"""Tests for utils.torch functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.utils.torch import (
    torch_entropy,
    torch_head_dimension,
    torch_temperature_softmax,
)


def test_temperature_softmax() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(torch_temperature_softmax(x, 2.0), torch.softmax(x / 2.0, dim=1))
    assert torch.equal(torch_temperature_softmax(x, torch.tensor(1.0)), torch.softmax(x, dim=1))


def test_head_dimension_accepts_custom_module_and_integer_like_values() -> None:
    class CustomHead(torch.nn.Module):
        out_features = np.int64(7)

    head = CustomHead()
    assert torch_head_dimension(head, "out_features") == 7
    with pytest.raises(TypeError, match="in_features"):
        torch_head_dimension(head, "in_features")


def test_entropy_dispatches_to_torch() -> None:
    from probly.utils import entropy  # noqa: PLC0415
    from probly.utils.torch import torch_entropy  # noqa: PLC0415

    p = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    torch.testing.assert_close(entropy(p), torch_entropy(p))
    torch.testing.assert_close(entropy(p), torch.tensor([math.log(2.0), 0.0]))


def test_torch_entropy_gradient_is_finite_at_exact_zeros() -> None:
    p = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64, requires_grad=True)

    (gradient,) = torch.autograd.grad(torch_entropy(p), p)

    assert torch.isfinite(gradient).all()
    # dH/dp_k = -(1 + log p_k) wherever p_k > 0.
    torch.testing.assert_close(gradient[:2], torch.full((2,), -(1.0 + math.log(0.5)), dtype=torch.float64))


def test_intersection_probability_dispatches_to_torch() -> None:
    from probly.utils import intersection_probability  # noqa: PLC0415
    from probly.utils.torch import torch_intersection_probability  # noqa: PLC0415

    lower = torch.tensor([[0.2, 0.3]])
    upper = torch.tensor([[0.6, 0.7]])
    result = intersection_probability(lower, upper)
    torch.testing.assert_close(result, torch_intersection_probability(lower, upper))
    torch.testing.assert_close(result.sum(-1), torch.ones(1))
