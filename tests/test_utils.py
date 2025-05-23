"""Tests for the utils module."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from probly.utils import (
    differential_entropy_gaussian,
    intersection_probability,
    kl_divergence_gaussian,
    powerset,
    temperature_softmax,
    torch_collect_outputs,
    torch_reset_all_parameters,
)


def test_powerset() -> None:
    assert powerset([]) == [()]
    assert powerset([1]) == [(), (1,)]
    assert powerset([1, 2]) == [(), (1,), (2,), (1, 2)]


def test_capacity() -> None:
    pass


def test_moebius() -> None:
    pass


def test_differential_entropy_gaussian() -> None:
    assert np.isclose(differential_entropy_gaussian(0.5), 1.54709559)
    assert np.allclose(differential_entropy_gaussian(np.array([1, 2]), base=np.e), np.array([1.41893853, 1.76551212]))


def test_kl_divergence_gaussian() -> None:
    mu1 = np.array([0.0, 1.0])
    mu2 = np.array([1.0, 0.0])
    sigma21 = np.array([0.1, 0.1])
    sigma22 = np.array([0.1, 0.1])
    assert np.isclose(kl_divergence_gaussian(1.0, 1.0, 1.0, 1.0), 0.0)
    assert np.isclose(kl_divergence_gaussian(1.0, 1.0, 1.0, 1.0, base=np.e), 0.0)
    assert np.allclose(kl_divergence_gaussian(mu1, sigma21, mu2, sigma22, base=np.e), np.array([5.0, 5.0]))


def test_torch_reset_all_parameters(conv_linear_model: torch.nn.Module) -> None:
    def flatten_params(model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([param.flatten() for param in model.parameters()])

    before = flatten_params(conv_linear_model)
    torch_reset_all_parameters(conv_linear_model)
    after = flatten_params(conv_linear_model)
    assert not torch.equal(before, after)


def test_torch_collect_outputs(conv_linear_model: torch.nn.Module) -> None:
    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 5, 5),
            torch.randn(
                2,
            ),
        )
    )
    outputs, targets = torch_collect_outputs(conv_linear_model, loader, torch.device("cpu"))
    assert outputs.shape == (2, 2)
    assert targets.shape == (2,)


def test_temperature_softmax() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(temperature_softmax(x, 2.0), torch.softmax(x / 2.0, dim=1))
    assert torch.equal(temperature_softmax(x, torch.tensor(1.0)), torch.softmax(x, dim=1))


def test_intersection_probability() -> None:
    rng = np.random.default_rng()

    probs = rng.dirichlet(np.ones(2), size=(5, 5))
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (5, 2)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)

    probs = rng.dirichlet(np.ones(10), size=(5, 5))
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (5, 10)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)

    probs = np.array([1 / 3, 1 / 3, 1 / 3] * 5).reshape(1, 5, 3)
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (1, 3)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)
