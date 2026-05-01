"""Tests for het_nets multi-sample training dispatch."""

from __future__ import annotations

from unittest import mock

import pytest

from probly.method.het_nets import HetNetsPredictor, het_nets
from probly_benchmark.train_funcs import train_epoch

torch = pytest.importorskip("torch")

from torch import nn, optim  # noqa: E402


def _make_predictor() -> HetNetsPredictor:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    return het_nets(model, num_factors=2, predictor_type="logit_classifier")


def test_train_epoch_het_net_with_one_sample_returns_valid_loss() -> None:
    predictor = _make_predictor()
    optimizer = optim.SGD(predictor.parameters(), lr=0.01)
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))

    loss = train_epoch(predictor, inputs, targets, optimizer, samples=1)

    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))


def test_train_epoch_het_net_calls_model_once_per_sample() -> None:
    predictor = _make_predictor()
    optimizer = optim.SGD(predictor.parameters(), lr=0.01)
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))

    n_samples = 5
    call_count = 0
    original_forward = predictor.forward

    def counting_forward(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original_forward(x)

    with mock.patch.object(predictor, "forward", counting_forward):
        train_epoch(predictor, inputs, targets, optimizer, samples=n_samples)

    assert call_count == n_samples


def test_train_epoch_het_net_with_multiple_samples_returns_valid_loss() -> None:
    predictor = _make_predictor()
    optimizer = optim.SGD(predictor.parameters(), lr=0.01)
    inputs = torch.randn(4, 4)
    targets = torch.randint(0, 3, (4,))

    loss = train_epoch(predictor, inputs, targets, optimizer, samples=10)

    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
