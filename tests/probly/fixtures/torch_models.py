"""Fixtures for models used in tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


@pytest.fixture
def torch_model_small_2d_2d() -> nn.Module:
    """Return a small linear model with 2 input and 2 output neurons."""
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
    )
    return model


@pytest.fixture
def torch_conv_linear_model() -> nn.Module:
    """Return a small convolutional model with 3 input channels and 2 output neurons."""
    model = nn.Sequential(
        nn.Conv2d(3, 5, 5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(5, 2),
    )
    return model


@pytest.fixture
def torch_regression_model_1d() -> nn.Module:
    """Return a small regression model with 2 input and 1 output neurons."""
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    )
    return model


@pytest.fixture
def torch_regression_model_2d() -> nn.Module:
    """Return a small regression model with 4 input and 2 output neurons."""
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    return model
