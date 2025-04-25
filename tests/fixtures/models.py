"""Fixtures for models used in tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn


@pytest.fixture
def model_small_2d_2d() -> torch.nn.Module:
    """Return a small model with 2 input and 2 output neurons."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
    )
    return model


@pytest.fixture
def conv_linear_model() -> nn.Module:
    model = nn.Sequential(nn.Conv2d(3, 5, 5), nn.ReLU(), nn.Flatten(), nn.Linear(5, 2))
    return model


@pytest.fixture
def regression_model_1d() -> nn.Module:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    return model


@pytest.fixture
def regression_model_2d() -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    return model
