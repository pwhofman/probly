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
