"""This fixture module contains some test models for the tests."""

import pytest
import torch


@pytest.fixture
def model_small_2d_2d() -> torch.nn.Module:
    """This fixture returns a very small model with 2 input and 2 output neurons."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2),
    )
    return model
