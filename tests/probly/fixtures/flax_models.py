"""Fixtures for models used in tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


@pytest.fixture
def flax_rngs() -> nnx.Rngs:
    """Return a random number generator for flax models."""
    return nnx.Rngs(0)


@pytest.fixture
def flax_model_small_2d_2d(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small linear model with 2 input and 2 output neurons."""
    model = nnx.Sequential(
        nnx.Linear(2, 2, rngs=flax_rngs),
        nnx.Linear(2, 2, rngs=flax_rngs),
        nnx.Linear(2, 2, rngs=flax_rngs),
    )
    return model


@pytest.fixture
def flax_conv_linear_model(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small convolutional model with 3 input channels and 2 output neurons."""
    model = nnx.Sequential(
        nnx.Conv(3, 5, (5, 5), rngs=flax_rngs),
        nnx.relu,
        nnx.flatten,
        nnx.Linear(5, 2, rngs=flax_rngs),
    )
    return model


@pytest.fixture
def flax_regression_model_1d(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small regression model with 2 input and 1 output neurons."""
    model = nnx.Sequential(
        nnx.Linear(2, 2, rngs=flax_rngs),
        nnx.relu,
        nnx.Linear(2, 1, rngs=flax_rngs),
    )
    return model


@pytest.fixture
def flax_regression_model_2d(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small regression model with 4 input and 2 output neurons."""
    model = nnx.Sequential(
        nnx.Linear(4, 4, rngs=flax_rngs),
        nnx.relu,
        nnx.Linear(4, 2, rngs=flax_rngs),
    )
    return model
