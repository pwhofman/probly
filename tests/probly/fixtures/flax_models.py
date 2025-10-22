"""Fixtures for models used in tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
from flax.typing import Array  # noqa: E402


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


@pytest.fixture
def flax_custom_model(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small custom model."""

    class TinyModel(nnx.Module):
        """A simple neural network model with two linear layers and activation functions.

        Attributes:
            linear1 : The first linear layer with input size 100 and output size 200.
            activation : The ReLU activation function applied after the first linear layer.
            linear2 : The second linear layer with input size 200 and output size 10.
            softmax : The softmax function for normalizing the output into probabilities.
        """

        def __init__(self, rngs: nnx.Rngs) -> None:
            """Initialize the TinyModel class."""
            super().__init__()

            self.linear1 = nnx.Linear(10, 20, rngs=rngs)
            self.activation = nnx.relu
            self.linear2 = nnx.Linear(20, 4, rngs=rngs)
            self.softmax = nnx.softmax

        def __call__(self, x: Array) -> Array:
            """Forward pass of the TinyModel model.

            Parameters:
                x: Input tensor to be processed by the forward method.

            Returns:
                Output tensor after being processed through the layers and activation functions.
            """
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.softmax(x)
            return x

    return TinyModel(rngs=flax_rngs)
