"""Fixtures for models used in tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor, nn  # noqa: E402


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


@pytest.fixture
def torch_dropout_model() -> nn.Module:
    """Return a small dropout model with 2 input and 2 output neurons."""
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2, 2),
    )
    return model


@pytest.fixture
def torch_custom_model() -> nn.Module:
    """Return a small custom model."""

    class TinyModel(nn.Module):
        """A simple neural network model with two linear layers and activation functions.

        Attributes:
            linear1 : The first linear layer with input size 100 and output size 200.
            activation : The ReLU activation function applied after the first linear layer.
            linear2 : The second linear layer with input size 200 and output size 10.
            softmax : The softmax function for normalizing the output into probabilities.
        """

        def __init__(self) -> None:
            """Initialize the TinyModel class."""
            super().__init__()

            self.linear1 = nn.Linear(10, 20)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(20, 4)
            self.softmax = nn.Softmax()

        def forward(self, x: Tensor) -> Tensor:
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

    return TinyModel()
