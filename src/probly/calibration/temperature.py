"""Temperature scaling implementation."""

from __future__ import annotations

import torch
from torch import nn, optim

from probly.losses import ExpectedCalibrationError
from probly.utils import temperature_softmax, torch_collect_outputs


class Temperature(nn.Module):
    """Implementation of temperature scaling :cite:`guoOnCalibration2017`.

    Attributes:
        model: The model to be calibrated.
        temperature: The temperature parameter to be optimized.
    """

    def __init__(self, base: nn.Module) -> None:
        """Initialize an instance of the Temperature class.

        Args:
            base: The base model to be calibrated.
        """
        super().__init__()
        self.model = base
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input data.

        Returns:
            Model output.
        """
        return self.model(x)

    def predict_pointwise(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model for point-wise prediction.

        Args:
            x: Input data.

        Returns:
            Model output.
        """
        x = self.model(x)
        return temperature_softmax(x, self.temperature)

    def fit(self, loader: torch.utils.data.DataLoader, learning_rate: float = 0.01, max_iter: int = 100) -> None:
        """Fit the temperature scaling model using the data.

        This method optimizes the temperature parameter based on the Expected Calibration Error (ECE) loss.

        Args:
            loader: Data loader to use for optimizing.
            learning_rate: Learning rate for the optimizer.
            max_iter: Maximum number of iterations for the optimizer.
        """
        inputs, targets = torch_collect_outputs(self.model, loader, self.temperature.device)
        criterion = ExpectedCalibrationError()
        optimizer = optim.LBFGS([self.temperature], lr=learning_rate, max_iter=max_iter)

        def closure() -> torch.Tensor:
            """Closure function for the optimizer.

            This function computes the loss and gradients.

            Returns:
                Loss value.
            """
            optimizer.zero_grad()
            outputs = temperature_softmax(inputs, self.temperature)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
