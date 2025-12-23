"""Basic Template for Calibration Methods with Torch and Flax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from flax import linen
    from jax import Array
    from torch import Tensor, device as TorchDevice, nn


class CalibratorBaseTorch(ABC):
    """Abstract base class for calibrators with torch."""

    def __init__(self, base_model: nn.Module, device: TorchDevice) -> None:
        """Create a calibrator.

        Args:
            base_model: The base model whose outputs are to be calibrated.
            device: Torch device or a device string (e.g. 'cpu', 'cuda:0').
        """
        self.model = base_model
        self.device = device

    @abstractmethod
    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> Self:
        """Fit calibrator from calibration_set (DataLoader-like)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Return calibrated probabilities for input x."""
        raise NotImplementedError


"""WILL BE IMPLEMENTED LATER"""


class CalibratorBaseFlax(ABC):
    """Abstract base class for calibrators with Flax."""

    def __init__(self, base_model: linen.Module, params: dict) -> None:
        """Create a calibrator.

        Args:
            base_model: The base Flax model whose outputs are to be calibrated.
            params: The model parameters (Flax uses explicit parameter passing).
        """
        self.model = base_model
        self.params = params

    @abstractmethod
    def fit(self, calibration_set: Array, truth_labels: Array) -> Self:
        """Fit calibrator from calibration_set.

        Args:
            calibration_set: Input data for calibration.
            truth_labels: Ground truth labels.

        Returns:
            Self with fitted calibration parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Array) -> Array:
        """Return calibrated probabilities for input x.

        Args:
            x: Input array.

        Returns:
            Calibrated probabilities.
        """
        raise NotImplementedError
