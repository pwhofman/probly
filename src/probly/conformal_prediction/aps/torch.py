"""Torch implementation of APS (Adaptive Prediction Sets)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from probly.conformal_prediction.aps.common import calculate_nonconformity_score, calculate_quantile
from probly.conformal_prediction.common import ConformalPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence


class APSPredictor(ConformalPredictor):
    """APS (Adaptive Prediction Sets) conformal predictor using PyTorch."""

    def __init__(
        self,
        model: nn.Module,
        device: str | None = None,
    ) -> None:
        """Initialize the APS predictor for PyTorch.

        Args:
            model: A trained PyTorch model that outputs class probabilities.
            device: Device to run the model on. If None, uses 'cuda' if available, else 'cpu'.
        """
        super().__init__(model=model)
        self.model: nn.Module = model

        # set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # move model to device
        self.model.to(self.device)
        self.model.eval()

    def _get_probs(
        self,
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Get predicted probabilities from the model for the given input data."""
        # convert x to torch tensor if it's a numpy array
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

            # handles tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # convert outputs to probabilities using softmax
            probs = torch.softmax(outputs, dim=1) if outputs.shape[1] > 1 else torch.sigmoid(outputs)

            return probs.cpu().numpy()

    def _compute_nonconformity(
        self,
        x: Sequence[Any],
        y: Sequence[Any],
    ) -> np.ndarray:
        """Compute nonconformity scores for the given data.

        Args:
            x: Input features.
            y: True labels.

        Returns:
            Nonconformity scores as a numpy array.
        """
        # get predicted probabilities
        probs = self._get_probs(x)

        # convert y to numpy array
        y_array = np.asarray(y)

        return calculate_nonconformity_score(probs, y_array)

    def predict(
        self,
        x: Sequence[Any],
        significance: float,
    ) -> list[list[int]]:
        """Generate prediction sets for the given input data.

        Args:
            x: Input features.
            significance: Significance level for the prediction sets.

        Returns:
            A list of prediction sets for each input instance.
        """
        _ = significance

        if not self.is_calibrated or self.threshold is None:
            error_msg = "Predictor must be calibrated before making predictions."
            raise RuntimeError(error_msg)

        probs = self._get_probs(x)
        n_samples = probs.shape[0]
        prediction_sets = []

        for i in range(n_samples):
            probs_i = probs[i]
            sorted_indices = np.argsort(-probs_i)
            sorted_probs = probs_i[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            include_idx = np.where(cumulative_probs <= self.threshold)[0]

            if len(include_idx) == 0:
                prediction_set = [int(sorted_indices[0])]
            else:
                prediction_set = [int(sorted_indices[idx]) for idx in include_idx]

            prediction_sets.append(prediction_set)

        return prediction_sets

    def calibrate(
        self,
        x_calib: Sequence[Any],
        y_calib: Sequence[Any],
        significance: float,
    ) -> float:
        """Calibrate the predictor using calibration data.

        Args:
            x_calib: Calibration input features.
            y_calib: Calibration true labels.
            significance: Significance level for calibration.
        """
        self.nonconformity_scores = self._compute_nonconformity(x_calib, y_calib)
        self.threshold = calculate_quantile(self.nonconformity_scores, significance)
        self.is_calibrated = True

        return self.threshold

    def __str__(self) -> str:
        """String representation of the predictor."""
        model_name = self.model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"APSPredictor(model={model_name}, status={status})"
