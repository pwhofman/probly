"""Conformal prediction implementation."""

import numpy as np
import torch
import torch.nn.functional as F


class ConformalPrediction:
    """Implementation of conformal prediction for a given model.

    Attributes:
        model: torch.nn.Module, the base model.
        alpha: float, the error rate for conformal prediction.
        q: float, the quantile value for conformal prediction.
    """

    def __init__(self, base: torch.nn.Module, alpha: float = 0.05) -> None:
        """Initialize an instance of the ConformalPrediction class.

        Args:
            base: torch.nn.Module, the base model to be used for conformal prediction.
            alpha: float, the error rate for conformal prediction.
        """
        self.model = base
        self.alpha = alpha
        self.q = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model without conformal prediction.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor, model output
        """
        return self.model(x)

    def predict_pointwise(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """Forward pass of the model without conformal prediction.

        Args:
            x: torch.Tensor, input data
            logits: bool, whether to return the logits.

        Returns:
            torch.Tensor, model output
        """
        if logits:
            return self.model(x)
        return F.softmax(self.model(x), dim=1)

    def predict_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Represent the uncertainty of the model by a conformal prediction set.

        Args:
            x: torch.Tensor, input data
        Returns:
            torch.Tensor of shape (n_instances, n_classes), the conformal prediction set,
            where each element is a boolean indicating whether the class is included in the set.
        """
        with torch.no_grad():
            outputs = self.model(x)
            scores = F.softmax(outputs, dim=1)
        sets = scores >= (1 - self.q)
        return sets

    def calibrate(self, loader: torch.utils.data.DataLoader) -> None:
        """Perform the calibration step for conformal prediction.

        Args:
            loader: DataLoader, The data loader for the calibration set.

        """
        self.model.eval()
        scores = []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                score = 1 - F.softmax(outputs, dim=1).numpy()
                score = score[torch.arange(score.shape[0]), targets]
                scores.append(score)
        scores = np.concatenate(scores)
        n = scores.shape[0]
        self.q = np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n, method="inverted_cdf")
