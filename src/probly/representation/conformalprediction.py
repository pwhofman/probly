"""Conformal prediction implementation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from probly.representation.predictor import (
    Classifier,
    RepresentationPredictor,
)


class ConformalPrediction[In, KwIn](
    Classifier[In, KwIn, torch.Tensor],
    RepresentationPredictor[In, KwIn, torch.Tensor, torch.Tensor],
):
    """Implementation of conformal prediction for a given model.

    Attributes:
        model: Classifier[In, KwIn, Out], the base model.
        alpha: float, the error rate for conformal prediction.
        q: float, the quantile value for conformal prediction.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        alpha: float = 0.05,
    ) -> None:
        """Initialize an instance of the ConformalPrediction class.

        Args:
            base: Predictor[In, KwIn, torch.Tensor], the base model to be used for conformal prediction.
            alpha: float, the error rate for conformal prediction.
        """
        self.model = base
        self.alpha = alpha

    def __call__(self, *args: In, logits: bool = False, **kwargs: KwIn) -> torch.Tensor:
        """Forward pass of the model without conformal prediction."""
        res: torch.Tensor = self.model(*args, **kwargs)
        if logits:
            return res
        return F.softmax(res, dim=1)

    def predict_pointwise(
        self,
        *args: In,
        logits: bool = False,
        **kwargs: KwIn,
    ) -> torch.Tensor:
        """Forward pass of the model without conformal prediction."""
        return self.__call__(*args, logits=logits, **kwargs)

    def predict_representation(
        self,
        *args: In,
        **kwargs: KwIn,
    ) -> torch.Tensor:
        """Represent the uncertainty of the model by a conformal prediction set.

        Args:
            *args: In, the input data to the model.

            logits: bool, whether to return logits instead of probabilities.
            **kwargs: KwIn, additional keyword arguments for the model.

        Returns:
            torch.Tensor of shape (n_instances, n_classes), the conformal prediction set,
            where each element is a boolean indicating whether the class is included in the set.
        """
        with torch.no_grad():
            scores = self.__call__(*args, logits=False, **kwargs)
        sets: torch.Tensor = scores >= (1 - self.q)
        return sets

    def calibrate(self, loader: torch.utils.data.DataLoader) -> None:
        """Perform the calibration step for conformal prediction.

        Args:
            loader: DataLoader, The data loader for the calibration set.

        """
        self.model.eval()
        scores_ = []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs)
                score = 1 - F.softmax(outputs, dim=1)
                score = score[torch.arange(score.shape[0]), targets]
                scores_.append(score)
        scores = torch.concatenate(scores_).numpy()
        n = scores.shape[0]
        self.q = np.quantile(
            scores,
            np.ceil((n + 1) * (1 - self.alpha)) / n,
            method="inverted_cdf",
        )
