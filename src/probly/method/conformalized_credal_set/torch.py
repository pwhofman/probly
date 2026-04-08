"""Torch-specific conformalized credal set predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from probly.conformal_prediction.methods.common import predict_probs
from probly.conformal_prediction.utils.quantile import calculate_quantile
from probly.representation.credal_set.torch import TorchProbabilityIntervalsCredalSet

from ._common import register

if TYPE_CHECKING:
    from collections.abc import Sequence


class TorchConformalizedCredalSetPredictor:
    """Conformalized credal set predictor for PyTorch models.

    Wraps a :class:`torch.nn.Module` classifier and exposes a two-step interface:

    1. :meth:`calibrate` – compute and store the conformal threshold from a
       held-out calibration set.
    2. :meth:`predict` – return a
       :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`
       for every test batch.

    The nonconformity score used during calibration is the LAC score
    :math:`s_i = 1 - \\hat{p}(y_i \\mid x_i)`.

    Attributes:
        model: The underlying PyTorch classifier.
        threshold: The calibrated conformal threshold (``None`` before calibration).
        is_calibrated: Whether :meth:`calibrate` has been called successfully.

    """

    def __init__(self, model: nn.Module) -> None:
        """Create a conformalized credal set predictor.

        Args:
            model: A PyTorch classification model that outputs logits or
                softmax probabilities of shape ``(n, K)``.

        """
        self.model = model
        self.threshold: float | None = None
        self.is_calibrated: bool = False

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate the predictor on a held-out calibration set.

        Computes LAC nonconformity scores on the calibration data and stores
        the :math:`\\lceil(n+1)(1-\\alpha)\\rceil/n`-quantile as the threshold.

        Args:
            x_cal: Calibration inputs (anything accepted by the wrapped model).
            y_cal: True class labels for the calibration inputs.
            alpha: Significance level in ``(0, 1)``.  Coverage guarantee is
                ``1 - alpha``.

        Returns:
            The calibrated threshold :math:`\\hat{q}`.

        """
        probs: torch.Tensor = predict_probs(self.model, x_cal)  # ty: ignore[invalid-assignment]

        device = probs.device
        labels = (
            y_cal.to(device).long()
            if isinstance(y_cal, torch.Tensor)
            else torch.as_tensor(y_cal, dtype=torch.long, device=device)
        )

        # LAC nonconformity score: 1 - p(true_label | x)
        true_probs = probs[torch.arange(len(labels), device=device), labels]
        scores: np.ndarray = (1.0 - true_probs).detach().cpu().numpy()

        self.threshold = calculate_quantile(scores, alpha)
        self.is_calibrated = True
        return self.threshold

    def predict(self, x_test: Sequence[Any]) -> TorchProbabilityIntervalsCredalSet:
        """Predict a credal set for each test instance.

        For each test point :math:`x` with predicted softmax probabilities
        :math:`\\hat{p}`, the credal set is

        .. math::

            C(x) = \\{p : \\max(0, \\hat{p}_k - \\hat{q}) \\le p_k \\le
                     \\min(1, \\hat{p}_k + \\hat{q})\\}

        where :math:`\\hat{q}` is the calibrated threshold.

        Args:
            x_test: Test inputs.

        Returns:
            A :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`
            of shape ``(n,)`` with ``K`` classes per instance.

        Raises:
            RuntimeError: If the predictor has not been calibrated yet.

        """
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        probs: torch.Tensor = predict_probs(self.model, x_test)  # ty: ignore[invalid-assignment]

        lower = torch.clamp(probs - self.threshold, min=0.0)
        upper = torch.clamp(probs + self.threshold, max=1.0)

        return TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)

    def __call__(self, x_test: Sequence[Any]) -> TorchProbabilityIntervalsCredalSet:
        """Alias for :meth:`predict`.

        Args:
            x_test: Test inputs.

        Returns:
            A :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`.

        """
        return self.predict(x_test)

    def __str__(self) -> str:
        """Return a human-readable description of this predictor."""
        model_name = self.model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        threshold_str = f", threshold={self.threshold:.6f}" if self.threshold is not None else ""
        return f"TorchConformalizedCredalSetPredictor(model={model_name}, {status}{threshold_str})"


def _create_torch_predictor(model: nn.Module) -> TorchConformalizedCredalSetPredictor:
    return TorchConformalizedCredalSetPredictor(model=model)


register(nn.Module, _create_torch_predictor)
