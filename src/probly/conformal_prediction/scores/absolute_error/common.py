"""Absolute Error Score implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from probly.conformal_prediction.methods.common import Predictor

from probly.conformal_prediction.scores.common import RegressionScore


class AbsoluteErrorScore(RegressionScore):
    """Standard absolute residual score: |y - y_hat|."""

    def __init__(self, model: Predictor) -> None:
        """Initialize with a prediction model."""
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute |y - y_hat|."""
        # get predictions
        y_hat = self.model(x_cal)

        # convert to numpy if needed
        y_hat_np = y_hat.detach().cpu().numpy() if hasattr(y_hat, "detach") else np.asarray(y_hat)

        # ensure numpy arrays and correct shapes
        y_hat_np = np.asarray(y_hat_np, dtype=float)
        y_cal_np = np.asarray(y_cal, dtype=float)

        if y_hat_np.ndim > 1:
            y_hat_np = y_hat_np.flatten()

        # compute absolute difference
        return cast(
            "npt.NDArray[np.floating]",
            np.abs(y_cal_np - y_hat_np),
        )

    def predict_nonconformity(
        self,
        _x_test: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Dummy implementation for protocol compliance."""
        # conform to protocol, but not used in regression intervals
        return np.array([], dtype=float)

    def construct_intervals(
        self,
        y_hat: npt.NDArray[np.floating],
        threshold: float,
    ) -> npt.NDArray[np.floating]:
        """Construct symmetric intervals [y - q, y + q]."""
        # ensure y_hat is 1D
        current_y_hat = y_hat

        if current_y_hat.ndim > 1:
            current_y_hat = current_y_hat.flatten()

        lower = current_y_hat - threshold
        upper = current_y_hat + threshold

        return np.stack([lower, upper], axis=1)
