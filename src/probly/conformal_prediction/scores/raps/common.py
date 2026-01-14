"""Common functions for RAPS (Regularized Adaptive Prediction Sets) nonconformity scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lazy_dispatch.isinstance import LazyType

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs


@lazydispatch
def raps_score_func[T](
    probs: T,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> npt.NDArray[np.floating]:
    """RAPS nonconformity scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)
    n_samples, n_classes = probs_np.shape

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=1)
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=1)

    # apply regularization: penalize large sets
    # add λ * max(0, rank - k_reg - 1)
    ranks = np.arange(1, n_classes + 1).reshape(1, -1)
    penalty = lambda_reg * np.maximum(0, ranks - k_reg - 1)

    # add epsilon for stability
    epsilon_penalty = epsilon * np.ones((n_samples, n_classes))

    # combine all components
    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original order
    inv_idx = np.argsort(srt_idx, axis=1)
    scores = np.take_along_axis(reg_cumsum, inv_idx, axis=1)

    return scores


def register(cls: LazyType, func: Callable) -> None:
    """Register implementation for specific type."""
    raps_score_func.register(cls=cls, func=func)


class RAPSScore:
    """Regularized Adaptive Prediction Sets (RAPS) nonconformity score."""

    def __init__(
        self,
        model: Predictor,
        lambda_reg: float = 0.1,  # regularization strength
        k_reg: int = 0,  # number of classes without penalty
        epsilon: float = 0.01,  # small constant
        randomize: bool = True,  # optional randomization
        random_state: int | None = None,
    ) -> None:
        """Initialize RAPS score with regularization parameters and optional randomization."""
        self.model = model
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.epsilon = epsilon
        self.randomize = randomize
        self.rng = np.random.default_rng(random_state)

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute calibration scores."""
        # get probabilities from model
        probs: npt.NDArray[np.floating] = predict_probs(self.model, x_cal)
        # get raps scores for all labels
        all_scores: npt.NDArray[np.floating] = raps_score_func(probs, self.lambda_reg, self.k_reg, self.epsilon)

        # convert to numpy arrays
        scores_np = np.asarray(all_scores, dtype=float)
        probs_np = np.asarray(probs, dtype=float)
        labels_np = np.asarray(y_cal, dtype=int)

        # extract scores for true labels
        idx = np.arange(len(labels_np))
        nonconformity: npt.NDArray[np.floating] = scores_np[idx, labels_np]

        # optional randomization (like APS)
        if self.randomize:
            u = self.rng.random(size=len(labels_np))
            true_probs = probs_np[idx, labels_np]
            nonconformity = nonconformity - (u * true_probs)

        return nonconformity

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute RAPS scores for all labels."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        # get raps scores for all labels
        all_scores: npt.NDArray[np.floating] = raps_score_func(probs, self.lambda_reg, self.k_reg, self.epsilon)
        scores_np = np.asarray(all_scores, dtype=float)

        # optional randomization
        if self.randomize:
            probs_np = np.asarray(probs, dtype=float)
            u = self.rng.random(size=(scores_np.shape[0], 1))
            scores_np = scores_np - (u * probs_np)

        return scores_np
