"""APS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lazy_dispatch.isinstance import LazyType

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs


@lazydispatch
def aps_score_func[T](probs: T) -> npt.NDArray[np.floating]:
    """APS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs)

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=1)

    # get sorted probabilities
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum = np.cumsum(srt_probs, axis=1)

    # sort back to original positions without in-place writes (JAX-safe)
    inv_idx = np.argsort(srt_idx, axis=1)
    scores = np.take_along_axis(cumsum, inv_idx, axis=1)

    return scores


def register(cls: LazyType, func: Callable) -> None:
    """Register a implementation for a specific type."""
    aps_score_func.register(cls=cls, func=func)


class APSScore:
    """Adaptive Prediction Sets (APS) nonconformity score."""

    def __init__(
        self,
        model: Predictor,
        randomize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize APS score with optional randomization."""
        self.model = model
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
        # get aps scores for all labels
        all_scores: npt.NDArray[np.floating] = aps_score_func(probs)

        # convert to numpy arrays
        scores_np = np.asarray(all_scores, dtype=float)  # (n, K)
        probs_np = np.asarray(probs, dtype=float)  # (n, K)
        labels_np = np.asarray(y_cal, dtype=int)  # (n,)

        # extract scores for true labels
        idx = np.arange(len(labels_np))
        nonconformity: npt.NDArray[np.floating] = scores_np[idx, labels_np]

        # randomization if enabled
        if self.randomize:
            u = self.rng.random(size=len(labels_np))
            true_probs = probs_np[idx, labels_np]
            nonconformity = nonconformity - (u * true_probs)

        return nonconformity

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: Any = None,  # noqa: ANN401
    ) -> npt.NDArray[np.floating]:
        """Compute scores for all labels."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        # get aps scores for all labels
        scores: npt.NDArray[np.floating] = aps_score_func(probs)
        scores_np = np.asarray(scores, dtype=float)

        # randomization if enabled
        if self.randomize:
            probs_np = np.asarray(probs, dtype=float)
            u = self.rng.random(size=(scores_np.shape[0], 1))
            scores_np = scores_np - (u * probs_np)

        return scores_np
