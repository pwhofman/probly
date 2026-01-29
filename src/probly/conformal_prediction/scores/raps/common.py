"""Common functions for RAPS (Regularized Adaptive Prediction Sets) nonconformity scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.scores.common import ClassificationScore


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


class RAPSScore(ClassificationScore):
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
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.epsilon = epsilon
        self.randomize = randomize
        self.rng = np.random.default_rng(random_state)

        super().__init__(model=model, score_func=self._compute_score)

    def _compute_score(self, probs: Any) -> Any:  # noqa: ANN401
        """Calculate RAPS scores with optional randomization U-term."""
        probs_np = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
        scores: Any = raps_score_func(
            probs_np,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            epsilon=self.epsilon,
        )
        if self.randomize:
            u = self.rng.random(size=(scores.shape[0], 1))
            scores = scores - (u * probs_np)
        return np.asarray(scores)
