"""APS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor


import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.scores.common import ClassificationScore


@lazydispatch
def aps_score_func[T](probs: T) -> npt.NDArray[np.floating]:
    """APS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs)

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=1)

    # get sorted probabilities
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=1)

    # sort back to original positions without in-place writes (JAX-safe)
    inv_idx = np.argsort(srt_idx, axis=1)
    scores = np.take_along_axis(cumsum_probs, inv_idx, axis=1)
    return scores


def register(cls: LazyType, func: Callable) -> None:
    """Register a implementation for a specific type."""
    aps_score_func.register(cls=cls, func=func)


class APSScore(ClassificationScore):
    """Adaptive Prediction Sets (APS) nonconformity score."""

    def __init__(
        self,
        model: Predictor,
        randomize: bool = True,
    ) -> None:
        """Initialize APS score with optional randomization."""
        self.randomize = randomize
        self.rng = np.random.default_rng()

        def compute_score(probs: Any) -> Any:  # noqa: ANN401
            """Calculate APS scores with randomization U-term."""
            # randomization if enabled
            scores: Any = aps_score_func(probs)

            # extract probabilities of true labels
            if self.randomize:
                u = self.rng.random(size=(scores.shape[0], 1))
                # get the probabilities corresponding to the true labels
                scores = scores - (u * probs)

            return scores

        super().__init__(model=model, score_func=compute_score)
