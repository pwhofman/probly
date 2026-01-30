"""Common for SAPS scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.scores.common import ClassificationScore


@lazydispatch
def saps_score_func[T](
    probs: T,
    lambda_val: float,
    u: T,
) -> npt.NDArray[np.floating]:
    """Compute SAPS Nonconformity Score for specific label (Reference: Eq 10).

    prob: 1D array with probabilities.
    label: true index
    lambda_val: lambda value for SAPS.
    u: optional random value in [0,1).
    """
    # convert to numpy arrays
    probs_np = np.asarray(probs, dtype=float)
    u_np = np.asarray(u, dtype=float)

    # get max probabilities for each sample
    max_probs = np.max(probs_np, axis=1, keepdims=True)

    # get ranks for each label, argsort along axis=1 in descending order
    sort_idx = np.argsort(-probs_np, axis=1)

    # find the rank (1-based) of each label
    # compare each position in sorted_indices with the corresponding label
    ranks_zero_based = np.argsort(sort_idx, axis=1)
    ranks = ranks_zero_based + 1  # +1 for 1-based rank

    scores = np.where(ranks == 1, u_np * max_probs, max_probs + (ranks - 2 + u_np) * lambda_val)

    return np.asarray(scores, dtype=float)


def register(cls: LazyType, func: Callable) -> None:
    """Register an implementation for a specific type."""
    saps_score_func.register(cls=cls, func=func)


class SAPSScore(ClassificationScore):
    """Sorted Adaptive Prediction Sets (SAPS) nonconformity score."""

    rng: Generator

    def __init__(
        self,
        model: Predictor,
        lambda_val: float = 0.1,
        random_state: int | None = 42,
    ) -> None:
        """Initialize SAPS score with reproducible random_state by default."""
        self.lambda_val = lambda_val
        self.rng = np.random.default_rng(random_state)
        super().__init__(model=model, score_func=self.compute_score, randomize=False)

    def compute_score(self, probs: Any) -> Any:  # noqa: ANN401
        """Calculate SAPS scores with randomization U-term."""
        n_samples = probs.shape[0]
        n_classes = probs.shape[1]

        # randomization: generate u values for each sample and class in [0,1)
        u = self.rng.random(size=(n_samples, n_classes))

        # get the scores from the SAPS score function
        return saps_score_func(
            probs,
            lambda_val=self.lambda_val,
            u=u,
        )
