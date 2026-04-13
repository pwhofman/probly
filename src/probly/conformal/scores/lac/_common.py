from __future__ import annotations

import numpy as np

from lazy_dispatch import lazydispatch
from probly.conformal.scores._common import ClassificationNonConformityScore
from probly.representation.distribution import ArrayCategoricalDistribution


@lazydispatch
def lac_score_func[T](probs: T, y_cal: T | None = None) -> T:
    msg = "LAC score computation not implemented for this type."
    raise NotImplementedError(msg)


@lac_score_func.register(np.ndarray)
def compute_lac_score_numpy(probs: np.ndarray, y_cal: np.ndarray | None = None) -> np.ndarray:
    scores = 1.0 - probs
    if y_cal is not None:
        scores = scores[np.arange(len(probs)), y_cal]
    return scores


@lac_score_func.register(ArrayCategoricalDistribution)
def compute_lac_score_categorical(probs: ArrayCategoricalDistribution, y_cal: np.ndarray | None = None) -> np.ndarray:
    return lac_score_func(probs.probabilities, y_cal)


class LACScore[T](ClassificationNonConformityScore[T]):
    non_conformity_score = lac_score_func
