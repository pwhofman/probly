from __future__ import annotations

from flextype import flexdispatch
import numpy as np

from probly.representation.distribution import NumpyCategoricalDistribution
from probly.representation.sample.numpy import NumpySample


@flexdispatch
def lac_score[T](probs: T, y_cal: T | None = None) -> T:
    """Compute the LAC nonconformity score."""
    msg = "LAC score computation not implemented for this type."
    raise NotImplementedError(msg)


@lac_score.register(np.ndarray)
def numpy_compute_lac_score(probs: np.ndarray, y_cal: np.ndarray | None = None) -> np.ndarray:
    probs_np = np.asarray(probs, dtype=float)
    if probs_np.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_np.shape}."
        raise ValueError(msg)

    scores = 1.0 - probs_np
    if y_cal is not None:
        labels = np.asarray(y_cal, dtype=np.intp)
        if labels.shape != probs_np.shape[:-1]:
            msg = (
                "y_cal must match probs batch shape (all axes except the class axis); "
                f"got y_cal shape {labels.shape} and probs shape {probs_np.shape}."
            )
            raise ValueError(msg)
        scores = np.take_along_axis(scores, labels[..., np.newaxis], axis=-1)
        scores = np.squeeze(scores, axis=-1)
    return scores


@lac_score.register(NumpyCategoricalDistribution)
def numpy_compute_lac_score_categorical(
    probs: NumpyCategoricalDistribution, y_cal: np.ndarray | None = None
) -> np.ndarray:
    return numpy_compute_lac_score(probs.probabilities, y_cal)


@lac_score.register(NumpySample)
def numpy_compute_lac_score_sample(probs: NumpySample, y_cal: np.ndarray | None = None) -> np.ndarray:
    return lac_score(probs.array, y_cal)
