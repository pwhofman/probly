"""APS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike


@flexdispatch
def _aps_score_dispatch[T](probs: T, y_cal: T | None = None, randomized: bool = True) -> T:
    """Compute the APS nonconformity score."""
    msg = "APS score computation not implemented for this type."
    raise NotImplementedError(msg)


@_aps_score_dispatch.register(np.ndarray | ArrayLike)
def compute_aps_score_numpy(
    probs: np.ndarray | ArrayLike, y_cal: np.ndarray | ArrayLike | None = None, randomized: bool = True
) -> np.ndarray:
    """APS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)

    if probs_np.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_np.shape}."
        raise ValueError(msg)

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=-1)

    # get sorted probabilities
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=-1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=-1)

    # sort back to original positions without in-place writes
    inv_idx = np.argsort(srt_idx, axis=-1)

    if randomized:
        u = np.random.default_rng().uniform(low=0, high=1, size=probs_np.shape)
        cumsum_probs -= srt_probs * u

    scores = np.take_along_axis(cumsum_probs, inv_idx, axis=-1)
    if y_cal is not None:
        y_cal_np = np.asarray(y_cal, dtype=np.intp)
        if y_cal_np.shape != probs_np.shape[:-1]:
            msg = (
                "y_cal must match probs batch shape (all axes except the class axis); "
                f"got y_cal shape {y_cal_np.shape} and probs shape {probs_np.shape}."
            )
            raise ValueError(msg)
        scores = np.take_along_axis(scores, y_cal_np[..., np.newaxis], axis=-1)
        scores = np.squeeze(scores, axis=-1)
    return scores


@dataclass(frozen=True, slots=True)
class APSScore:
    """Configurable APS nonconformity-score callable."""

    randomized: bool = True

    def __call__(self, y_pred: Any, y_true: Any | None = None) -> Any:  # noqa: ANN401
        """Compute APS scores with constructor-provided configuration."""
        return _aps_score_dispatch(y_pred, y_true, randomized=self.randomized)


aps_score = APSScore()


__all__ = ["APSScore", "aps_score"]
