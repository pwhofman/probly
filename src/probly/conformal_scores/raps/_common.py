"""RAPS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.representation.distribution import ArrayCategoricalDistribution
from probly.representation.sample.array import ArraySample


@flexdispatch
def _raps_score_dispatch[T](
    probs: T,
    y_cal: T | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> T:
    """Compute the RAPS nonconformity score."""
    msg = "RAPS score computation not implemented for this type."
    raise NotImplementedError(msg)


@_raps_score_dispatch.register(np.ndarray)
def compute_raps_score_numpy(
    probs: np.ndarray,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)
    if probs_np.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_np.shape}."
        raise ValueError(msg)

    n_classes = probs_np.shape[-1]

    # sorting indices for descending probabilities
    srt_idx = np.argsort(-probs_np, axis=-1)
    srt_probs = np.take_along_axis(probs_np, srt_idx, axis=-1)

    # calculate cumulative sums
    cumsum_probs = np.cumsum(srt_probs, axis=-1)

    if randomized:
        u = np.random.default_rng().uniform(low=0, high=1, size=probs_np.shape)
        cumsum_probs -= srt_probs * u

    # apply regularization: penalize large sets
    ranks = np.arange(1, n_classes + 1, dtype=probs_np.dtype).reshape((1,) * (probs_np.ndim - 1) + (n_classes,))
    penalty = lambda_reg * np.maximum(0, ranks - k_reg - 1)
    epsilon_penalty = epsilon * np.ones_like(probs_np)

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original order
    inv_idx = np.argsort(srt_idx, axis=-1)
    scores = np.take_along_axis(reg_cumsum, inv_idx, axis=-1)

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


@_raps_score_dispatch.register(ArrayCategoricalDistribution)
def _(
    probs: ArrayCategoricalDistribution,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for ArrayCategoricalDistributions."""
    return compute_raps_score_numpy(
        probs.probabilities,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )


@_raps_score_dispatch.register(ArraySample)
def _(
    probs: ArraySample,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> np.ndarray:
    """RAPS Nonconformity-Scores for ArraySamples."""
    return _raps_score_dispatch(
        probs.array,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )


@dataclass(frozen=True, slots=True)
class RAPSScore:
    """Configurable RAPS nonconformity-score callable."""

    randomized: bool = True
    lambda_reg: float = 0.1
    k_reg: int = 0
    epsilon: float = 0.01

    def __call__(self, y_pred: Any, y_true: Any | None = None) -> Any:  # noqa: ANN401
        """Compute RAPS scores with constructor-provided configuration."""
        return _raps_score_dispatch(
            y_pred,
            y_true,
            randomized=self.randomized,
            lambda_reg=self.lambda_reg,
            k_reg=self.k_reg,
            epsilon=self.epsilon,
        )


raps_score = RAPSScore()


__all__ = ["RAPSScore", "raps_score"]
