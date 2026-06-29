"""SAPS Score implementation with optional Randomization (U-term)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike


@flexdispatch
def _saps_score_dispatch[T](
    probs: T,
    y_cal: T | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> T:
    """Compute the SAPS nonconformity score."""
    msg = "SAPS score computation not implemented for this type."
    raise NotImplementedError(msg)


@_saps_score_dispatch.register(np.ndarray | ArrayLike)
def compute_saps_score_func_numpy(
    probs: np.ndarray | ArrayLike,
    y_cal: np.ndarray | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> np.ndarray:
    """SAPS Nonconformity-Scores for numpy arrays."""
    probs_np = np.asarray(probs, dtype=float)
    if probs_np.ndim < 1:
        msg = f"probs must have at least one dimension with classes on the last axis, got shape {probs_np.shape}."
        raise ValueError(msg)

    u = np.random.default_rng().uniform(size=probs_np.shape) if randomized else np.zeros_like(probs_np)

    max_probs = np.max(probs_np, axis=-1, keepdims=True)
    sort_idx = np.argsort(-probs_np, axis=-1)
    ranks_zero_based = np.argsort(sort_idx, axis=-1)
    ranks = ranks_zero_based + 1

    scores = np.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

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
    return np.asarray(scores, dtype=float)


@dataclass(frozen=True, slots=True)
class SAPSScore:
    """Configurable SAPS nonconformity-score callable."""

    randomized: bool = True
    lambda_val: float = 0.1

    def __call__(self, y_pred: Any, y_true: Any | None = None) -> Any:  # noqa: ANN401
        """Compute SAPS scores with constructor-provided configuration."""
        return _saps_score_dispatch(
            y_pred,
            y_true,
            randomized=self.randomized,
            lambda_val=self.lambda_val,
        )


saps_score = SAPSScore()


__all__ = ["SAPSScore", "saps_score"]
