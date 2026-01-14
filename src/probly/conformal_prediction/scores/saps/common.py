"""Common for SAPS scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs


@lazydispatch
def saps_score_func(
    probs: npt.NDArray[np.floating],
    label: int,
    lambda_val: float = 0.1,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for specific label (Reference: Eq 10).

    prob: 1D array with probabilities.
    label: true index
    lambda_val: lambda value for SAPS.
    u: optional random value in [0,1).
    """
    probs_np = np.asarray(probs, dtype=float)

    if probs_np.ndim == 2:
        if probs_np.shape[0] != 1:
            raise ValueError
        probs_np = probs_np[0]

    if not (0 <= label < probs_np.shape[0]):
        raise ValueError

    if u is None:
        u = float(np.random.default_rng().random())

    max_prob = float(np.max(probs_np))
    sorted_indices = np.argsort(-probs_np)
    rank = int(np.where(sorted_indices == label)[0][0]) + 1  # 1-based rank

    if rank == 1:
        return u * max_prob
    return max_prob + (rank - 2 + u) * lambda_val


def register(cls: lazydispatch, func: Callable) -> None:
    """Register an implementation for a specific type."""
    saps_score_func.register(cls=cls, func=func)


# batch helper function for convenience
def saps_score_func_batch(
    probs: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    lambda_val: float = 0.1,
    us: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Batch version of SAPS Nonconformity Score."""
    probs_np = np.asarray(probs, dtype=float)
    n_samples = probs_np.shape[0]

    if us is None:
        us = np.random.default_rng().random(size=n_samples)

    # Get max probabilities for each sample
    max_probs = np.max(probs_np, axis=1)

    # Get ranks for each label, argsort along axis=1 in descending order
    sorted_indices = np.argsort(-probs_np, axis=1)

    # Find the rank (1-based) of each label
    # Compare each position in sorted_indices with the corresponding label
    rank_mask = sorted_indices == labels[:, None]
    ranks = np.argmax(rank_mask, axis=1) + 1  # +1 for 1-based rank

    # Compute scores based on ranks
    scores = np.where(
        ranks == 1,
        us * max_probs,
        max_probs + (ranks - 2 + us) * lambda_val,
    )
    return scores


class SAPSScore:
    """Sorted Adaptive Prediction Sets (SAPS) nonconformity score."""

    def __init__(self, model: Predictor, lambda_val: float = 0.1, random_state: int | None = None) -> None:
        """Initialize SAPS score."""
        self.model = model
        self.lambda_val = lambda_val
        self.rng = np.random.default_rng(random_state)

    def calibration_nonconformity(
        self,
        x_calib: Sequence[Any],
        y_calib: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores for calibration data."""
        probs: npt.NDArray[np.floating]
        probs = predict_probs(self.model, x_calib)
        labels_np = np.asarray(y_calib, dtype=int)

        us = self.rng.uniform(0, 1, size=len(labels_np))

        scores = saps_score_func_batch(
            probs=probs,
            labels=labels_np,
            lambda_val=self.lambda_val,
            us=us,
        )
        return scores

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute scores for all labels."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        probs_np = np.asarray(probs, dtype=float)

        # Ensure 2D shape
        if probs_np.ndim == 1:
            probs_np = probs_np.reshape(1, -1)

        n_samples, n_classes = probs_np.shape

        # Create label array for all classes
        labels_all = np.tile(np.arange(n_classes), (n_samples, 1))

        # Generate random values for all samples and classes
        us_all = self.rng.uniform(0, 1, size=(n_samples, n_classes))

        # Get max probabilities for each sample (repeated for classes)
        max_probs = np.max(probs_np, axis=1)
        max_probs_expanded = max_probs[:, np.newaxis].repeat(n_classes, axis=1)

        # Get ranks for all labels
        # Argsort each sample's probabilities in descending order
        sorted_indices = np.argsort(-probs_np, axis=1)

        # Find ranks for all labels
        # Compare sorted_indices with each label
        rank_mask = sorted_indices[:, np.newaxis, :] == labels_all[:, :, np.newaxis]
        ranks = np.argmax(rank_mask, axis=2) + 1  # +1 for 1-based rank

        # Compute scores based on ranks
        scores = np.where(
            ranks == 1,
            us_all * max_probs_expanded,
            max_probs_expanded + (ranks - 2 + us_all) * self.lambda_val,
        )
        return scores
