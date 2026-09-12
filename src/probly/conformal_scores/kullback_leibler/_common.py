"""Kullback-Leibler divergence nonconformity score implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flextype import flexdispatch
import numpy as np

from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike
from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample.array import ArraySample


@flexdispatch
def kl_divergence_score_func[T](y_pred: T, y_true: T | None = None) -> T:
    """Compute KL divergence with type-based target semantics.

    Args:
        y_pred: Predicted categorical probabilities or a categorical representation.
        y_true: Integer class labels, floating-point probability vectors, or a
            categorical representation. Integer arrays always encode labels,
            including arrays containing only zeros and ones. Batch dimensions
            broadcast normally; shapes never select the target interpretation.

    Returns:
        Scores with the broadcast batch shape in the prediction backend.
    """
    msg = "Kullback-Leibler divergence score not implemented for this type."
    raise NotImplementedError(msg)


@kl_divergence_score_func.register(np.ndarray | ArrayLike)
def compute_kl_divergence_score_numpy(y_pred: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Computes the Kullback-Leibler divergence score using NumPy Arrays.

    Args:
        y_pred: Predicted probabilities.
        y_true: Ground truth labels (class indices or probability vectors).
    """
    y_pred_np = np.asarray(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_np = np.asarray(y_true.probabilities if distribution_target else y_true)
    if y_pred_np.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    eps = 1e-12
    if not distribution_target and np.issubdtype(y_true_np.dtype, np.integer):
        batch_shape = np.broadcast_shapes(y_pred_np.shape[:-1], y_true_np.shape)
        probabilities = np.broadcast_to(y_pred_np, (*batch_shape, y_pred_np.shape[-1]))
        labels = np.broadcast_to(y_true_np, batch_shape)
        selected = np.take_along_axis(probabilities, labels[..., None], axis=-1).squeeze(-1)
        return -np.log(np.clip(selected, eps, 1.0))

    if not distribution_target and not np.issubdtype(y_true_np.dtype, np.floating):
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_np.ndim == 0 or y_true_np.shape[-1] != y_pred_np.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    y_pred_safe = np.clip(y_pred_np, eps, 1.0)
    y_true_safe = np.clip(y_true_np, eps, 1.0)

    return np.sum(y_true_np * np.log(y_true_safe / y_pred_safe), axis=-1)


@kl_divergence_score_func.register(ArrayCategoricalDistribution)
def _(y_pred: ArrayCategoricalDistribution, y_true: np.ndarray) -> np.ndarray:
    """Compute the score from normalized categorical probabilities."""
    return kl_divergence_score_func(y_pred.probabilities, y_true)


@kl_divergence_score_func.register(ArraySample)
def _(y_pred: ArraySample, y_true: np.ndarray) -> np.ndarray:
    """Compute memberwise scores for NumPy samples."""
    return kl_divergence_score_func(y_pred.array, y_true)


@dataclass(frozen=True, slots=True)
class KLDivergenceScore[In, Out](NonConformityScore):
    """Kullback-Leibler divergence non-conformity score."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:  # noqa: ANN401
        if y_true is None:
            msg = "y_true is required for Kullback-Leibler divergence."
            raise ValueError(msg)
        return kl_divergence_score_func(y_pred, y_true)


kl_divergence_score = KLDivergenceScore()

__all__ = ["KLDivergenceScore", "kl_divergence_score"]
