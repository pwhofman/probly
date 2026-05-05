"""PyTorch implementation of Metrics.

Note:
    There is no ``TorchSingletonCredalSet`` or ``TorchDiscreteCredalSet`` in
    :mod:`probly.representation.credal_set.torch`; for those semantics, use
    the numpy-side ``ArraySingletonCredalSet`` / ``ArrayDiscreteCredalSet``
    types. The remaining torch credal sets (Convex, DistanceBased,
    ProbabilityIntervals, DirichletLevelSet) all use the interval-dominance
    rule via their ``lower()`` / ``upper()`` envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from probly.metrics import (
    auc,
    average_interval_width,
    average_precision_score,
    convex_hull_coverage,
    coverage,
    efficiency,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from probly.metrics.array import _convex_hull_lp_coverage
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)

if TYPE_CHECKING:
    from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution


@auc.register(torch.Tensor)
def auc_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute area under a curve using the trapezoid rule."""
    return torch.trapezoid(y, x, dim=-1)


@average_precision_score.register(torch.Tensor)
def average_precision_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute average precision for PyTorch tensors."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -torch.sum(torch.diff(recall, dim=-1) * precision[..., :-1], dim=-1)  # ty:ignore[invalid-argument-type, not-subscriptable]


@precision_recall_curve.register(torch.Tensor)
def precision_recall_curve_torch(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()
    n = y_score.shape[-1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.take_along_dim(y_score, desc_idx, dim=-1)
    y_true_sorted = torch.take_along_dim(y_true, desc_idx, dim=-1)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    predicted_pos = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype)
    total_pos = tps[..., -1:]

    precision = tps / predicted_pos
    recall = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )

    ones = torch.ones((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    precision = torch.cat([precision.flip(-1), ones], dim=-1)
    recall = torch.cat([recall.flip(-1), zeros], dim=-1)

    return precision, recall, y_score_sorted


@roc_auc_score.register(torch.Tensor)
def roc_auc_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute area under the ROC curve for PyTorch tensors."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]


@roc_curve.register(torch.Tensor)
def roc_curve_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()
    n = y_score.shape[-1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.take_along_dim(y_score, desc_idx, dim=-1)
    y_true_sorted = torch.take_along_dim(y_true, desc_idx, dim=-1)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    fps = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype) - tps

    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )
    fpr = torch.where(
        total_neg > 0, fps / torch.where(total_neg > 0, total_neg, torch.ones_like(total_neg)), torch.zeros_like(fps)
    )

    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    tpr = torch.cat([zeros, tpr], dim=-1)
    fpr = torch.cat([zeros, fpr], dim=-1)
    thresholds = torch.cat([y_score_sorted[..., :1] + 1, y_score_sorted], dim=-1)

    return fpr, tpr, thresholds


# --- Predicted-set metrics ----------------------------------------------------


def _interval_dominance_mask(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Return the boolean class mask selected by the interval-dominance rule.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Boolean tensor of shape ``(..., C)``.
    """
    threshold, _ = torch.max(lower, dim=-1, keepdim=True)
    return upper >= threshold


def _onehot_membership(mask: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Look up the membership flag for the true class along the last axis.

    ``y_true`` is coerced onto ``mask.device``; transfer cost falls on the
    caller if the devices mismatch.
    """
    indices = torch.as_tensor(y_true, device=mask.device).to(dtype=torch.int64).unsqueeze(-1)
    return torch.gather(mask, -1, indices).squeeze(-1)


def _envelope_coverage(lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(_onehot_membership(mask, y_true).float().mean().item())


def _envelope_efficiency(lower: torch.Tensor, upper: torch.Tensor) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(mask.float().sum(dim=-1).mean().item())


def _envelope_average_interval_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    return float((upper - lower).float().mean().item())


@coverage.register(TorchOneHotConformalSet)
def _coverage_torch_onehot(y_pred: TorchOneHotConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for a one-hot conformal set."""
    membership = _onehot_membership(y_pred.tensor, y_true)
    return float(membership.float().mean().item())


@efficiency.register(TorchOneHotConformalSet)
def _efficiency_torch_onehot(y_pred: TorchOneHotConformalSet) -> float:
    """Average cardinality of a one-hot conformal set."""
    return float(y_pred.tensor.float().sum(dim=-1).mean().item())


@coverage.register(TorchIntervalConformalSet)
def _coverage_torch_interval(y_pred: TorchIntervalConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for an interval conformal set."""
    arr = y_pred.tensor
    y = torch.as_tensor(y_true, device=arr.device)
    inside = (y >= arr[..., 0]) & (y <= arr[..., 1])
    return float(inside.float().mean().item())


@efficiency.register(TorchIntervalConformalSet)
def _efficiency_torch_interval(y_pred: TorchIntervalConformalSet) -> float:
    """Average width of an interval conformal set."""
    return float((y_pred.tensor[..., 1] - y_pred.tensor[..., 0]).float().mean().item())


def _credal_containment_coverage_torch(lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of instances where ``y_true`` lies in ``[lower, upper]`` for all classes.

    Args:
        lower: Lower probability envelope of shape ``(N, C)``.
        upper: Upper probability envelope of shape ``(N, C)``.
        y_true: Target probability tensors of shape ``(N, C)``.

    Returns:
        Mean containment indicator as a Python float.
    """
    y = torch.as_tensor(y_true, device=lower.device, dtype=lower.dtype)
    covered = ((lower <= y) & (y <= upper)).all(dim=-1)
    return float(covered.float().mean().item())


def _credal_interval_efficiency_torch(lower: torch.Tensor, upper: torch.Tensor) -> float:
    """Efficiency of a credal set as ``1 - mean(upper - lower)``.

    Args:
        lower: Lower probability envelope of shape ``(N, C)``.
        upper: Upper probability envelope of shape ``(N, C)``.

    Returns:
        Scalar in ``(-inf, 1]``; higher means a tighter (more efficient) credal set.
    """
    return float(1.0 - (upper - lower).float().mean().item())


@coverage.register(TorchConvexCredalSet)
def _coverage_torch_convex(y_pred: TorchConvexCredalSet, y_true: torch.Tensor) -> float:
    """Containment coverage for a convex credal set.

    Args:
        y_pred: Convex credal set.
        y_true: Target probability tensors of shape ``(N, C)``.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage_torch(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchConvexCredalSet)
def _efficiency_torch_convex(y_pred: TorchConvexCredalSet) -> float:
    """Interval-width efficiency for a convex credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency_torch(y_pred.lower(), y_pred.upper())


@coverage.register(TorchDistanceBasedCredalSet)
def _coverage_torch_distance(y_pred: TorchDistanceBasedCredalSet, y_true: torch.Tensor) -> float:
    """Interval-dominance coverage for a distance-based credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchDistanceBasedCredalSet)
def _efficiency_torch_distance(y_pred: TorchDistanceBasedCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a distance-based credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@coverage.register(TorchProbabilityIntervalsCredalSet)
def _coverage_torch_probability_intervals(y_pred: TorchProbabilityIntervalsCredalSet, y_true: torch.Tensor) -> float:
    """Containment coverage for a probability-intervals credal set.

    Args:
        y_pred: Probability-intervals credal set.
        y_true: Target probability tensors of shape ``(N, C)``.

    Returns:
        Fraction of instances where the target lies in ``[lower, upper]`` for all classes.
    """
    return _credal_containment_coverage_torch(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchProbabilityIntervalsCredalSet)
def _efficiency_torch_probability_intervals(y_pred: TorchProbabilityIntervalsCredalSet) -> float:
    """Interval-width efficiency for a probability-intervals credal set: ``1 - mean(upper - lower)``.

    Returns:
        Scalar efficiency; higher means a tighter credal set.
    """
    return _credal_interval_efficiency_torch(y_pred.lower(), y_pred.upper())


@coverage.register(TorchDirichletLevelSetCredalSet)
def _coverage_torch_dirichlet_level_set(y_pred: TorchDirichletLevelSetCredalSet, y_true: torch.Tensor) -> float:
    """Interval-dominance coverage for a Dirichlet-level-set credal set.

    The lower/upper envelopes are estimated by Monte-Carlo sampling, so the
    result is stochastic; pin a torch seed for reproducible values.
    """
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchDirichletLevelSetCredalSet)
def _efficiency_torch_dirichlet_level_set(y_pred: TorchDirichletLevelSetCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a Dirichlet-level-set credal set.

    The lower/upper envelopes are estimated by Monte-Carlo sampling, so the
    result is stochastic; pin a torch seed for reproducible values.
    """
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


@average_interval_width.register(TorchConvexCredalSet)
def _average_interval_width_torch_convex(y_pred: TorchConvexCredalSet) -> float:
    """Mean per-class width of the vertex-derived envelope of a convex credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(TorchDistanceBasedCredalSet)
def _average_interval_width_torch_distance(y_pred: TorchDistanceBasedCredalSet) -> float:
    """Mean per-class width of the L1-clip envelope of a distance-based credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(TorchProbabilityIntervalsCredalSet)
def _average_interval_width_torch_probability_intervals(y_pred: TorchProbabilityIntervalsCredalSet) -> float:
    """Mean per-class interval width of a probability-intervals credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@average_interval_width.register(TorchDirichletLevelSetCredalSet)
def _average_interval_width_torch_dirichlet_level_set(y_pred: TorchDirichletLevelSetCredalSet) -> float:
    """Mean per-class width of the MC-sampled envelope of a Dirichlet-level-set credal set."""
    return _envelope_average_interval_width(y_pred.lower(), y_pred.upper())


@convex_hull_coverage.register(TorchConvexCredalSet)
def _convex_hull_coverage_torch_convex(
    y_pred: TorchConvexCredalSet,
    y_true: TorchCategoricalDistribution,
    *,
    epsilon: float = 0.0,
    **linprog_kwargs: object,
) -> object:
    """Hull coverage for a torch convex credal set; routes through the numpy LP solver.

    ``scipy.linprog`` is numpy-only, so the vertex tensor and target tensor
    are detached, moved to CPU, and converted to numpy arrays.
    """
    vertices = y_pred.tensor.probabilities.detach().cpu().numpy()
    targets = y_true.probabilities.detach().cpu().numpy()
    return _convex_hull_lp_coverage(vertices, targets, epsilon, **linprog_kwargs)
