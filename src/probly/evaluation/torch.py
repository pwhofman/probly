"""PyTorch implementations of :func:`coverage` and :func:`efficiency`.

Mirrors :mod:`probly.evaluation.array` for the torch-backed conformal-set and
credal-set representations. Semantics match the NumPy implementations exactly;
the only differences are the use of :func:`torch.gather` in place of
:func:`numpy.take_along_axis` and the explicit ``.float().mean().item()``
reduction at the end of each handler.

Note:
    There is no ``TorchSingletonCredalSet`` or ``TorchDiscreteCredalSet`` in
    :mod:`probly.representation.credal_set.torch`; for those semantics, use
    the numpy-side ``ArraySingletonCredalSet`` / ``ArrayDiscreteCredalSet``
    types. The remaining torch credal sets (Convex, DistanceBased,
    ProbabilityIntervals, DirichletLevelSet) all use the interval-dominance
    rule via their ``lower()`` / ``upper()`` envelopes.
"""

from __future__ import annotations

import torch

from probly.evaluation.metrics import average_interval_width, coverage, efficiency
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)


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


@coverage.register(TorchConvexCredalSet)
def _coverage_torch_convex(y_pred: TorchConvexCredalSet, y_true: torch.Tensor) -> float:
    """Interval-dominance coverage for a convex credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchConvexCredalSet)
def _efficiency_torch_convex(y_pred: TorchConvexCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a convex credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


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
    """Interval-dominance coverage for a probability-intervals credal set."""
    return _envelope_coverage(y_pred.lower(), y_pred.upper(), y_true)


@efficiency.register(TorchProbabilityIntervalsCredalSet)
def _efficiency_torch_probability_intervals(y_pred: TorchProbabilityIntervalsCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a probability-intervals credal set."""
    return _envelope_efficiency(y_pred.lower(), y_pred.upper())


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
