"""PyTorch implementations of :func:`coverage` and :func:`efficiency`.

Mirrors :mod:`probly.evaluation.array` for the torch-backed conformal-set and
credal-set representations. Semantics match the NumPy implementations exactly;
the only differences are the use of ``torch.gather`` in place of
``np.take_along_axis`` and the explicit ``.float().mean().item()`` reduction
at the end of each handler.
"""

from __future__ import annotations

import torch

from probly.evaluation.metrics import coverage, efficiency
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
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
    """Look up the membership flag for the true class along the last axis."""
    indices = y_true.to(dtype=torch.int64).unsqueeze(-1)
    return torch.gather(mask, -1, indices).squeeze(-1)


@coverage.register(TorchOneHotConformalSet)
def _coverage_torch_onehot(y_pred: TorchOneHotConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for a one-hot conformal set."""
    membership = _onehot_membership(y_pred.tensor, torch.as_tensor(y_true))
    return float(membership.float().mean().item())


@efficiency.register(TorchOneHotConformalSet)
def _efficiency_torch_onehot(y_pred: TorchOneHotConformalSet) -> float:
    """Average cardinality of a one-hot conformal set."""
    return float(y_pred.set_size.float().mean().item())


@coverage.register(TorchIntervalConformalSet)
def _coverage_torch_interval(y_pred: TorchIntervalConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for an interval conformal set."""
    arr = y_pred.tensor
    y = torch.as_tensor(y_true)
    inside = (y >= arr[..., 0]) & (y <= arr[..., 1])
    return float(inside.float().mean().item())


@efficiency.register(TorchIntervalConformalSet)
def _efficiency_torch_interval(y_pred: TorchIntervalConformalSet) -> float:
    """Average width of an interval conformal set."""
    return float(y_pred.set_size.float().mean().item())


def _envelope_coverage(lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(_onehot_membership(mask, torch.as_tensor(y_true)).float().mean().item())


def _envelope_efficiency(lower: torch.Tensor, upper: torch.Tensor) -> float:
    mask = _interval_dominance_mask(lower, upper)
    return float(mask.float().sum(dim=-1).mean().item())


@coverage.register(TorchConvexCredalSet)
def _coverage_torch_convex(y_pred: TorchConvexCredalSet, y_true: torch.Tensor) -> float:
    """Coverage for a convex credal set: any vertex's argmax matches the true class."""
    probs = y_pred.tensor.unnormalized_probabilities
    argmax_per_vertex = torch.argmax(probs, dim=-1)
    y = torch.as_tensor(y_true).unsqueeze(-1)
    matches = (argmax_per_vertex == y).any(dim=-1)
    return float(matches.float().mean().item())


@efficiency.register(TorchConvexCredalSet)
def _efficiency_torch_convex(y_pred: TorchConvexCredalSet) -> float:
    """Average number of distinct argmax classes across the vertex set."""
    probs = y_pred.tensor.unnormalized_probabilities
    num_classes = probs.shape[-1]
    argmax_per_vertex = torch.argmax(probs, dim=-1)
    class_indices = torch.arange(num_classes, device=probs.device)
    classes_picked = (argmax_per_vertex.unsqueeze(-1) == class_indices).any(dim=-2)
    return float(classes_picked.float().sum(dim=-1).mean().item())


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
    return _envelope_coverage(y_pred.lower_bounds, y_pred.upper_bounds, y_true)


@efficiency.register(TorchProbabilityIntervalsCredalSet)
def _efficiency_torch_probability_intervals(y_pred: TorchProbabilityIntervalsCredalSet) -> float:
    """Interval-dominance prediction-set cardinality for a probability-intervals credal set."""
    return _envelope_efficiency(y_pred.lower_bounds, y_pred.upper_bounds)
