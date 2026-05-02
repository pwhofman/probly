"""Metrics for evaluating Bayesian optimization runs."""

from __future__ import annotations

import numpy as np
import torch


def best_so_far(y: torch.Tensor) -> torch.Tensor:
    """Return the running minimum of an observation sequence.

    Args:
        y: 1-D tensor of evaluations in evaluation order.

    Returns:
        Tensor of the same shape giving the cumulative minimum.
    """
    return torch.cummin(y.view(-1), dim=0).values


def simple_regret(best_y: float, optimal_value: float) -> float:
    """Return the simple regret ``best_y - optimal_value``.

    For a minimization objective this is non-negative and equals zero only
    when the global minimum has been found.

    Args:
        best_y: Minimum value observed so far.
        optimal_value: Known global minimum of the objective.

    Returns:
        The non-negative regret.
    """
    return float(best_y - optimal_value)


def regret_curve(y: torch.Tensor, optimal_value: float) -> torch.Tensor:
    """Return per-iteration simple regret of the running best.

    Args:
        y: Evaluations in iteration order, shape ``(n,)``.
        optimal_value: Known global minimum of the objective.

    Returns:
        1-D tensor of shape ``(n,)`` whose ``i``-th entry is the regret of
        the running best after ``i + 1`` evaluations.
    """
    return best_so_far(y) - optimal_value


def regret_nauc(y: torch.Tensor, optimal_value: float) -> float:
    """Compute the normalized AUC of the regret curve (lower is better).

    Normalizes by the area under a constant curve at the maximum observed
    regret, so the metric is in ``[0, 1]`` for typical runs (0 = found the
    optimum at the very first evaluation, 1 = no improvement at all).

    Args:
        y: Evaluations in iteration order, shape ``(n,)``.
        optimal_value: Known global minimum used as the regret reference.

    Returns:
        Normalized AUC value, or NaN if fewer than two iterations.
    """
    curve = regret_curve(y, optimal_value).detach().cpu().numpy().astype(float)
    if curve.size < 2:
        return float("nan")
    x = np.arange(curve.size, dtype=float)
    actual_auc = float(np.trapezoid(curve, x=x))
    ref = float(curve.max() * (x[-1] - x[0]))
    if ref == 0.0:
        return 0.0
    return actual_auc / ref
