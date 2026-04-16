"""Uncertainty measures for torch-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import torch

from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet
from probly.utils.torch import torch_entropy

from ._common import LogBase, generalized_hartley, lower_entropy, upper_entropy

_BISECT_ITERS = 64
_LBFGS_ITERS = 128


def _apply_base(result: torch.Tensor, n_classes: int, base: LogBase) -> torch.Tensor:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]


@upper_entropy.register(TorchProbabilityIntervalsCredalSet)
def torch_intervals_upper_entropy(
    credal_set: TorchProbabilityIntervalsCredalSet,
    base: LogBase = None,
) -> torch.Tensor:
    """Compute the upper entropy of a probability-intervals credal set.

    Maximize entropy over {p : lower <= p <= upper, sum(p) = 1}.

    The KKT conditions give p_i = clip(exp(mu - 1), lower_i, upper_i) where mu
    is the Lagrange multiplier for sum(p) = 1. Since the sum is monotone in mu,
    bisection finds the unique root.
    """
    lower, upper = credal_set.lower_bounds, credal_set.upper_bounds
    lo = 1.0 + lower.amin(dim=-1).clamp_min(torch.finfo(lower.dtype).tiny).log()
    hi = lower.new_ones(lower.shape[:-1])
    for _ in range(_BISECT_ITERS):
        mu = (lo + hi) / 2
        g = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper).sum(-1) - 1.0
        lo = torch.where(g < 0, mu, lo)
        hi = torch.where(g >= 0, mu, hi)
    mu = (lo + hi) / 2
    result = torch_entropy(torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper))
    return _apply_base(result, credal_set.num_classes, base)


@lower_entropy.register(TorchProbabilityIntervalsCredalSet)
def torch_intervals_lower_entropy(credal_set: TorchProbabilityIntervalsCredalSet, base: LogBase = None) -> torch.Tensor:
    """Compute the lower entropy of a probability-intervals credal set.

    Minimize entropy over {p : lower <= p <= upper, sum(p) = 1}.

    Since entropy is concave the minimum is at an extreme point of the polytope.
    This greedy heuristic tries each class as the primary recipient of excess
    mass and returns the configuration with the lowest entropy found.
    """
    lower, upper = credal_set.lower_bounds, credal_set.upper_bounds
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - lower.sum(-1)
    best = lower.new_full(lower.shape[:-1], float("inf"))
    for j in range(n_classes):
        p = lower.detach().clone()
        rem = residual.clone()
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = torch.minimum(rem.clamp(min=0.0), capacity[..., i])
            p[..., i] = p[..., i] + fill
            rem = rem - fill
        best = torch.minimum(best, torch_entropy(p))
    return _apply_base(best, credal_set.num_classes, base)


@upper_entropy.register(TorchConvexCredalSet)
def torch_convex_upper_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
) -> torch.Tensor:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over conv(vertices) via L-BFGS on softmax weights.

    Since entropy is concave the maximum over a convex hull may be in the
    interior; L-BFGS on the unconstrained softmax parameterization handles this.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    flat_v = vertices.detach().reshape(-1, n_vertices, n_classes)
    n = flat_v.shape[0]

    w_logits = flat_v.new_zeros(n, n_vertices).requires_grad_(True)
    opt = torch.optim.LBFGS([w_logits], max_iter=_LBFGS_ITERS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        p = (w_logits.softmax(-1).unsqueeze(-1) * flat_v).sum(-2)
        loss = -torch_entropy(p).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        p = (w_logits.softmax(-1).unsqueeze(-1) * flat_v).sum(-2)
    result = torch_entropy(p).reshape(batch_shape)

    return _apply_base(result, credal_set.num_classes, base)


@lower_entropy.register(TorchConvexCredalSet)
def torch_convex_lower_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
) -> torch.Tensor:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex.
    """
    result = torch_entropy(credal_set.tensor.probabilities).min(-1).values
    return _apply_base(result, credal_set.num_classes, base)


def _lower_probability(vertices: torch.Tensor, subset: tuple[int, ...]) -> torch.Tensor:
    """Upper probability P*(A) = max_v sum_{i in A} v_i, shape (...)."""
    if not subset:
        return vertices.new_zeros(vertices.shape[:-2])
    return vertices[..., list(subset)].sum(-1).min(-1).values


def _moebius(vertices: torch.Tensor, subset: tuple[int, ...]) -> torch.Tensor:
    """Mobius mass m(A) via inclusion-exclusion over all subsets of A."""
    result = vertices.new_zeros(vertices.shape[:-2])
    for r in range(len(subset) + 1):
        sign = (-1) ** (len(subset) - r)
        for b in combinations(list(subset), r):
            result = result + sign * _lower_probability(vertices, b)
    return result


@generalized_hartley.register(TorchConvexCredalSet)
def torch_convex_generalized_hartley(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
) -> torch.Tensor:
    """Compute the generalized Hartley measure of a convex credal set.

    Based on :cite:`abellanDisaggregatedTotal2006`. Computed via the Mobius
    transform of the lower probability function over all subsets of the class
    space.
    """
    vertices = credal_set.tensor.probabilities  # (..., n_vertices, n_classes)
    n_classes = credal_set.num_classes
    log_b = None if base is None else math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]
    result = vertices.new_zeros(vertices.shape[:-2])
    for a in chain.from_iterable(combinations(range(n_classes), r) for r in range(1, n_classes + 1)):
        log_a = math.log(len(a)) / log_b if log_b else math.log(len(a))
        result = result + _moebius(vertices, a) * log_a
    return result
