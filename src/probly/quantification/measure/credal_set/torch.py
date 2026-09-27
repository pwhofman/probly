"""Uncertainty measures for torch-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import torch

from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchDirichletLevelSetCredalSet,
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.utils.torch import torch_entropy

from ._common import LogBase, generalized_hartley, lower_entropy, upper_entropy

_BISECT_ITERS = 64
_CONVEX_ITERS = 100


def _apply_base(result: torch.Tensor, n_classes: int, base: LogBase) -> torch.Tensor:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]


@upper_entropy.register(TorchProbabilityIntervalsCredalSet)
def torch_intervals_upper_entropy(
    credal_set: TorchProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    p = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper)
    result = _apply_base(torch_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(TorchProbabilityIntervalsCredalSet)
def torch_intervals_lower_entropy(
    credal_set: TorchProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    best_p = torch.empty_like(lower)
    for j in range(n_classes):
        p = lower.detach().clone()
        rem = residual.clone()
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = torch.minimum(rem.clamp(min=0.0), capacity[..., i])
            p[..., i] = p[..., i] + fill
            rem = rem - fill
        h = torch_entropy(p)
        if return_distribution:
            improved = h < best
            best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best = torch.minimum(best, h)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


@upper_entropy.register(TorchDistanceBasedCredalSet)
def torch_distance_based_upper_entropy(
    credal_set: TorchDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the upper entropy of a distance-based credal set.

    The TV ball {p : TV(p, p_hat) <= r} implies per-class bounds
    lower_i = max(0, p_hat_i - r), upper_i = min(1, p_hat_i + r).
    Uses bisection on the Lagrange multiplier for sum(p) = 1.
    """
    lower = credal_set.lower()
    upper = credal_set.upper()
    lo = 1.0 + lower.amin(dim=-1).clamp_min(torch.finfo(lower.dtype).tiny).log()
    hi = lower.new_ones(lower.shape[:-1])
    for _ in range(_BISECT_ITERS):
        mu = (lo + hi) / 2
        g = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper).sum(-1) - 1.0
        lo = torch.where(g < 0, mu, lo)
        hi = torch.where(g >= 0, mu, hi)
    mu = (lo + hi) / 2
    p = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper)
    result = _apply_base(torch_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(TorchDistanceBasedCredalSet)
def torch_distance_based_lower_entropy(
    credal_set: TorchDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a distance-based credal set.

    The TV ball implies per-class bounds. Since entropy is concave, the
    minimum is at an extreme point of the polytope. Greedy heuristic tries
    each class as the primary recipient of excess mass.
    """
    lower = credal_set.lower()
    upper = credal_set.upper()
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - lower.sum(-1)
    best = lower.new_full(lower.shape[:-1], float("inf"))
    best_p = torch.empty_like(lower)
    for j in range(n_classes):
        p = lower.detach().clone()
        rem = residual.clone()
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = torch.minimum(rem.clamp(min=0.0), capacity[..., i])
            p[..., i] = p[..., i] + fill
            rem = rem - fill
        h = torch_entropy(p)
        if return_distribution:
            improved = h < best
            best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best = torch.minimum(best, h)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


def _entropy_scores(vertices: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixture ``p = weights @ vertices``, entropy gradient ``g = -log(p) - 1`` and vertex scores ``vertices @ g``.

    The score of a vertex is the derivative of the entropy in its direction, up to a constant.
    The gradient is finite at exact zeros, where the log is taken of the smallest positive
    float, so a class that is zero in every vertex adds nothing to the scores.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        Tuple of the mixtures of shape ``(N, K)``, the gradients of shape ``(N, K)`` and the
        scores of shape ``(N, V)``.
    """
    p = (weights.unsqueeze(-1) * vertices).sum(-2)
    gradient = -p.clamp_min(torch.finfo(p.dtype).tiny).log() - 1.0
    return p, gradient, (vertices * gradient.unsqueeze(-2)).sum(-1)


def _entropy_line_search(p: torch.Tensor, direction: torch.Tensor, max_step: torch.Tensor) -> torch.Tensor:
    """Step ``t`` in ``[0, max_step]`` that maximizes ``H(p + t * direction)``.

    The entropy is concave along the line, so its derivative decreases in ``t`` and bisection,
    run to the precision of the dtype, finds where it changes sign. If the entropy still rises
    at ``max_step``, exactly ``max_step`` is returned, so that a mixture weight that reaches
    zero becomes exactly zero.

    Args:
        p: Distributions of shape ``(N, K)``.
        direction: Directions of shape ``(N, K)`` that sum to zero.
        max_step: Largest steps of shape ``(N,)``.

    Returns:
        Steps of shape ``(N,)`` that do not lower the entropy.
    """
    tiny = torch.finfo(p.dtype).tiny

    def slope(step: torch.Tensor) -> torch.Tensor:
        return -(direction * ((p + step.unsqueeze(-1) * direction).clamp_min(tiny).log() + 1.0)).sum(-1)

    low, high = torch.zeros_like(max_step), max_step
    for _ in range(round(-math.log2(torch.finfo(p.dtype).eps))):
        middle = (low + high) / 2
        rising = slope(middle) > 0
        low, high = torch.where(rising, middle, low), torch.where(rising, high, middle)
    return torch.where(slope(max_step) >= 0, max_step, low)


def _pairwise_step(vertices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pairwise Frank-Wolfe step: move weight from the worst vertex in use to the best vertex.

    Along the edge from the vertex in use with the smallest derivative to the vertex with the
    largest one, the entropy rises whenever the weights are not optimal. The exact line search
    may add the best vertex or drop the worst one, and the sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        The new weights of shape ``(N, V)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    toward = scores.argmax(-1, keepdim=True)
    away = torch.where(weights > 0, scores, torch.inf).argmin(-1, keepdim=True)
    ones = torch.ones_like(weights[..., :1])
    direction = torch.zeros_like(weights).scatter_add(-1, toward, ones).scatter_add(-1, away, -ones)
    step = _entropy_line_search(p, (direction.unsqueeze(-1) * vertices).sum(-2), weights.gather(-1, away).squeeze(-1))
    return (weights + step.unsqueeze(-1) * direction).clamp_min(0.0)


def _newton_step(vertices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Newton step on the weights of the vertices in use, followed by an exact line search.

    The Hessian of the entropy in the weights is ``-vertices diag(1 / p) vertices^T``. The step
    maximizes the second-order model among the vertices in use, keeping the sum of the weights,
    by solving its KKT system. A small ridge keeps the system solvable for repeated or affinely
    dependent vertices. The line search stops at the first weight that reaches zero, and the
    sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        The new weights of shape ``(N, V)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    in_use = weights > 0
    mask = in_use.to(weights.dtype)
    inverse = torch.where(p > 0, 1.0 / p.clamp_min(torch.finfo(p.dtype).tiny), 0.0)
    curvature = (vertices * inverse.unsqueeze(-2)) @ vertices.transpose(-1, -2)
    curvature = torch.where(in_use.unsqueeze(-1) & in_use.unsqueeze(-2), curvature, 0.0)
    diagonal = curvature.diagonal(dim1=-2, dim2=-1)
    ridge = math.sqrt(torch.finfo(p.dtype).eps) * diagonal.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    system = curvature + torch.diag_embed(torch.where(in_use, ridge, 1.0))
    zero = torch.zeros_like(mask[..., :1])
    kkt = torch.cat([torch.cat([system, mask.unsqueeze(-1)], -1), torch.cat([mask, zero], -1).unsqueeze(-2)], -2)
    solution, info = torch.linalg.solve_ex(kkt, torch.cat([scores * mask, zero], -1))
    direction = torch.where(
        in_use & (info == 0).unsqueeze(-1), solution[..., :-1].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    # Remove the rounding error of the solve, so that the weights keep their sum.
    direction = (direction - (direction * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
    ratio = torch.where(direction < 0, weights / -direction, torch.inf)
    max_step = ratio.amin(-1)
    max_step = torch.where(max_step.isfinite(), max_step, 0.0)
    step = _entropy_line_search(p, (direction.unsqueeze(-1) * vertices).sum(-2), max_step).unsqueeze(-1)
    return torch.where(ratio <= step, 0.0, weights + step * direction).clamp_min(0.0)


def _convex_max_entropy_weights(vertices: torch.Tensor) -> torch.Tensor:
    """Entropy-maximizing mixture weights of convex hulls, each hull optimized on its own.

    Entropy is concave in the mixture weights, so every stationary point is a maximum. Each
    iteration makes a pairwise Frank-Wolfe step, which can add and drop vertices, and a Newton
    step on the vertices in use, which converges fast once the right vertices are in use. Both
    keep the weights nonnegative and summing to one. The Frank-Wolfe gap
    ``max_v vertices[v] @ g - p @ g`` bounds how far ``H(p)`` is below the maximum, since
    entropy is concave. A set stops once its gap is at most 1e-10 in float64 or 1e-5 in other
    dtypes. It also stops when an iteration leaves its weights unchanged, since every later
    iteration would do the same, and after at most ``_CONVEX_ITERS`` iterations. Every step
    only uses the set's own vertices, so the result does not depend on the other sets in the
    batch.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.

    Returns:
        Mixture weights of shape ``(N, V)``.
    """
    tolerance = 1e-10 if vertices.dtype == torch.float64 else 1e-5
    weights = vertices.new_full(vertices.shape[:-1], 1.0 / vertices.shape[-2])
    active = torch.arange(vertices.shape[0], device=vertices.device)
    for _ in range(_CONVEX_ITERS):
        if active.numel() == 0:
            break
        current, points = weights[active], vertices[active]
        p, gradient, scores = _entropy_scores(points, current)
        gap = scores.amax(-1) - (p * gradient).sum(-1)
        converged = gap <= tolerance
        updated = _newton_step(points, _pairwise_step(points, current))
        weights[active] = torch.where(converged.unsqueeze(-1), current, updated)
        active = active[~converged & (updated != current).any(-1)]
    return weights / weights.sum(-1, keepdim=True)


@upper_entropy.register(TorchConvexCredalSet)
def torch_convex_upper_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over conv(vertices), which may be attained in the interior since entropy
    is concave. Each set is optimized on its own until a bound certifies that it is within
    1e-10 (float64) or 1e-5 (other dtypes) of the maximum; see ``_convex_max_entropy_weights``.
    Half-precision sets are optimized in float32, and the results are cast back.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    # torch.linalg cannot solve in half precision, so optimize in at least single precision.
    flat_v = vertices.detach().reshape(-1, n_vertices, n_classes).to(torch.promote_types(vertices.dtype, torch.float32))

    with torch.no_grad():
        p = (_convex_max_entropy_weights(flat_v).unsqueeze(-1) * flat_v).sum(-2)
    result = _apply_base(torch_entropy(p).reshape(batch_shape).to(vertices.dtype), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes).to(vertices.dtype)
    return result


@lower_entropy.register(TorchConvexCredalSet)
def torch_convex_lower_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex.
    """
    vertices = credal_set.tensor.probabilities  # (..., n_vertices, n_classes)
    vertex_entropies = torch_entropy(vertices)  # (..., n_vertices)
    if return_distribution:
        min_h, min_idx = vertex_entropies.min(-1)
        n_classes = vertices.shape[-1]
        gather_idx = min_idx.unsqueeze(-1).unsqueeze(-1).expand(*min_idx.shape, 1, n_classes)
        best_p = vertices.gather(-2, gather_idx).squeeze(-2)
        return _apply_base(min_h, credal_set.num_classes, base), best_p
    return _apply_base(vertex_entropies.min(-1).values, credal_set.num_classes, base)


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


@upper_entropy.register(TorchDirichletLevelSetCredalSet)
def torch_dirichlet_level_set_upper_entropy(
    credal_set: TorchDirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the upper entropy of a Dirichlet level set credal set.

    Uses per-class bounds from Monte Carlo sampling, then applies the
    bisection algorithm for sum(p) = 1 constrained entropy maximization.
    """
    lower = credal_set.lower()
    upper = credal_set.upper()
    lo = 1.0 + lower.amin(dim=-1).clamp_min(torch.finfo(lower.dtype).tiny).log()
    hi = lower.new_ones(lower.shape[:-1])
    for _ in range(_BISECT_ITERS):
        mu = (lo + hi) / 2
        g = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper).sum(-1) - 1.0
        lo = torch.where(g < 0, mu, lo)
        hi = torch.where(g >= 0, mu, hi)
    mu = (lo + hi) / 2
    p = torch.clamp((mu.unsqueeze(-1) - 1).exp(), lower, upper)
    result = _apply_base(torch_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(TorchDirichletLevelSetCredalSet)
def torch_dirichlet_level_set_lower_entropy(
    credal_set: TorchDirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a Dirichlet level set credal set.

    Uses per-class bounds from Monte Carlo sampling, then applies the
    greedy heuristic for entropy minimization at extreme points.
    """
    lower = credal_set.lower()
    upper = credal_set.upper()
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - lower.sum(-1)
    best = lower.new_full(lower.shape[:-1], float("inf"))
    best_p = torch.empty_like(lower)
    for j in range(n_classes):
        p = lower.detach().clone()
        rem = residual.clone()
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = torch.minimum(rem.clamp(min=0.0), capacity[..., i])
            p[..., i] = p[..., i] + fill
            rem = rem - fill
        h = torch_entropy(p)
        if return_distribution:
            improved = h < best
            best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best = torch.minimum(best, h)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result
