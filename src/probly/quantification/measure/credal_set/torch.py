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

from ._common import (
    Approximate,
    LogBase,
    generalized_hartley,
    lower_entropy,
    upper_entropy,
    use_approximate_lower_entropy,
)

_BISECT_ITERS = 64
_LBFGS_ITERS = 128


def _apply_base(result: torch.Tensor, n_classes: int, base: LogBase) -> torch.Tensor:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]


def _approximate_min_entropy_distribution(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    Since entropy is concave the minimum is at an extreme point of the polytope.
    This greedy heuristic tries each class as the primary recipient of excess
    mass and keeps the configuration with the lowest entropy found.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
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
        improved = h < best
        best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best = torch.minimum(best, h)
    return best, best_p


def _exact_min_entropy_distribution(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` exactly.

    Since entropy is concave the minimum is at an extreme point of the polytope. At an extreme
    point, every class but one sits at its lower or upper bound, and the remaining class, the
    free class, takes the rest of the mass. For every free class, all subsets of the other
    classes are tried at their upper bound, and a subset gives an extreme point if the rest of
    the mass fits within the bounds of the free class. The entropy of a candidate is a sum over
    the classes, so the sums over all subsets are one matrix product.

    Bounds that admit no distribution keep ``lower`` if ``sum(lower) > 1`` and ``upper`` if
    ``sum(upper) < 1``, as the greedy heuristic does.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - lower.sum(-1, keepdim=True)
    # Row s marks the classes of subset s, for all 2**C subsets.
    codes = torch.arange(2**n_classes, device=lower.device).unsqueeze(-1)
    subsets = ((codes >> torch.arange(n_classes, device=lower.device)) & 1).to(lower.dtype)
    tolerance = 8 * n_classes * torch.finfo(lower.dtype).eps
    lower_terms = torch.special.entr(lower)
    gain = torch.special.entr(upper) - lower_terms
    is_class = torch.arange(n_classes, device=lower.device)
    best = torch.full_like(residual.squeeze(-1), torch.inf)
    best_p = torch.where(residual <= 0, lower, upper)
    for free in range(n_classes):
        raised = subsets[subsets[:, free] == 0]
        free_capacity = capacity[..., free : free + 1]
        # Mass left for the free class above its lower bound, for every subset.
        mass = residual - capacity @ raised.T
        fits = (mass >= -tolerance) & (mass <= free_capacity + tolerance)
        mass = torch.minimum(mass.clamp_min(0.0), free_capacity)
        entropy = (
            lower_terms.sum(-1, keepdim=True)
            - lower_terms[..., free : free + 1]
            + gain @ raised.T
            + torch.special.entr(lower[..., free : free + 1] + mass)
        )
        value, index = torch.where(fits, entropy, torch.inf).min(-1)
        p = torch.where(
            is_class == free, lower + mass.gather(-1, index.unsqueeze(-1)), lower + capacity * raised[index]
        )
        improved = value < best
        best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best = torch.where(improved, value, best)
    return torch_entropy(best_p), best_p


def _min_entropy_distribution(
    lower: torch.Tensor, upper: torch.Tensor, approximate: Approximate
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    Exactly, by trying every extreme point, or with the greedy heuristic, as selected by
    ``approximate`` (see :func:`lower_entropy`).

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.
        approximate: Whether to use the greedy heuristic, ``True``, ``False`` or ``"auto"``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
    if use_approximate_lower_entropy(lower.shape[-1], approximate):
        return _approximate_min_entropy_distribution(lower, upper)
    return _exact_min_entropy_distribution(lower, upper)


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
    approximate: Approximate = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a probability-intervals credal set.

    Minimize entropy over {p : lower <= p <= upper, sum(p) = 1}.

    Since entropy is concave the minimum is at an extreme point of the polytope.
    The exact value tries every extreme point. The approximation, selected with
    ``approximate`` (see :func:`lower_entropy`), is a greedy heuristic that tries
    each class as the primary recipient of excess mass.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds, approximate)
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


@lower_entropy.register_approx(TorchDistanceBasedCredalSet)
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


@upper_entropy.register(TorchConvexCredalSet)
def torch_convex_upper_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    result = _apply_base(torch_entropy(p).reshape(batch_shape), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes)
    return result


@lower_entropy.register(TorchConvexCredalSet)
def torch_convex_lower_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex.
    The result is exact for every value of ``approximate``.
    """
    del approximate
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
    approximate: Approximate = "auto",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the lower entropy of a Dirichlet level set credal set.

    Uses per-class bounds from Monte Carlo sampling, then minimizes the entropy
    over the extreme points of these bounds, exactly or, with ``approximate``,
    with the greedy heuristic.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower(), credal_set.upper(), approximate)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result
