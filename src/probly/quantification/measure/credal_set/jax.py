"""Uncertainty measures for jax-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import jax
import jax.numpy as jnp

from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.utils.jax import jax_entropy

from ._common import LogBase, generalized_hartley, lower_entropy, upper_entropy

_BISECT_ITERS = 64
_CONVEX_ITERS = 100


def _apply_base(result: jax.Array, n_classes: int, base: LogBase) -> jax.Array:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]


def _max_entropy_distribution(lower: jax.Array, upper: jax.Array) -> jax.Array:
    """Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    The KKT conditions give ``p_i = clip(exp(mu - 1), lower_i, upper_i)`` where
    ``mu`` is the Lagrange multiplier for ``sum(p) = 1``. Since the sum is
    monotone in ``mu``, bisection finds the unique root.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        The maximum-entropy distribution of shape ``(..., C)``.
    """
    tiny = float(jnp.finfo(lower.dtype).tiny)
    lo = 1.0 + jnp.log(jnp.clip(jnp.min(lower, axis=-1), min=tiny))
    hi = jnp.ones(lower.shape[:-1], dtype=lower.dtype)
    for _ in range(_BISECT_ITERS):
        mu = (lo + hi) / 2
        g = jnp.sum(jnp.clip(jnp.exp(jnp.expand_dims(mu, axis=-1) - 1), lower, upper), axis=-1) - 1.0
        lo = jnp.where(g < 0, mu, lo)
        hi = jnp.where(g >= 0, mu, hi)
    mu = (lo + hi) / 2
    return jnp.clip(jnp.exp(jnp.expand_dims(mu, axis=-1) - 1), lower, upper)


def _min_entropy_distribution(lower: jax.Array, upper: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    residual = 1.0 - jnp.sum(lower, axis=-1)
    best = jnp.full(lower.shape[:-1], jnp.inf, dtype=lower.dtype)
    best_p = jnp.zeros_like(lower)
    for j in range(n_classes):
        p = lower
        rem = residual
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = jnp.minimum(jnp.clip(rem, min=0.0), capacity[..., i])
            p = p.at[..., i].add(fill)
            rem = rem - fill
        h = jax_entropy(p)
        improved = h < best
        best_p = jnp.where(jnp.expand_dims(improved, axis=-1), p, best_p)
        best = jnp.minimum(best, h)
    return best, best_p


@upper_entropy.register(JaxProbabilityIntervalsCredalSet)
def jax_intervals_upper_entropy(
    credal_set: JaxProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a probability-intervals credal set.

    Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` by bisection
    on the Lagrange multiplier of the sum constraint.
    """
    p = _max_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds)
    result = _apply_base(jax_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(JaxProbabilityIntervalsCredalSet)
def jax_intervals_lower_entropy(
    credal_set: JaxProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a probability-intervals credal set.

    Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` with the
    greedy extreme-point heuristic.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


@upper_entropy.register(JaxDistanceBasedCredalSet)
def jax_distance_based_upper_entropy(
    credal_set: JaxDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a distance-based credal set.

    The TV ball ``{p : TV(p, p_hat) <= r}`` implies per-class bounds
    ``lower_i = max(0, p_hat_i - r)`` and ``upper_i = min(1, p_hat_i + r)``.
    Uses bisection on the Lagrange multiplier for ``sum(p) = 1``.
    """
    p = _max_entropy_distribution(credal_set.lower(), credal_set.upper())
    result = _apply_base(jax_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(JaxDistanceBasedCredalSet)
def jax_distance_based_lower_entropy(
    credal_set: JaxDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a distance-based credal set.

    The TV ball implies per-class bounds. Since entropy is concave, the minimum
    is at an extreme point of the polytope; a greedy heuristic tries each class
    as the primary recipient of excess mass.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower(), credal_set.upper())
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


def _entropy_scores(vertices: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mixture ``p = weights @ vertices``, entropy gradient ``g = -log(p) - 1`` and vertex scores ``vertices @ g``.

    The score of a vertex is the derivative of the entropy in its direction, up to a constant.
    The gradient is finite at exact zeros, where the log is taken of the smallest positive
    float, so a class that is zero in every vertex adds nothing to the scores.

    Args:
        vertices: Vertex probabilities of shape ``(V, C)``.
        weights: Mixture weights of shape ``(V,)``.

    Returns:
        Tuple of the mixture of shape ``(C,)``, the gradient of shape ``(C,)`` and the scores of
        shape ``(V,)``.
    """
    p = jnp.sum(jnp.expand_dims(weights, axis=-1) * vertices, axis=-2)
    gradient = -jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)) - 1.0
    return p, gradient, jnp.sum(vertices * gradient, axis=-1)


def _entropy_line_search(p: jax.Array, direction: jax.Array, max_step: jax.Array) -> jax.Array:
    """Step ``t`` in ``[0, max_step]`` that maximizes ``H(p + t * direction)``.

    The entropy is concave along the line, so its derivative decreases in ``t`` and bisection,
    run to the precision of the dtype, finds where it changes sign. If the entropy still rises
    at ``max_step``, exactly ``max_step`` is returned, so that a mixture weight that reaches
    zero becomes exactly zero.

    Args:
        p: Distribution of shape ``(C,)``.
        direction: Direction of shape ``(C,)`` that sums to zero.
        max_step: Largest step, a scalar.

    Returns:
        A step that does not lower the entropy.
    """
    tiny = jnp.finfo(p.dtype).tiny

    def slope(step: jax.Array) -> jax.Array:
        return -jnp.sum(direction * (jnp.log(jnp.maximum(p + step * direction, tiny)) + 1.0))

    def bisect(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = (low + high) / 2
        rising = slope(middle) > 0
        return jnp.where(rising, middle, low), jnp.where(rising, high, middle)

    iterations = round(-math.log2(jnp.finfo(p.dtype).eps))
    low, _ = jax.lax.fori_loop(0, iterations, bisect, (jnp.zeros_like(max_step), max_step))
    return jnp.where(slope(max_step) >= 0, max_step, low)


def _pairwise_step(vertices: jax.Array, weights: jax.Array) -> jax.Array:
    """Pairwise Frank-Wolfe step: move weight from the worst vertex in use to the best vertex.

    Along the edge from the vertex in use with the smallest derivative to the vertex with the
    largest one, the entropy rises whenever the weights are not optimal. The exact line search
    may add the best vertex or drop the worst one, and the sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(V, C)``.
        weights: Mixture weights of shape ``(V,)``.

    Returns:
        The new weights of shape ``(V,)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    toward = jnp.argmax(scores)
    away = jnp.argmin(jnp.where(weights > 0, scores, jnp.inf))
    direction = jnp.zeros_like(weights).at[toward].add(1.0).at[away].add(-1.0)
    moved = jnp.sum(jnp.expand_dims(direction, axis=-1) * vertices, axis=-2)
    step = _entropy_line_search(p, moved, weights[away])
    return jnp.clip(weights + step * direction, min=0.0)


def _newton_step(vertices: jax.Array, weights: jax.Array) -> jax.Array:
    """Newton step on the weights of the vertices in use, followed by an exact line search.

    The Hessian of the entropy in the weights is ``-vertices diag(1 / p) vertices^T``. The step
    maximizes the second-order model among the vertices in use, keeping the sum of the weights,
    by solving its KKT system. A small ridge keeps the system solvable for repeated or affinely
    dependent vertices. The line search stops at the first weight that reaches zero, and the
    sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(V, C)``.
        weights: Mixture weights of shape ``(V,)``.

    Returns:
        The new weights of shape ``(V,)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    in_use = weights > 0
    mask = in_use.astype(weights.dtype)
    inverse = jnp.where(p > 0, 1.0 / jnp.maximum(p, jnp.finfo(p.dtype).tiny), 0.0)
    curvature = (vertices * inverse) @ vertices.T
    curvature = jnp.where(in_use[:, None] & in_use[None, :], curvature, 0.0)
    ridge = math.sqrt(jnp.finfo(p.dtype).eps) * jnp.trace(curvature) / jnp.sum(mask)
    system = curvature + jnp.diag(jnp.where(in_use, ridge, 1.0))
    zero = jnp.zeros((1, 1), dtype=weights.dtype)
    kkt = jnp.block([[system, mask[:, None]], [mask[None, :], zero]])
    solution = jnp.linalg.solve(kkt, jnp.concatenate([scores * mask, zero[0]]))
    direction = jnp.where(in_use, jnp.nan_to_num(solution[:-1], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    # Remove the rounding error of the solve, so that the weights keep their sum.
    direction = (direction - jnp.sum(direction * mask) / jnp.sum(mask)) * mask
    ratio = jnp.where(direction < 0, weights / -direction, jnp.inf)
    max_step = jnp.min(ratio)
    max_step = jnp.where(jnp.isfinite(max_step), max_step, 0.0)
    moved = jnp.sum(jnp.expand_dims(direction, axis=-1) * vertices, axis=-2)
    step = _entropy_line_search(p, moved, max_step)
    return jnp.clip(jnp.where(ratio <= step, 0.0, weights + step * direction), min=0.0)


def _convex_max_entropy_weights(vertices: jax.Array) -> jax.Array:
    """Entropy-maximizing mixture weights of one convex hull.

    Entropy is concave in the mixture weights, so every stationary point is a maximum. Each
    iteration makes a pairwise Frank-Wolfe step, which can add and drop vertices, and a Newton
    step on the vertices in use, which converges fast once the right vertices are in use. Both
    keep the weights nonnegative and summing to one. The Frank-Wolfe gap
    ``max_v vertices[v] @ g - p @ g`` bounds how far ``H(p)`` is below the maximum, since
    entropy is concave. The search starts from uniform weights and stops once the gap is at
    most 1e-10 in float64 or 1e-5 in other dtypes. It also stops when an iteration leaves the
    weights unchanged, since every later iteration would do the same, and after at most
    ``_CONVEX_ITERS`` iterations. This is the iteration of the torch implementation for a
    single set.

    Args:
        vertices: Vertex probabilities of shape ``(V, C)``.

    Returns:
        Mixture weights of shape ``(V,)``.
    """
    tolerance = 1e-10 if vertices.dtype == jnp.float64 else 1e-5

    def condition(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        iteration, current, previous = state
        p, gradient, scores = _entropy_scores(vertices, current)
        gap = jnp.max(scores) - jnp.sum(p * gradient)
        return (iteration < _CONVEX_ITERS) & (gap > tolerance) & jnp.any(current != previous)

    def step(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        iteration, current, _ = state
        return iteration + 1, _newton_step(vertices, _pairwise_step(vertices, current)), current

    weights = jnp.full(vertices.shape[:1], 1.0 / vertices.shape[0], dtype=vertices.dtype)
    # The previous weights start as nan, which differs from any weights.
    _, weights, _ = jax.lax.while_loop(condition, step, (jnp.array(0), weights, jnp.full_like(weights, jnp.nan)))
    return weights / jnp.sum(weights)


@upper_entropy.register(JaxConvexCredalSet)
def jax_convex_upper_entropy(
    credal_set: JaxConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over ``conv(vertices)``, which may be attained in the interior since
    entropy is concave. Each set is optimized on its own with pairwise Frank-Wolfe and Newton
    steps until a bound certifies that the entropy is within 1e-10 (float64) or 1e-5 (other
    dtypes) of the maximum; see ``_convex_max_entropy_weights``. The torch implementation runs
    the same iteration, so the two agree within that bound. Half-precision sets are optimized
    in float32, and the results are cast back.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    # jnp.linalg cannot solve in half precision, so optimize in at least single precision.
    flat_v = vertices.reshape(-1, n_vertices, n_classes).astype(jnp.promote_types(vertices.dtype, jnp.float32))

    weights = jax.vmap(_convex_max_entropy_weights)(flat_v)
    p = jnp.sum(jnp.expand_dims(weights, axis=-1) * flat_v, axis=-2)
    result = _apply_base(jax_entropy(p).reshape(batch_shape).astype(vertices.dtype), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes).astype(vertices.dtype)
    return result


@lower_entropy.register(JaxConvexCredalSet)
def jax_convex_lower_entropy(
    credal_set: JaxConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex.
    """
    vertices = credal_set.tensor.probabilities
    vertex_entropies = jax_entropy(vertices)
    if return_distribution:
        min_idx = jnp.argmin(vertex_entropies, axis=-1)
        min_h = jnp.min(vertex_entropies, axis=-1)
        gather_idx = jnp.expand_dims(jnp.expand_dims(min_idx, axis=-1), axis=-1)
        best_p = jnp.squeeze(jnp.take_along_axis(vertices, gather_idx, axis=-2), axis=-2)
        return _apply_base(min_h, credal_set.num_classes, base), best_p
    return _apply_base(jnp.min(vertex_entropies, axis=-1), credal_set.num_classes, base)


def _lower_probability(vertices: jax.Array, subset: tuple[int, ...]) -> jax.Array:
    """Lower probability ``P_*(A) = min_v sum_{i in A} v_i``, shape ``(...)``."""
    if not subset:
        return jnp.zeros(vertices.shape[:-2], dtype=vertices.dtype)
    selected = jnp.take(vertices, jnp.asarray(subset), axis=-1)
    return jnp.min(jnp.sum(selected, axis=-1), axis=-1)


def _moebius(vertices: jax.Array, subset: tuple[int, ...]) -> jax.Array:
    """Mobius mass ``m(A)`` via inclusion-exclusion over all subsets of ``A``."""
    result = jnp.zeros(vertices.shape[:-2], dtype=vertices.dtype)
    for r in range(len(subset) + 1):
        sign = (-1) ** (len(subset) - r)
        for b in combinations(list(subset), r):
            result = result + sign * _lower_probability(vertices, b)
    return result


@generalized_hartley.register(JaxConvexCredalSet)
def jax_convex_generalized_hartley(
    credal_set: JaxConvexCredalSet,
    base: LogBase = None,
) -> jax.Array:
    """Compute the generalized Hartley measure of a convex credal set.

    Based on :cite:`abellanDisaggregatedTotal2006`. Computed via the Mobius
    transform of the lower probability function over all subsets of the class
    space.
    """
    vertices = credal_set.tensor.probabilities
    n_classes = credal_set.num_classes
    log_b = None if base is None else math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]
    result = jnp.zeros(vertices.shape[:-2], dtype=vertices.dtype)
    for a in chain.from_iterable(combinations(range(n_classes), r) for r in range(1, n_classes + 1)):
        log_a = math.log(len(a)) / log_b if log_b else math.log(len(a))
        result = result + _moebius(vertices, a) * log_a
    return result


@upper_entropy.register(JaxDirichletLevelSetCredalSet)
def jax_dirichlet_level_set_upper_entropy(
    credal_set: JaxDirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a Dirichlet level set credal set.

    Uses per-class bounds from Monte Carlo sampling, then applies the bisection
    algorithm for ``sum(p) = 1`` constrained entropy maximization.
    """
    p = _max_entropy_distribution(credal_set.lower(), credal_set.upper())
    result = _apply_base(jax_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(JaxDirichletLevelSetCredalSet)
def jax_dirichlet_level_set_lower_entropy(
    credal_set: JaxDirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a Dirichlet level set credal set.

    Uses per-class bounds from Monte Carlo sampling, then applies the greedy
    heuristic for entropy minimization at extreme points.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower(), credal_set.upper())
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result
