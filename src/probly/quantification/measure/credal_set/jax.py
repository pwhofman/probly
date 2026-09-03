"""Uncertainty measures for jax-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import jax
import jax.numpy as jnp
import jax.scipy.optimize

from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.utils.jax import jax_entropy

from ._common import LogBase, generalized_hartley, lower_entropy, upper_entropy

_BISECT_ITERS = 64
_BFGS_ITERS = 128


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


def _convex_max_entropy_weights(vertices: jax.Array) -> jax.Array:
    """Find entropy-maximizing softmax weights over a single set of vertices.

    Args:
        vertices: Vertex probabilities of shape ``(n_vertices, n_classes)``.

    Returns:
        Unconstrained softmax logits of shape ``(n_vertices,)``.
    """

    def objective(logits: jax.Array) -> jax.Array:
        p = jnp.sum(jnp.expand_dims(jax.nn.softmax(logits, axis=-1), axis=-1) * vertices, axis=-2)
        return -jax_entropy(p)

    x0 = jnp.zeros(vertices.shape[0], dtype=vertices.dtype)
    result = jax.scipy.optimize.minimize(objective, x0, method="BFGS", options={"maxiter": _BFGS_ITERS})
    return jnp.where(jnp.isfinite(result.x).all(), result.x, x0)


@upper_entropy.register(JaxConvexCredalSet)
def jax_convex_upper_entropy(
    credal_set: JaxConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over ``conv(vertices)`` with BFGS on softmax weights.

    Since entropy is concave the maximum over a convex hull may lie in the
    interior; the unconstrained softmax parameterization handles this. This is
    the jax counterpart of the torch L-BFGS implementation, so results agree
    only up to optimizer tolerance.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    flat_v = vertices.reshape(-1, n_vertices, n_classes)

    w_logits = jax.vmap(_convex_max_entropy_weights)(flat_v)
    p = jnp.sum(jnp.expand_dims(jax.nn.softmax(w_logits, axis=-1), axis=-1) * flat_v, axis=-2)
    result = _apply_base(jax_entropy(p).reshape(batch_shape), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes)
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
