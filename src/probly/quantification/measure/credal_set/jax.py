"""Uncertainty measures for jax-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import jax
from jax.core import Tracer
import jax.numpy as jnp
import jax.scipy.optimize

from probly.representation.credal_set.jax import JaxConvexCredalSet
from probly.utils.jax import jax_entropy

from ._common import (
    Approximate,
    LogBase,
    _amax,
    _amin,
    _apply_base,
    _constant_like,
    _take_along_axis,
    _where,
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)

_BFGS_ITERS = 128
_FRANK_WOLFE_ITERS = 512
_FRANK_WOLFE_TOL = 1e-5


@_take_along_axis.register((jax.Array, Tracer))
def _jax_take_along_axis(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather the entries at the indices along the last axis."""
    indices = jnp.broadcast_to(indices, (*values.shape[:-1], indices.shape[-1]))
    return jnp.take_along_axis(values, indices, axis=-1)


@_constant_like.register((jax.Array, Tracer))
def _jax_constant_like(reference: jax.Array, values: object) -> jax.Array:
    """Create a JAX array with the dtype of the reference."""
    return jnp.asarray(values, dtype=reference.dtype)


@_where.register((jax.Array, Tracer))
def _jax_where(condition: jax.Array, a: jax.Array | float, b: jax.Array | float) -> jax.Array:
    """Choose between two arrays by a boolean condition."""
    return jnp.where(condition, a, b)


@_amin.register((jax.Array, Tracer))
def _jax_amin(values: jax.Array) -> jax.Array:
    """Return the minimum along the last axis."""
    return values.min(axis=-1)


@_amax.register((jax.Array, Tracer))
def _jax_amax(values: jax.Array) -> jax.Array:
    """Return the maximum along the last axis."""
    return values.max(axis=-1)


def _convex_entropy_fallback(vertices: jax.Array, weights: jax.Array) -> jax.Array:
    """Improve feasible mixture weights using pairwise Frank-Wolfe steps.

    Entropy is concave in the mixture weights. Transfer mass from the active
    vertex with the smallest derivative to the vertex with the largest one.
    A bounded line search preserves feasibility, and the Frank-Wolfe gap
    bounds the remaining entropy improvement.
    """
    tiny = jnp.finfo(vertices.dtype).tiny

    def condition(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        iteration, _, gap = state
        return (iteration < _FRANK_WOLFE_ITERS) & (gap > _FRANK_WOLFE_TOL)

    def step(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        iteration, current, _ = state
        p = current @ vertices
        gradient = -jnp.log(jnp.maximum(p, tiny)) - 1.0
        derivatives = vertices @ gradient
        index = jnp.argmax(derivatives)
        away = jnp.argmin(jnp.where(current > 0, derivatives, jnp.inf))
        gap = derivatives[index] - p @ gradient
        direction = vertices[index] - vertices[away]

        def bisect(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            low, high = bounds
            mid = (low + high) / 2.0
            derivative = -jnp.sum(direction * (jnp.log(jnp.maximum(p + mid * direction, tiny)) + 1.0))
            return jnp.where(derivative > 0, mid, low), jnp.where(derivative > 0, high, mid)

        low, high = jax.lax.fori_loop(0, 32, bisect, (jnp.array(0.0, p.dtype), current[away]))
        fraction = (low + high) / 2.0
        candidate = current.at[away].add(-fraction).at[index].add(fraction)
        improved = jax_entropy(candidate @ vertices) >= jax_entropy(p)
        return iteration + 1, jnp.where(improved, candidate, current), gap

    _, result, _ = jax.lax.while_loop(condition, step, (jnp.array(0), weights, jnp.array(jnp.inf, vertices.dtype)))
    return result


def _convex_max_entropy_weights(vertices: jax.Array) -> jax.Array:
    """Find entropy-maximizing mixture weights over a single set of vertices.

    Args:
        vertices: Vertex probabilities of shape ``(n_vertices, n_classes)``.

    Returns:
        Nonnegative mixture weights summing to one, shape ``(n_vertices,)``.
    """

    def objective(logits: jax.Array) -> jax.Array:
        p = jnp.sum(jnp.expand_dims(jax.nn.softmax(logits, axis=-1), axis=-1) * vertices, axis=-2)
        return -jax_entropy(p)

    x0 = jnp.zeros(vertices.shape[0], dtype=vertices.dtype)
    result = jax.scipy.optimize.minimize(objective, x0, method="BFGS", options={"maxiter": _BFGS_ITERS})
    uniform = jax.nn.softmax(x0)
    best_vertex = jnp.argmax(jax_entropy(vertices))
    vertex_weights = jax.nn.one_hot(best_vertex, vertices.shape[0], dtype=vertices.dtype)
    baseline = jnp.where(jax_entropy(uniform @ vertices) >= jax_entropy(vertices[best_vertex]), uniform, vertex_weights)
    candidate = jax.nn.softmax(result.x)
    candidate_entropy = jax_entropy(candidate @ vertices)
    usable = jnp.isfinite(candidate).all() & jnp.isfinite(candidate_entropy)
    improved = usable & (candidate_entropy >= jax_entropy(baseline @ vertices))
    best = jnp.where(improved, candidate, baseline)
    # Failed iterates may still be useful starting points, but are not accepted
    # as an optimum. The fallback never replaces a better feasible candidate.
    return jax.lax.cond(result.success & improved, lambda: best, lambda: _convex_entropy_fallback(vertices, best))


@upper_entropy.register(JaxConvexCredalSet)
def jax_convex_upper_entropy(
    credal_set: JaxConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over ``conv(vertices)`` with BFGS on softmax weights,
    falling back to feasible Frank-Wolfe steps if optimization fails.

    Since entropy is concave the maximum over a convex hull may lie in the
    interior; the unconstrained softmax parameterization handles this. This is
    the jax counterpart of the torch L-BFGS implementation, so results agree
    only up to optimizer tolerance.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    flat_v = vertices.reshape(-1, n_vertices, n_classes)

    weights = jax.vmap(_convex_max_entropy_weights)(flat_v)
    p = jnp.sum(jnp.expand_dims(weights, axis=-1) * flat_v, axis=-2)
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
    approximate: Approximate = "auto",
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex.
    The result is exact for every value of ``approximate``.
    """
    del approximate
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
