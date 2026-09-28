"""Uncertainty measures for jax-based credal sets."""

from __future__ import annotations

from itertools import chain, combinations
import math

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jax.scipy.special
import numpy as np

from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.utils.jax import jax_entropy

from ._common import (
    Approximate,
    LogBase,
    generalized_hartley,
    lower_entropy,
    upper_entropy,
    use_approximate_lower_entropy,
)

_BISECT_ITERS = 64
_BFGS_ITERS = 128
_FRANK_WOLFE_ITERS = 512
_FRANK_WOLFE_TOL = 1e-5


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


def _approximate_min_entropy_distribution(lower: jax.Array, upper: jax.Array) -> tuple[jax.Array, jax.Array]:
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


def _exact_min_entropy_distribution(lower: jax.Array, upper: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    residual = 1.0 - jnp.sum(lower, axis=-1, keepdims=True)
    # Row s marks the classes of subset s, for all 2**C subsets.
    subsets = (np.arange(2**n_classes)[:, None] >> np.arange(n_classes)) & 1
    tolerance = 8 * n_classes * jnp.finfo(lower.dtype).eps
    lower_terms = jax.scipy.special.entr(lower)
    gain = jax.scipy.special.entr(upper) - lower_terms
    is_class = jnp.arange(n_classes)
    best = jnp.full(lower.shape[:-1], jnp.inf, dtype=lower.dtype)
    best_p = jnp.where(residual <= 0, lower, upper)
    for free in range(n_classes):
        raised = jnp.asarray(subsets[subsets[:, free] == 0], dtype=lower.dtype)
        free_capacity = capacity[..., free : free + 1]
        # Mass left for the free class above its lower bound, for every subset.
        mass = residual - capacity @ raised.T
        fits = (mass >= -tolerance) & (mass <= free_capacity + tolerance)
        mass = jnp.minimum(jnp.clip(mass, min=0.0), free_capacity)
        entropy = (
            jnp.sum(lower_terms, axis=-1, keepdims=True)
            - lower_terms[..., free : free + 1]
            + gain @ raised.T
            + jax.scipy.special.entr(lower[..., free : free + 1] + mass)
        )
        entropy = jnp.where(fits, entropy, jnp.inf)
        index = jnp.argmin(entropy, axis=-1)
        value = jnp.take_along_axis(entropy, index[..., None], axis=-1)[..., 0]
        p = jnp.where(
            is_class == free,
            lower + jnp.take_along_axis(mass, index[..., None], axis=-1),
            lower + capacity * raised[index],
        )
        improved = value < best
        best_p = jnp.where(improved[..., None], p, best_p)
        best = jnp.where(improved, value, best)
    return jax_entropy(best_p), best_p


def _min_entropy_distribution(
    lower: jax.Array, upper: jax.Array, approximate: Approximate
) -> tuple[jax.Array, jax.Array]:
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
    approximate: Approximate = "auto",
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a probability-intervals credal set.

    Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``, exactly or, with
    ``approximate``, with the greedy extreme-point heuristic.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds, approximate)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


def _nominal_and_radius(credal_set: JaxDistanceBasedCredalSet) -> tuple[jax.Array, jax.Array]:
    """Nominal distributions of shape ``(..., K)`` and radii of shape ``(..., 1)`` of a distance-based set.

    The radius broadcasts as in ``lower()`` and ``upper()``, and a negative radius counts as zero.
    """
    nominal = credal_set.nominal.probabilities
    radius = jnp.asarray(credal_set.radius)
    if radius.ndim == nominal.ndim - 1:
        radius = radius[..., None]
    dtype = jnp.result_type(nominal, radius)
    nominal, radius = jnp.broadcast_arrays(nominal.astype(dtype), jnp.clip(radius.astype(dtype), min=0.0))
    return nominal, radius[..., :1]


@jax.jit
def _tv_ball_max_entropy_distribution(nominal: jax.Array, radius: jax.Array) -> jax.Array:
    """Maximize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    The maximum moves mass from the largest classes to the smallest: the classes below a level
    ``low`` rise to ``low`` and the classes above a level ``high`` drop to ``high``, so that the
    same mass ``m`` is added and removed. Moving more mass raises the entropy until the levels
    meet at the uniform distribution, so ``m = min(radius, TV(nominal, uniform))``. With the
    classes sorted, ``low = min_k (m + S_k) / k`` over the sums ``S_k`` of the ``k`` smallest
    classes and ``high = max_k (T_k - m) / k`` over the sums ``T_k`` of the ``k`` largest.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Nonnegative radii of shape ``(..., 1)``.

    Returns:
        The maximum-entropy distributions of shape ``(..., K)``.
    """
    n_classes = nominal.shape[-1]
    uniform = jnp.sum(nominal, axis=-1, keepdims=True) / n_classes
    mass = jnp.minimum(radius, jnp.sum(jnp.clip(nominal - uniform, min=0.0), axis=-1, keepdims=True))
    counts = jnp.arange(1, n_classes + 1, dtype=nominal.dtype)
    ascending = jnp.sort(nominal, axis=-1)
    low = jnp.min((mass + jnp.cumsum(ascending, axis=-1)) / counts, axis=-1, keepdims=True)
    high = jnp.max((jnp.cumsum(ascending[..., ::-1], axis=-1) - mass) / counts, axis=-1, keepdims=True)
    return jnp.clip(nominal, low, high)


@jax.jit
def _tv_ball_min_entropy_distribution(nominal: jax.Array, radius: jax.Array) -> jax.Array:
    """Minimize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    Entropy is concave, so the minimum is at a vertex of the ball, where one class gains mass
    and the others lose it. Giving the mass to the largest class and taking it from the
    smallest classes first gives the most concentrated distribution in the ball, so this is
    the minimum. The largest class can take at most the mass of the other classes, so the
    mass moved is ``m = min(radius, 1 - max(nominal))``.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Nonnegative radii of shape ``(..., 1)``.

    Returns:
        The minimum-entropy distributions of shape ``(..., K)``.
    """
    largest = jnp.argmax(nominal, axis=-1, keepdims=True)
    # Sort the donors, the classes other than the largest, in ascending order; the largest class comes last.
    is_largest = jnp.arange(nominal.shape[-1]) == largest
    order = jnp.argsort(jnp.where(is_largest, jnp.inf, nominal), axis=-1)
    ascending = jnp.take_along_axis(nominal, order, -1)
    donors = ascending[..., :-1]
    cumulative = jnp.cumsum(donors, axis=-1)
    # The largest class can take at most the mass of the donors, computed from the same cumulative sum
    # that empties them, so that emptied classes are exactly zero.
    mass = jnp.minimum(radius, cumulative[..., -1:]) if donors.shape[-1] > 0 else jnp.zeros_like(radius)
    # Empty the smallest donors in turn until the mass is taken.
    kept = jnp.clip(cumulative - mass, 0.0, donors)
    sorted_p = jnp.concatenate([kept, ascending[..., -1:] + mass], axis=-1)
    return jnp.put_along_axis(jnp.zeros_like(nominal), order, sorted_p, axis=-1, inplace=False)


@upper_entropy.register(JaxDistanceBasedCredalSet)
def jax_distance_based_upper_entropy(
    credal_set: JaxDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the upper entropy of a distance-based credal set.

    Maximize entropy over the TV ball ``{p : TV(p, p_hat) <= r}`` by moving mass ``r`` (or
    less, if the uniform distribution is closer) from the largest classes to the smallest. The
    result is exact; see ``_tv_ball_max_entropy_distribution``. A negative radius counts as zero.
    """
    p = _tv_ball_max_entropy_distribution(*_nominal_and_radius(credal_set))
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
    approximate: Approximate = "auto",
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute the lower entropy of a distance-based credal set.

    Minimize entropy over the TV ball ``{p : TV(p, p_hat) <= r}`` by moving mass ``r`` into
    the largest class, taken from the smallest classes first. The result is exact for every
    value of ``approximate``; see ``_tv_ball_min_entropy_distribution``. A negative radius
    counts as zero.
    """
    del approximate
    p = _tv_ball_min_entropy_distribution(*_nominal_and_radius(credal_set))
    result = _apply_base(jax_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


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
    approximate: Approximate = "auto",
) -> jax.Array | tuple[jax.Array, jax.Array]:
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
