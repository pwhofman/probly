"""Uncertainty measures for numpy-based credal sets."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import entr

from probly.representation.credal_set.numpy import (
    NumpyConvexCredalSet,
    NumpyDistanceBasedCredalSet,
    NumpyProbabilityIntervalsCredalSet,
)

from ._common import (
    Approximate,
    LogBase,
    generalized_hartley,
    lower_entropy,
    upper_entropy,
    use_approximate_lower_entropy,
)

_BISECT_ITERS = 64
_CONVEX_ITERS = 100


def _apply_base(result: np.ndarray, n_classes: int, base: LogBase) -> np.ndarray:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)  # type: ignore[arg-type]


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis, with ``0 * log(0)`` counted as zero."""
    safe_p = np.where(p > 0, p, 1.0)
    return np.maximum(-np.sum(p * np.log(safe_p), axis=-1), 0.0)


def _max_entropy_distribution(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    The KKT conditions give ``p_i = clip(exp(mu - 1), lower_i, upper_i)`` where ``mu`` is the Lagrange multiplier
    for ``sum(p) = 1``. Since the sum is monotone in ``mu``, bisection finds the unique root.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        The maximum-entropy distribution of shape ``(..., C)``.
    """
    tiny = np.finfo(lower.dtype).tiny
    lo = 1.0 + np.log(np.maximum(np.min(lower, axis=-1), tiny))
    hi = np.ones(lower.shape[:-1], dtype=lower.dtype)
    for _ in range(_BISECT_ITERS):
        mu = (lo + hi) / 2
        g = np.sum(np.clip(np.exp(mu[..., None] - 1), lower, upper), axis=-1) - 1.0
        lo = np.where(g < 0, mu, lo)
        hi = np.where(g >= 0, mu, hi)
    mu = (lo + hi) / 2
    return np.clip(np.exp(mu[..., None] - 1), lower, upper)


def _approximate_min_entropy_distribution(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    Since entropy is concave the minimum is at an extreme point of the polytope. This greedy heuristic tries each
    class as the primary recipient of excess mass and keeps the configuration with the lowest entropy found.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - np.sum(lower, axis=-1)
    best = np.full(lower.shape[:-1], np.inf, dtype=lower.dtype)
    best_p = np.zeros_like(lower)
    for j in range(n_classes):
        p = lower.copy()
        rem = residual
        for i in [j, *[k for k in range(n_classes) if k != j]]:
            fill = np.minimum(np.maximum(rem, 0.0), capacity[..., i])
            p[..., i] += fill
            rem = rem - fill
        h = _entropy(p)
        improved = h < best
        best_p = np.where(improved[..., None], p, best_p)
        best = np.minimum(best, h)
    return best, best_p


def _exact_min_entropy_distribution(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` exactly.

    Since entropy is concave the minimum is at an extreme point of the polytope. At an extreme point, every class but
    one sits at its lower or upper bound, and the remaining class, the free class, takes the rest of the mass. For
    every free class, all subsets of the other classes are tried at their upper bound, and a subset gives an extreme
    point if the rest of the mass fits within the bounds of the free class. The entropy of a candidate is a sum over
    the classes, so the sums over all subsets are one matrix product.

    Bounds that admit no distribution keep ``lower`` if ``sum(lower) > 1`` and ``upper`` if ``sum(upper) < 1``, as
    the greedy heuristic does.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = 1.0 - np.sum(lower, axis=-1, keepdims=True)
    # Row s marks the classes of subset s, for all 2**C subsets.
    subsets = (np.arange(2**n_classes)[:, None] >> np.arange(n_classes)) & 1
    tolerance = 8 * n_classes * np.finfo(lower.dtype).eps
    lower_terms = entr(lower)
    gain = entr(upper) - lower_terms
    is_class = np.arange(n_classes)
    best = np.full(lower.shape[:-1], np.inf, dtype=lower.dtype)
    best_p = np.where(residual <= 0, lower, upper)
    for free in range(n_classes):
        raised = subsets[subsets[:, free] == 0].astype(lower.dtype)
        free_capacity = capacity[..., free : free + 1]
        # Mass left for the free class above its lower bound, for every subset.
        mass = residual - capacity @ raised.T
        fits = (mass >= -tolerance) & (mass <= free_capacity + tolerance)
        mass = np.minimum(np.maximum(mass, 0.0), free_capacity)
        entropy = (
            np.sum(lower_terms, axis=-1, keepdims=True)
            - lower_terms[..., free : free + 1]
            + gain @ raised.T
            + entr(lower[..., free : free + 1] + mass)
        )
        entropy = np.where(fits, entropy, np.inf)
        index = np.argmin(entropy, axis=-1)
        value = np.take_along_axis(entropy, index[..., None], axis=-1)[..., 0]
        p = np.where(
            is_class == free,
            lower + np.take_along_axis(mass, index[..., None], axis=-1),
            lower + capacity * raised[index],
        )
        improved = value < best
        best_p = np.where(improved[..., None], p, best_p)
        best = np.where(improved, value, best)
    return _entropy(best_p), best_p


def _min_entropy_distribution(
    lower: np.ndarray, upper: np.ndarray, approximate: Approximate
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    Exactly, by trying every extreme point, or with the greedy heuristic, as selected by ``approximate`` (see
    :func:`lower_entropy`).

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


@upper_entropy.register(NumpyProbabilityIntervalsCredalSet)
def numpy_intervals_upper_entropy(
    credal_set: NumpyProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the upper entropy of a probability-intervals credal set.

    Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` by bisection on the Lagrange multiplier of the
    sum constraint.
    """
    p = _max_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds)
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(NumpyProbabilityIntervalsCredalSet)
def numpy_intervals_lower_entropy(
    credal_set: NumpyProbabilityIntervalsCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the lower entropy of a probability-intervals credal set.

    Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``, exactly or, with ``approximate``, with the
    greedy extreme-point heuristic.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower_bounds, credal_set.upper_bounds, approximate)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


def _nominal_and_radius(credal_set: NumpyDistanceBasedCredalSet) -> tuple[np.ndarray, np.ndarray]:
    """Nominal distributions of shape ``(..., K)`` and radii of shape ``(..., 1)`` of a distance-based set.

    The radius broadcasts as in ``lower()`` and ``upper()``, and a negative radius counts as zero.
    """
    nominal = credal_set.nominal.probabilities
    radius = np.asarray(credal_set.radius)
    if radius.ndim == nominal.ndim - 1:
        radius = radius[..., None]
    dtype = np.result_type(nominal, radius)
    nominal, radius = np.broadcast_arrays(nominal.astype(dtype), np.maximum(radius.astype(dtype), 0.0))
    return nominal, radius[..., :1]


def _tv_ball_max_entropy_distribution(nominal: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Maximize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    The maximum moves mass from the largest classes to the smallest: the classes below a level ``low`` rise to
    ``low`` and the classes above a level ``high`` drop to ``high``, so that the same mass ``m`` is added and removed.
    Moving more mass raises the entropy until the levels meet at the uniform distribution, so
    ``m = min(radius, TV(nominal, uniform))``. With the classes sorted, ``low = min_k (m + S_k) / k`` over the sums
    ``S_k`` of the ``k`` smallest classes and ``high = max_k (T_k - m) / k`` over the sums ``T_k`` of the ``k``
    largest.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Nonnegative radii of shape ``(..., 1)``.

    Returns:
        The maximum-entropy distributions of shape ``(..., K)``.
    """
    n_classes = nominal.shape[-1]
    uniform = np.sum(nominal, axis=-1, keepdims=True) / n_classes
    mass = np.minimum(radius, np.sum(np.maximum(nominal - uniform, 0.0), axis=-1, keepdims=True))
    counts = np.arange(1, n_classes + 1, dtype=nominal.dtype)
    ascending = np.sort(nominal, axis=-1)
    low = np.min((mass + np.cumsum(ascending, axis=-1)) / counts, axis=-1, keepdims=True)
    high = np.max((np.cumsum(ascending[..., ::-1], axis=-1) - mass) / counts, axis=-1, keepdims=True)
    return np.clip(nominal, low, high)


def _tv_ball_min_entropy_distribution(nominal: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Minimize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    Entropy is concave, so the minimum is at a vertex of the ball, where one class gains mass and the others lose
    it. Giving the mass to the largest class and taking it from the smallest classes first gives the most
    concentrated distribution in the ball, so this is the minimum. The largest class can take at most the mass of
    the other classes, so the mass moved is ``m = min(radius, 1 - max(nominal))``.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Nonnegative radii of shape ``(..., 1)``.

    Returns:
        The minimum-entropy distributions of shape ``(..., K)``.
    """
    largest = np.argmax(nominal, axis=-1, keepdims=True)
    # Sort the donors, the classes other than the largest, in ascending order; the largest class comes last.
    is_largest = np.arange(nominal.shape[-1]) == largest
    order = np.argsort(np.where(is_largest, np.inf, nominal), axis=-1)
    ascending = np.take_along_axis(nominal, order, axis=-1)
    donors = ascending[..., :-1]
    cumulative = np.cumsum(donors, axis=-1)
    # The largest class can take at most the mass of the donors, computed from the same cumulative sum that empties
    # them, so that emptied classes are exactly zero.
    mass = np.minimum(radius, cumulative[..., -1:]) if donors.shape[-1] > 0 else np.zeros_like(radius)
    # Empty the smallest donors in turn until the mass is taken.
    kept = np.clip(cumulative - mass, 0.0, donors)
    p = np.empty_like(nominal)
    np.put_along_axis(p, order, np.concatenate([kept, ascending[..., -1:] + mass], axis=-1), axis=-1)
    return p


@upper_entropy.register(NumpyDistanceBasedCredalSet)
def numpy_distance_based_upper_entropy(
    credal_set: NumpyDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the upper entropy of a distance-based credal set.

    Maximize entropy over the TV ball ``{p : TV(p, p_hat) <= r}`` by moving mass ``r`` (or less, if the uniform
    distribution is closer) from the largest classes to the smallest. The result is exact; see
    ``_tv_ball_max_entropy_distribution``. A negative radius counts as zero.
    """
    p = _tv_ball_max_entropy_distribution(*_nominal_and_radius(credal_set))
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(NumpyDistanceBasedCredalSet)
def numpy_distance_based_lower_entropy(
    credal_set: NumpyDistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the lower entropy of a distance-based credal set.

    Minimize entropy over the TV ball ``{p : TV(p, p_hat) <= r}`` by moving mass ``r`` into the largest class, taken
    from the smallest classes first. The result is exact for every value of ``approximate``; see
    ``_tv_ball_min_entropy_distribution``. A negative radius counts as zero.
    """
    del approximate
    p = _tv_ball_min_entropy_distribution(*_nominal_and_radius(credal_set))
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


def _entropy_scores(vertices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixture ``p = weights @ vertices``, entropy gradient ``g = -log(p) - 1`` and vertex scores ``vertices @ g``.

    The score of a vertex is the derivative of the entropy in its direction, up to a constant. The gradient is finite
    at exact zeros, where the log is taken of the smallest positive float, so a class that is zero in every vertex
    adds nothing to the scores.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        Tuple of the mixtures of shape ``(N, K)``, the gradients of shape ``(N, K)`` and the scores of shape
        ``(N, V)``.
    """
    p = np.sum(weights[..., None] * vertices, axis=-2)
    gradient = -np.log(np.maximum(p, np.finfo(p.dtype).tiny)) - 1.0
    return p, gradient, np.sum(vertices * gradient[..., None, :], axis=-1)


def _entropy_line_search(p: np.ndarray, direction: np.ndarray, max_step: np.ndarray) -> np.ndarray:
    """Step ``t`` in ``[0, max_step]`` that maximizes ``H(p + t * direction)``.

    The entropy is concave along the line, so its derivative decreases in ``t`` and bisection, run to the precision
    of the dtype, finds where it changes sign. If the entropy still rises at ``max_step``, exactly ``max_step`` is
    returned, so that a mixture weight that reaches zero becomes exactly zero.

    Args:
        p: Distributions of shape ``(N, K)``.
        direction: Directions of shape ``(N, K)`` that sum to zero.
        max_step: Largest steps of shape ``(N,)``.

    Returns:
        Steps of shape ``(N,)`` that do not lower the entropy.
    """
    tiny = np.finfo(p.dtype).tiny

    def slope(step: np.ndarray) -> np.ndarray:
        return -np.sum(direction * (np.log(np.maximum(p + step[..., None] * direction, tiny)) + 1.0), axis=-1)

    low, high = np.zeros_like(max_step), max_step
    for _ in range(round(-math.log2(np.finfo(p.dtype).eps))):
        middle = (low + high) / 2
        rising = slope(middle) > 0
        low, high = np.where(rising, middle, low), np.where(rising, high, middle)
    return np.where(slope(max_step) >= 0, max_step, low)


def _pairwise_step(vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pairwise Frank-Wolfe step: move weight from the worst vertex in use to the best vertex.

    Along the edge from the vertex in use with the smallest derivative to the vertex with the largest one, the
    entropy rises whenever the weights are not optimal. The exact line search may add the best vertex or drop the
    worst one, and the sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        The new weights of shape ``(N, V)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    rows = np.arange(weights.shape[0])
    toward = np.argmax(scores, axis=-1)
    away = np.argmin(np.where(weights > 0, scores, np.inf), axis=-1)
    direction = np.zeros_like(weights)
    direction[rows, toward] += 1.0
    direction[rows, away] -= 1.0
    step = _entropy_line_search(p, np.sum(direction[..., None] * vertices, axis=-2), weights[rows, away])
    return np.maximum(weights + step[..., None] * direction, 0.0)


def _newton_step(vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Newton step on the weights of the vertices in use, followed by an exact line search.

    The Hessian of the entropy in the weights is ``-vertices diag(1 / p) vertices^T``. The step maximizes the
    second-order model among the vertices in use, keeping the sum of the weights, by solving its KKT system. A small
    ridge keeps the system solvable for repeated or affinely dependent vertices. The line search stops at the first
    weight that reaches zero, and the sum of the weights is kept.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.
        weights: Mixture weights of shape ``(N, V)``.

    Returns:
        The new weights of shape ``(N, V)``.
    """
    p, _, scores = _entropy_scores(vertices, weights)
    n_sets, n_vertices = weights.shape
    in_use = weights > 0
    mask = in_use.astype(weights.dtype)
    inverse = np.where(p > 0, 1.0 / np.maximum(p, np.finfo(p.dtype).tiny), 0.0)
    curvature = (vertices * inverse[..., None, :]) @ np.swapaxes(vertices, -1, -2)
    curvature = np.where(in_use[..., :, None] & in_use[..., None, :], curvature, 0.0)
    diagonal = np.arange(n_vertices)
    ridge = math.sqrt(np.finfo(p.dtype).eps) * np.trace(curvature, axis1=-2, axis2=-1) / np.sum(mask, axis=-1)
    kkt = np.zeros((n_sets, n_vertices + 1, n_vertices + 1), dtype=weights.dtype)
    kkt[:, :-1, :-1] = curvature
    kkt[:, diagonal, diagonal] += np.where(in_use, ridge[..., None], 1.0)
    kkt[:, :-1, -1] = mask
    kkt[:, -1, :-1] = mask
    rhs = np.concatenate([scores * mask, np.zeros((n_sets, 1), dtype=weights.dtype)], axis=-1)
    solution = np.linalg.solve(kkt, rhs[..., None])[..., 0]
    direction = np.where(in_use, np.nan_to_num(solution[..., :-1], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    # Remove the rounding error of the solve, so that the weights keep their sum.
    direction = (
        direction - np.sum(direction * mask, axis=-1, keepdims=True) / np.sum(mask, axis=-1, keepdims=True)
    ) * mask
    ratio = np.divide(weights, -direction, out=np.full_like(weights, np.inf), where=direction < 0)
    max_step = np.min(ratio, axis=-1)
    max_step = np.where(np.isfinite(max_step), max_step, 0.0)
    step = _entropy_line_search(p, np.sum(direction[..., None] * vertices, axis=-2), max_step)[..., None]
    return np.maximum(np.where(ratio <= step, 0.0, weights + step * direction), 0.0)


def _convex_max_entropy_weights(vertices: np.ndarray) -> np.ndarray:
    """Entropy-maximizing mixture weights of convex hulls, each hull optimized on its own.

    Entropy is concave in the mixture weights, so every stationary point is a maximum. Each iteration makes a
    pairwise Frank-Wolfe step, which can add and drop vertices, and a Newton step on the vertices in use, which
    converges fast once the right vertices are in use. Both keep the weights nonnegative and summing to one. The
    Frank-Wolfe gap ``max_v vertices[v] @ g - p @ g`` bounds how far ``H(p)`` is below the maximum, since entropy is
    concave. A set stops once its gap is at most 1e-10 in float64 or 1e-5 in other dtypes. It also stops when an
    iteration leaves its weights unchanged, since every later iteration would do the same, and after at most
    ``_CONVEX_ITERS`` iterations. Every step only uses the set's own vertices, so the result does not depend on the
    other sets in the batch.

    Args:
        vertices: Vertex probabilities of shape ``(N, V, K)``.

    Returns:
        Mixture weights of shape ``(N, V)``.
    """
    tolerance = 1e-10 if vertices.dtype == np.float64 else 1e-5
    weights = np.full(vertices.shape[:-1], 1.0 / vertices.shape[-2], dtype=vertices.dtype)
    active = np.arange(vertices.shape[0])
    for _ in range(_CONVEX_ITERS):
        if active.size == 0:
            break
        current, points = weights[active], vertices[active]
        p, gradient, scores = _entropy_scores(points, current)
        gap = np.max(scores, axis=-1) - np.sum(p * gradient, axis=-1)
        converged = gap <= tolerance
        updated = _newton_step(points, _pairwise_step(points, current))
        weights[active] = np.where(converged[..., None], current, updated)
        active = active[~converged & np.any(updated != current, axis=-1)]
    return weights / np.sum(weights, axis=-1, keepdims=True)


@upper_entropy.register(NumpyConvexCredalSet)
def numpy_convex_upper_entropy(
    credal_set: NumpyConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over ``conv(vertices)``, which may be attained in the interior since entropy is concave. Each
    set is optimized on its own until a bound certifies that it is within 1e-10 (float64) or 1e-5 (other dtypes) of
    the maximum; see ``_convex_max_entropy_weights``. Half-precision sets are optimized in float32, and the results
    are cast back.
    """
    vertices = credal_set.array.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    # np.linalg cannot solve in half precision, so optimize in at least single precision.
    flat_v = vertices.reshape(-1, n_vertices, n_classes).astype(np.promote_types(vertices.dtype, np.float32))

    p = np.sum(_convex_max_entropy_weights(flat_v)[..., None] * flat_v, axis=-2)
    result = _apply_base(_entropy(p).reshape(batch_shape).astype(vertices.dtype), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes).astype(vertices.dtype)
    return result


@lower_entropy.register(NumpyConvexCredalSet)
def numpy_convex_lower_entropy(
    credal_set: NumpyConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is always at a vertex. The result is exact for every
    value of ``approximate``.
    """
    del approximate
    vertices = credal_set.array.probabilities  # (..., n_vertices, n_classes)
    vertex_entropies = _entropy(vertices)  # (..., n_vertices)
    min_idx = np.argmin(vertex_entropies, axis=-1)
    min_h = np.take_along_axis(vertex_entropies, min_idx[..., None], axis=-1)[..., 0]
    result = _apply_base(min_h, credal_set.num_classes, base)
    if return_distribution:
        best_p = np.take_along_axis(vertices, min_idx[..., None, None], axis=-2)[..., 0, :]
        return result, best_p
    return result


@generalized_hartley.register(NumpyConvexCredalSet)
def numpy_convex_generalized_hartley(
    credal_set: NumpyConvexCredalSet,
    base: LogBase = None,
) -> np.ndarray:
    """Compute the generalized Hartley measure of a convex credal set.

    Based on :cite:`abellanDisaggregatedTotal2006`. The Mobius masses of the lower probability function follow from
    its values on all subsets of the class space by inverting the sum over subsets one class at a time.
    """
    vertices = credal_set.array.probabilities  # (..., n_vertices, n_classes)
    n_classes = credal_set.num_classes
    codes = np.arange(2**n_classes)
    # Row s marks the classes of subset s, for all 2**C subsets.
    subsets = (codes[:, None] >> np.arange(n_classes)) & 1
    # Lower probability P_*(A) = min_v sum_{i in A} v_i, shape (..., 2**C).
    moebius = np.stack([np.min(vertices @ s.astype(vertices.dtype), axis=-1) for s in subsets], axis=-1)
    for i in range(n_classes):
        contains = subsets[:, i] == 1
        moebius[..., contains] -= moebius[..., codes[contains] ^ (1 << i)]
    # The empty set has zero mass, so its log size does not matter.
    log_sizes = np.log(np.maximum(np.sum(subsets, axis=-1), 1)).astype(vertices.dtype)
    return _apply_base(moebius @ log_sizes, n_classes, base)
