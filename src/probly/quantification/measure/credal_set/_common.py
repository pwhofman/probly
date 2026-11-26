"""Common definitions of credal set measures."""

from __future__ import annotations

from functools import cache, wraps
import math
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override
import warnings

from flextype import Flexdispatch, flexdispatch
import numpy as np

from probly.representation.credal_set._common import (
    ConvexCredalSet,
    DirichletLevelSetCredalSet,
    DistanceBasedCredalSet,
    ProbabilityIntervalsCredalSet,
)
from probly.utils import elementwise_entropy, entropy, machine_epsilon

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype import LazyType
    from ndlike.array_like import WeakArrayLike

    from probly.lazy_types import NativeArray
    from probly.representation.credal_set._common import CredalSet

type LogBase = float | Literal["normalize"] | None
type Approximate = bool | Literal["auto"]
type _EntropyResult = WeakArrayLike | tuple[WeakArrayLike, WeakArrayLike]

# The generic implementations below combine arrays of one backend with operators and with the array methods that
# NumPy, PyTorch and JAX share. ty cannot check these on the union of the three array types, so their helpers are
# annotated with this alias.
type _Array = Any

# Probability intervals with at most this many classes get the exact lower entropy.
EXACT_LOWER_ENTROPY_MAX_CLASSES = 14

# Bisection steps for the level of the maximum-entropy distribution within per-class bounds.
_BISECTION_STEPS = 64


class _EntropyDispatcher(Flexdispatch[..., _EntropyResult]):
    """A flexdispatcher whose result type depends on ``return_distribution``."""

    # Keep the overloads on the callable object so its inherited registration
    # API remains visible alongside the precise return types.
    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[False] = False,
    ) -> WeakArrayLike: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[True],
    ) -> tuple[WeakArrayLike, WeakArrayLike]: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: bool,
    ) -> _EntropyResult: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> _EntropyResult:
        """Forward arguments unchanged to the registered implementation."""
        return super().__call__(*args, **kwargs)


class _LowerEntropyDispatcher(_EntropyDispatcher):
    """An entropy dispatcher that also takes the ``approximate`` option of :func:`lower_entropy`."""

    @overload
    def register_approx[F: Callable[..., Any]](self, cls: LazyType, func: F) -> F: ...

    @overload
    def register_approx[F: Callable[..., Any]](self, cls: LazyType) -> Callable[[F], F]: ...

    def register_approx(self, cls: LazyType, func: Callable | None = None) -> Callable:
        """Register an approximation-only implementation without an ``approximate`` parameter.

        The registered wrapper rejects ``approximate=False`` and consumes ``True`` or
        ``"auto"`` before calling the implementation.

        Args:
            cls: The dispatch type or lazy type specification.
            func: The approximation. If omitted, return a registration decorator.

        Returns:
            The original function, or a decorator that registers and returns it.
        """
        if func is None:
            return lambda implementation: self.register_approx(cls, implementation)

        @wraps(func)
        def approx(
            credal_set: CredalSet, *args: object, approximate: Approximate = "auto", **kwargs: object
        ) -> _EntropyResult:
            if approximate is False:
                msg = (
                    f"The lower entropy of {type(credal_set).__name__} only has an approximate implementation, "
                    "so approximate=False is not supported. Use approximate=True or approximate='auto'."
                )
                raise ValueError(msg)
            return func(credal_set, *args, **kwargs)

        super().register(cls, approx)
        return func

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[False] = False,
        approximate: Approximate = "auto",
    ) -> WeakArrayLike: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: Literal[True],
        approximate: Approximate = "auto",
    ) -> tuple[WeakArrayLike, WeakArrayLike]: ...

    @overload
    def __call__(
        self,
        credal_set: CredalSet,
        base: LogBase = None,
        *,
        return_distribution: bool,
        approximate: Approximate = "auto",
    ) -> _EntropyResult: ...

    @override
    def __call__(self, credal_set: CredalSet, *args: Any, **kwargs: Any) -> _EntropyResult:
        """Check ``approximate`` and forward the arguments to the registered implementation."""
        approximate = kwargs.setdefault("approximate", "auto")
        if approximate not in (True, False, "auto"):
            msg = f"approximate must be True, False or 'auto', got {approximate!r}."
            raise ValueError(msg)
        return super().__call__(credal_set, *args, **kwargs)


@_EntropyDispatcher
def upper_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> WeakArrayLike | tuple[WeakArrayLike, WeakArrayLike]:
    """Compute the upper entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the maximizer of shape ``(..., num_classes)``.
    """
    msg = f"Upper entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


@_LowerEntropyDispatcher
def lower_entropy(
    credal_set: CredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> WeakArrayLike | tuple[WeakArrayLike, WeakArrayLike]:
    """Compute the lower entropy of a credal set.

    If ``return_distribution`` is ``True``, returns ``(entropy, distribution)``
    where ``distribution`` is the minimizer of shape ``(..., num_classes)``.

    ``approximate=False`` requires an exact computation, while ``True`` permits an
    approximation. The default, ``"auto"``, selects the available strategy. Exact-only
    implementations accept all three values and always compute the exact result.
    Approximation-only implementations reject ``False`` with a ``ValueError``.

    For probability intervals and Dirichlet level sets, the exact lower entropy tries every
    extreme point of the set, which is only feasible for up to 14 classes. ``approximate``
    selects how it is computed:

    - ``False``: exactly. More than 14 classes raise a ``ValueError``.
    - ``True``: with a greedy search, which gives an upper bound on the lower entropy.
    - ``"auto"`` (default): exactly for up to 14 classes, and with the greedy search and a warning for
      more classes.

    Distance-based credal sets have an exact lower entropy for every value of ``approximate``.
    Dirichlet level sets use sampled per-class bounds; their ``approximate`` option
    controls entropy optimization over those bounds, not the sampling approximation.
    """
    msg = f"Lower entropy is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


def use_approximate_lower_entropy(n_classes: int, approximate: Approximate) -> bool:
    """Return whether the lower entropy of probability intervals with ``n_classes`` classes is approximated.

    Args:
        n_classes: Number of classes of the credal set.
        approximate: The ``approximate`` option of :func:`lower_entropy`.

    Returns:
        ``True`` if the greedy search is used, ``False`` if the lower entropy is exact.

    Raises:
        ValueError: If ``approximate`` is ``False`` and there are more than
            ``EXACT_LOWER_ENTROPY_MAX_CLASSES`` classes.
    """
    if approximate is True:
        return True
    if n_classes <= EXACT_LOWER_ENTROPY_MAX_CLASSES:
        return False
    if approximate == "auto":
        warnings.warn(
            f"The lower entropy of probability intervals with {n_classes} classes is approximated by a "
            f"greedy search, since the exact value is only computed for up to "
            f"{EXACT_LOWER_ENTROPY_MAX_CLASSES} classes. The approximation is an upper bound, so the "
            "aleatoric uncertainty can be too high and the epistemic uncertainty too low.",
            stacklevel=2,
        )
        return True
    msg = (
        f"The exact lower entropy of probability intervals is only computed for up to "
        f"{EXACT_LOWER_ENTROPY_MAX_CLASSES} classes, but this credal set has {n_classes}. Pass "
        "approximate=True to use a greedy search, which gives an upper bound on the lower entropy, "
        "or approximate='auto' to use it only when there are too many classes."
    )
    raise ValueError(msg)


@flexdispatch
def generalized_hartley(credal_set: CredalSet, base: LogBase = None) -> WeakArrayLike:
    """Compute the generalized Hartley measure of a credal set."""
    msg = f"Generalized Hartley measure is not supported for credal sets of type {type(credal_set)}."
    raise NotImplementedError(msg)


# The measures below are implemented once for every backend. They need two array operations that the backends
# spell differently.
@flexdispatch
def _take_along_axis[A: NativeArray](values: A, indices: object) -> A:
    """Return the entries of ``values`` ``(..., K)`` at the integer ``indices`` ``(..., M)`` along the last axis."""
    msg = f"Credal set measures are not implemented for arrays of type {type(values).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _constant_like[A: NativeArray](reference: A, values: object) -> A:
    """Create an array of ``values`` with the backend, device and dtype of the array ``reference``."""
    msg = f"Credal set measures are not implemented for arrays of type {type(reference).__name__}."
    raise NotImplementedError(msg)


def _apply_base(result: _Array, n_classes: int, base: LogBase) -> _Array:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)


def _entropy(p: _Array) -> _Array:
    """Compute the Shannon entropy in nats along the last axis."""
    return cast("Any", entropy(p))


def _select(condition: _Array, a: _Array, b: _Array) -> _Array:
    """Choose ``a`` where the boolean ``condition`` holds and ``b`` elsewhere, exactly for finite values."""
    return a * condition + b * ~condition


def _minimum(a: _Array, b: _Array) -> _Array:
    """Return the elementwise minimum of two arrays."""
    return a.clip(None, b)


def _min(values: _Array) -> _Array:
    """Return the minimum along the last axis, which PyTorch returns together with its index."""
    return _take_along_axis(values, values.argmin(-1)[..., None])[..., 0]


def _max(values: _Array) -> _Array:
    """Return the maximum along the last axis, which PyTorch returns together with its index."""
    return _take_along_axis(values, values.argmax(-1)[..., None])[..., 0]


@cache
def _subsets(n_classes: int) -> np.ndarray:
    """Return the indicator rows of all subsets of the classes, of shape ``(2**n_classes, n_classes)``."""
    codes = np.arange(2**n_classes)[:, None]
    return ((codes >> np.arange(n_classes)) & 1).astype(np.float64)


def _box_max_entropy_distribution(lower: _Array, upper: _Array) -> _Array:
    """Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``.

    The KKT conditions give ``p_i = clip(level, lower_i, upper_i)`` with one level per set. The sum of ``p`` grows
    with the level, so bisection finds the level at which it is one.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        The maximum-entropy distributions of shape ``(..., C)``.
    """
    low = lower.sum(-1) * 0.0
    high = low + 1.0
    for _ in range(_BISECTION_STEPS):
        level = (low + high) / 2
        too_small = level[..., None].clip(lower, upper).sum(-1) < 1.0
        low, high = _select(too_small, level, low), _select(too_small, high, level)
    level = (low + high) / 2
    return level[..., None].clip(lower, upper)


def _approximate_min_entropy_distribution(lower: _Array, upper: _Array) -> tuple[_Array, _Array]:
    """Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}`` with a greedy heuristic.

    Since entropy is concave the minimum is at an extreme point of the polytope. The heuristic tries each class as
    the first recipient of the mass above the lower bounds, fills the other classes in their order, and keeps the
    configuration with the lowest entropy.

    Args:
        lower: Lower probability envelope of shape ``(..., C)``.
        upper: Upper probability envelope of shape ``(..., C)``.

    Returns:
        Tuple of ``(entropies, distributions)`` of shapes ``(...)`` and ``(..., C)``.
    """
    n_classes = lower.shape[-1]
    capacity = upper - lower
    residual = (1.0 - lower.sum(-1))[..., None].clip(0.0)
    best = best_p = None
    for first in range(n_classes):
        order = [first, *(i for i in range(n_classes) if i != first)]
        position = [order.index(i) for i in range(n_classes)]
        ordered = capacity[..., order]
        filled_before = ordered.cumsum(-1) - ordered
        fill = _minimum((residual - filled_before).clip(0.0), ordered)
        p = (lower[..., order] + fill)[..., position]
        h = _entropy(p)
        if best is None:
            best, best_p = h, p
            continue
        improved = h < best
        best_p = _select(improved[..., None], p, best_p)
        best = _select(improved, h, best)
    return best, best_p


def _exact_min_entropy_distribution(lower: _Array, upper: _Array) -> tuple[_Array, _Array]:
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
    residual = (1.0 - lower.sum(-1))[..., None]
    tolerance = 8 * n_classes * machine_epsilon(lower)
    lower_terms = elementwise_entropy(lower)
    gain = elementwise_entropy(upper) - lower_terms
    # Candidates that do not fit get a penalty above every entropy, so that they are never selected.
    penalty = math.log(n_classes) + 1.0
    best = lower.sum(-1) * 0.0 + (penalty - 0.5)
    best_p = _select(residual <= 0, lower, upper)
    # The subsets that leave out the last class; swapping two columns makes them leave out the free class instead.
    without_last = _constant_like(lower, _subsets(n_classes)[: 2 ** (n_classes - 1)])
    for free in range(n_classes):
        columns = list(range(n_classes))
        columns[free], columns[-1] = columns[-1], columns[free]
        raised = without_last[:, columns]  # (S, C)
        is_free = _constant_like(lower, [float(i == free) for i in range(n_classes)])
        free_capacity = capacity[..., free : free + 1]
        # Mass left for the free class above its lower bound, for every subset.
        mass = residual - capacity @ raised.T
        fits = (mass >= -tolerance) & (mass <= free_capacity + tolerance)
        mass = _minimum(mass.clip(0.0), free_capacity)
        candidates = (
            lower_terms.sum(-1)[..., None]
            - lower_terms[..., free : free + 1]
            + gain @ raised.T
            + elementwise_entropy(lower[..., free : free + 1] + mass)
        )
        candidates = candidates + ~fits * penalty
        index = candidates.argmin(-1)
        value = _take_along_axis(candidates, index[..., None])[..., 0]
        free_mass = _take_along_axis(mass, index[..., None])
        p = (lower + capacity * raised[index]) * (1.0 - is_free) + (lower + free_mass) * is_free
        improved = value < best
        best_p = _select(improved[..., None], p, best_p)
        best = _select(improved, value, best)
    return _entropy(best_p), best_p


def _min_entropy_distribution(lower: _Array, upper: _Array, approximate: Approximate) -> tuple[_Array, _Array]:
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


@upper_entropy.register((ProbabilityIntervalsCredalSet, DirichletLevelSetCredalSet))
def _envelope_upper_entropy(
    credal_set: ProbabilityIntervalsCredalSet | DirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> NativeArray | tuple[NativeArray, NativeArray]:
    """Compute the upper entropy of a credal set given by per-class bounds.

    Maximize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``. The bounds of a Dirichlet level set are its
    sampled lower and upper probabilities.
    """
    p = _box_max_entropy_distribution(credal_set.lower(), credal_set.upper())
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register((ProbabilityIntervalsCredalSet, DirichletLevelSetCredalSet))
def _envelope_lower_entropy(
    credal_set: ProbabilityIntervalsCredalSet | DirichletLevelSetCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> NativeArray | tuple[NativeArray, NativeArray]:
    """Compute the lower entropy of a credal set given by per-class bounds.

    Minimize entropy over ``{p : lower <= p <= upper, sum(p) = 1}``, exactly over its extreme points or, with
    ``approximate``, with the greedy heuristic (see :func:`lower_entropy`). The bounds of a Dirichlet level set are
    its sampled lower and upper probabilities.
    """
    best, best_p = _min_entropy_distribution(credal_set.lower(), credal_set.upper(), approximate)
    result = _apply_base(best, credal_set.num_classes, base)
    if return_distribution:
        return result, best_p
    return result


def _nominal_and_radius(credal_set: DistanceBasedCredalSet) -> tuple[_Array, _Array]:
    """Return the nominal distributions ``(..., K)`` and the radii ``(..., 1)`` of a distance-based credal set.

    The radius broadcasts as in ``lower()`` and ``upper()``. A negative radius counts as zero and a radius above
    one as one, the largest total variation distance between two distributions.
    """
    nominal: _Array = credal_set.nominal.probabilities
    radius: _Array = credal_set.radius
    if radius.ndim == nominal.ndim - 1:
        radius = radius[..., None]
    radius = radius.clip(0.0, 1.0)
    nominal = nominal + 0.0 * radius
    return nominal, (radius + 0.0 * nominal)[..., :1]


def _total_variation_max_entropy_distribution(nominal: _Array, radius: _Array) -> _Array:
    """Maximize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    The maximum moves mass from the largest classes to the smallest: the classes below a level ``low`` rise to
    ``low`` and the classes above a level ``high`` drop to ``high``, so that the same mass ``m`` is added and removed.
    Moving more mass raises the entropy until the levels meet at the uniform distribution, so
    ``m = min(radius, TV(nominal, uniform))``. With the classes sorted, ``low = min_k (m + S_k) / k`` over the sums
    ``S_k`` of the ``k`` smallest classes and ``high = max_k (T_k - m) / k`` over the sums ``T_k`` of the ``k``
    largest.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Radii in ``[0, 1]`` of shape ``(..., 1)``.

    Returns:
        The maximum-entropy distributions of shape ``(..., K)``.
    """
    n_classes = nominal.shape[-1]
    uniform = nominal.sum(-1)[..., None] / n_classes
    mass = _minimum(radius, (nominal - uniform).clip(0.0).sum(-1)[..., None])
    counts = _constant_like(nominal, list(range(1, n_classes + 1)))
    ascending = _take_along_axis(nominal, nominal.argsort(-1))
    descending = _take_along_axis(nominal, (-nominal).argsort(-1))
    low = _min((mass + ascending.cumsum(-1)) / counts)[..., None]
    high = _max((descending.cumsum(-1) - mass) / counts)[..., None]
    return nominal.clip(low, high)


def _total_variation_min_entropy_distribution(nominal: _Array, radius: _Array) -> _Array:
    """Minimize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}``.

    Entropy is concave, so the minimum is at a vertex of the ball, where one class gains mass and the others lose
    it. Giving the mass to the largest class and taking it from the smallest classes first gives the most
    concentrated distribution in the ball, so this is the minimum. The largest class can take at most the mass of
    the other classes, so the mass moved is ``m = min(radius, 1 - max(nominal))``.

    Args:
        nominal: Nominal distributions of shape ``(..., K)``.
        radius: Radii in ``[0, 1]`` of shape ``(..., 1)``.

    Returns:
        The minimum-entropy distributions of shape ``(..., K)``.
    """
    n_classes = nominal.shape[-1]
    if n_classes == 1:
        return nominal
    classes = _constant_like(nominal, list(range(n_classes)))
    is_largest = classes == nominal.argmax(-1)[..., None]
    # Probabilities are at most one, so adding two sorts the largest class last and the donors before it.
    order = (nominal + is_largest * 2.0).argsort(-1)
    ascending = _take_along_axis(nominal, order)
    cumulative = ascending.cumsum(-1)
    # The largest class can take at most the mass of the donors, computed from the same cumulative sum that empties
    # them, so that emptied classes are exactly zero.
    mass = _minimum(radius, cumulative[..., -2:-1])
    # Empty the smallest donors in turn until the mass is taken.
    kept = _minimum((cumulative - mass).clip(0.0), ascending)
    is_last = _constant_like(nominal, [0.0] * (n_classes - 1) + [1.0])
    moved = kept * (1.0 - is_last) + (ascending + mass) * is_last
    return _take_along_axis(moved, order.argsort(-1))


@upper_entropy.register(DistanceBasedCredalSet)
def _distance_based_upper_entropy(
    credal_set: DistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> NativeArray | tuple[NativeArray, NativeArray]:
    """Compute the upper entropy of a distance-based credal set.

    Maximize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}`` by moving mass from the largest
    classes to the smallest. The result is exact.
    """
    p = _total_variation_max_entropy_distribution(*_nominal_and_radius(credal_set))
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


@lower_entropy.register(DistanceBasedCredalSet)
def _distance_based_lower_entropy(
    credal_set: DistanceBasedCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> NativeArray | tuple[NativeArray, NativeArray]:
    """Compute the lower entropy of a distance-based credal set.

    Minimize entropy over the total variation ball ``{p : TV(p, nominal) <= radius}`` by moving mass into the largest
    class, taken from the smallest classes first. The result is exact for every value of ``approximate``.
    """
    del approximate
    p = _total_variation_min_entropy_distribution(*_nominal_and_radius(credal_set))
    result = _apply_base(_entropy(p), credal_set.num_classes, base)
    if return_distribution:
        return result, p
    return result


def _vertices(credal_set: ConvexCredalSet) -> _Array:
    """Return the vertex probabilities ``(..., V, K)`` of a convex credal set."""
    (vertices,) = cast("Any", credal_set).protected_values().values()
    return vertices.probabilities


@lower_entropy.register(ConvexCredalSet)
def _convex_lower_entropy(
    credal_set: ConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
    approximate: Approximate = "auto",
) -> NativeArray | tuple[NativeArray, NativeArray]:
    """Compute the lower entropy of a convex hull credal set.

    Since entropy is concave, the minimum over a convex hull is at a vertex. The result is exact for every value of
    ``approximate``.
    """
    del approximate
    vertices = _vertices(credal_set)
    vertex_entropies = _entropy(vertices)  # (..., V)
    index = vertex_entropies.argmin(-1)[..., None]
    result = _apply_base(_take_along_axis(vertex_entropies, index)[..., 0], credal_set.num_classes, base)
    if return_distribution:
        is_minimizer = _constant_like(vertices, list(range(vertices.shape[-2]))) == index  # (..., V)
        return result, (vertices * is_minimizer[..., None]).sum(-2)
    return result


@cache
def _generalized_hartley_coefficients(n_classes: int) -> np.ndarray:
    """Return ``c(B) = sum_{A >= B} (-1)^(|A| - |B|) log|A|`` for every subset ``B``, in the order of ``_subsets``."""
    codes = np.arange(2**n_classes)
    coefficients = np.log(np.maximum(_subsets(n_classes).sum(-1), 1.0))
    for i in range(n_classes):
        without = (codes >> i) & 1 == 0
        coefficients[without] -= coefficients[codes[without] | (1 << i)]
    return coefficients


@generalized_hartley.register(ConvexCredalSet)
def _convex_generalized_hartley(credal_set: ConvexCredalSet, base: LogBase = None) -> NativeArray:
    """Compute the generalized Hartley measure of a convex credal set.

    Based on :cite:`abellanDisaggregatedTotal2006`. The measure is the sum of ``m(A) log|A|`` over the Moebius masses
    ``m`` of the lower probability ``P(B) = min_v sum_{i in B} v_i``. The Moebius transform is linear, so the sum
    equals ``sum_B P(B) c(B)`` with coefficients that depend only on the number of classes.
    """
    vertices = _vertices(credal_set)
    n_classes = credal_set.num_classes
    subset_probabilities = vertices @ _constant_like(vertices, _subsets(n_classes)).T  # (..., V, S)
    lower_probabilities = _min(subset_probabilities.swapaxes(-1, -2))  # (..., S)
    result = lower_probabilities @ _constant_like(vertices, _generalized_hartley_coefficients(n_classes))
    return _apply_base(result, n_classes, base)
