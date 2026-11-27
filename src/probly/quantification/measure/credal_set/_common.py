"""Common definitions of credal set measures."""

from __future__ import annotations

from functools import cache, wraps
import math
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override
import warnings

from flextype import Flexdispatch, flexdispatch
import numpy as np

from probly.representation.credal_set._common import (
    DirichletLevelSetCredalSet,
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

    Distance-based credal sets only have an approximate lower entropy implementation.
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


# The measures below are implemented once for every backend. They need three array operations that the backends
# spell differently, each registered with the backend's own function.
@flexdispatch
def _take_along_axis[A: NativeArray](values: A, indices: object) -> A:
    """Return the entries of ``values`` ``(..., K)`` at the integer ``indices`` ``(..., M)`` along the last axis.

    The indices broadcast against the leading axes of ``values``.
    """
    msg = f"Credal set measures are not implemented for arrays of type {type(values).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _constant_like[A: NativeArray](reference: A, values: object) -> A:
    """Create an array of ``values`` with the backend, device and dtype of the array ``reference``."""
    msg = f"Credal set measures are not implemented for arrays of type {type(reference).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _where[A: NativeArray](condition: A, a: object, b: object) -> A:
    """Choose ``a`` where the boolean array ``condition`` holds and ``b`` elsewhere."""
    msg = f"Credal set measures are not implemented for arrays of type {type(condition).__name__}."
    raise NotImplementedError(msg)


def _apply_base(result: _Array, n_classes: int, base: LogBase) -> _Array:
    """Rescale natural-log entropy to the requested log base."""
    if base is None:
        return result
    return result / math.log(n_classes if base == "normalize" else base)


def _entropy(p: _Array) -> _Array:
    """Compute the Shannon entropy in nats along the last axis."""
    return cast("Any", entropy(p))


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
    low, high = _constant_like(lower, 0.0), _constant_like(lower, 1.0)
    for _ in range(_BISECTION_STEPS):
        level = (low + high) / 2
        too_small = level[..., None].clip(lower, upper).sum(-1) < 1.0
        low, high = _where(too_small, level, low), _where(too_small, high, level)
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
    best, best_p = _constant_like(lower, math.inf), lower
    for first in range(n_classes):
        order = [first, *(i for i in range(n_classes) if i != first)]
        position = [order.index(i) for i in range(n_classes)]
        ordered = capacity[..., order]
        filled_before = ordered.cumsum(-1) - ordered
        fill = (residual - filled_before).clip(0.0).clip(None, ordered)
        p = (lower[..., order] + fill)[..., position]
        h = _entropy(p)
        improved = h < best
        best_p = _where(improved[..., None], p, best_p)
        best = _where(improved, h, best)
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
    classes = _constant_like(lower, list(range(n_classes)))
    best = _constant_like(lower, math.inf)
    best_p = _where(residual <= 0, lower, upper)
    # The subsets that leave out the last class; swapping two columns makes them leave out the free class instead.
    without_last = _constant_like(lower, _subsets(n_classes)[: 2 ** (n_classes - 1)])
    for free in range(n_classes):
        columns = list(range(n_classes))
        columns[free], columns[-1] = columns[-1], columns[free]
        raised = without_last[:, columns]  # (S, C)
        free_capacity = capacity[..., free : free + 1]
        # Mass left for the free class above its lower bound, for every subset.
        mass = residual - capacity @ raised.T
        fits = (mass >= -tolerance) & (mass <= free_capacity + tolerance)
        mass = mass.clip(0.0).clip(None, free_capacity)
        candidates = (
            lower_terms.sum(-1)[..., None]
            - lower_terms[..., free : free + 1]
            + gain @ raised.T
            + elementwise_entropy(lower[..., free : free + 1] + mass)
        )
        candidates = _where(fits, candidates, math.inf)
        index = candidates.argmin(-1)
        value = _take_along_axis(candidates, index[..., None])[..., 0]
        free_mass = _take_along_axis(mass, index[..., None])
        p = _where(classes == free, lower + free_mass, lower + capacity * raised[index])
        improved = value < best
        best_p = _where(improved[..., None], p, best_p)
        best = _where(improved, value, best)
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
