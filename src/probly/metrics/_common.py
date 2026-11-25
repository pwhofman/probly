"""Dispatched metric factories for classification evaluation."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Any, Protocol, SupportsFloat, cast, overload

from flextype import flexdispatch
from ndlike.array_like import WeakArrayLike

from probly.representation.conformal_set._common import IntervalConformalSet, OneHotConformalSet
from probly.representation.credal_set._common import (
    CategoricalCredalSet,
    ConvexCredalSet,
    DirichletLevelSetCredalSet,
    DiscreteCredalSet,
    DistanceBasedCredalSet,
    ProbabilityIntervalsCredalSet,
    SingletonCredalSet,
)

if TYPE_CHECKING:
    from probly.lazy_types import NativeArray
    from probly.representation.distribution import CategoricalDistribution
    from probly.representation.representation import Representation


class _MetricScalar(WeakArrayLike, SupportsFloat, Protocol):
    """A scalar backend result that can be converted to a Python float."""


# Default L1 tolerance for coverage. A target counts as covered when some member of
# the credal set lies within this L1 distance of it. Shared by interval coverage and convex_hull_coverage.
CREDAL_COVERAGE_EPSILON: float = 0.0005

type ScoringRule = Callable[[object, object], object]
"""A per-instance scoring rule ``(y_prob, y_true) -> losses``, such as :func:`log_loss_per_instance`."""


@flexdispatch
def auc(x: object, y: object) -> object:
    """Compute area under a curve using the trapezoid rule.

    Args:
        x: x-coordinates of shape ``(..., n)``, monotonic along the last axis.
        y: y-coordinates of shape ``(..., n)``.

    Returns:
        Area under the curve of shape ``(...)``.
    """
    xs, ys = cast("Any", x), cast("Any", y)
    return ((xs[..., 1:] - xs[..., :-1]) * (ys[..., 1:] + ys[..., :-1]) / 2.0).sum(axis=-1)


@flexdispatch
def roc_curve(y_true: object, y_score: object) -> tuple[object, object, object]:
    """Compute receiver operating characteristic (ROC) curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        fpr: False positive rates.
        tpr: True positive rates.
        thresholds: Decreasing score thresholds.
    """
    msg = f"No roc_curve implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


@flexdispatch
def precision_recall_curve(y_true: object, y_score: object) -> tuple[object, object, object]:
    """Compute precision-recall curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores (higher means more likely positive).

    Returns:
        precision: Precision values.
        recall: Recall values.
        thresholds: Decreasing score thresholds.
    """
    msg = f"No precision_recall_curve implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


@flexdispatch
def roc_auc_score(y_true: object, y_score: object) -> object:
    """Compute area under the ROC curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        AUROC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


@flexdispatch
def average_precision_score(y_true: object, y_score: object) -> object:
    """Compute average precision (area under the precision-recall curve).

    Uses the step-function interpolation.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_score: Predicted scores.

    Returns:
        Average precision value.
    """
    precision, recall, _ = cast("Any", precision_recall_curve(y_true, y_score))
    return -((recall[..., 1:] - recall[..., :-1]) * precision[..., :-1]).sum(axis=-1)


@flexdispatch
def accuracy(y_pred: object, y_true: object) -> object:
    """Compute top-1 classification accuracy.

    Args:
        y_pred: Predicted class labels of shape ``(n,)``, class probabilities of shape ``(n, k)``, or a
            categorical distribution; probabilities and distributions are reduced to labels via argmax.
        y_true: Integer ground-truth class labels of shape ``(n,)``.

    Returns:
        The fraction of correct predictions in ``[0, 1]``.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``y_pred``.
        TypeError: If ``y_pred`` is a sample of predictions.
    """
    predicted = cast("Any", _probabilities(y_pred))
    labels = cast("Any", y_true).reshape(-1)
    if predicted.ndim == 2:
        predicted = predicted.argmax(axis=-1)
    elif predicted.ndim != 1:
        msg = f"accuracy expects predictions of shape (n,) or (n, k), got shape {tuple(predicted.shape)}."
        raise ValueError(msg)
    if predicted.shape[0] != labels.shape[0]:
        msg = (
            "accuracy labels must match predictions batch size. "
            f"Got {labels.shape[0]} labels for {predicted.shape[0]} predictions."
        )
        raise ValueError(msg)
    return (predicted == labels).sum() / labels.shape[0]


@flexdispatch
def expected_calibration_error(y_prob: object, y_true: object, *, num_bins: int = 15) -> object:
    """Compute the confidence expected calibration error (ECE) :cite:`guoOnCalibration2017`.

    Predictions are grouped into ``num_bins`` equal-width bins by confidence, i.e. the maximum predicted
    probability, and the ECE is the frequency-weighted mean absolute gap between accuracy and mean confidence
    across bins. In contrast to :func:`classwise_ece`, only the winning class probability enters.

    Args:
        y_prob: Predicted class probabilities of shape ``(n, k)``, or a categorical distribution.
        y_true: Integer ground-truth class labels of shape ``(n,)``.
        num_bins: Number of equal-width confidence bins.

    Returns:
        The expected calibration error in ``[0, 1]``.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``y_prob``.
    """
    msg = f"No expected_calibration_error implementation registered for type {type(y_prob)}"
    raise NotImplementedError(msg)


@flexdispatch
def false_positive_rate(y_pred: object, y_true: object) -> object:
    """Compute the false positive rate ``FP / (FP + TN)`` of binary predictions.

    Args:
        y_pred: Predicted binary labels (0 or 1) of shape ``(n,)``.
        y_true: Ground-truth binary labels (0 or 1) of shape ``(n,)``.

    Returns:
        The fraction of true negatives that are predicted positive, or NaN if there are no negative samples.

    Raises:
        ValueError: If ``y_pred`` and ``y_true`` differ in batch size.
    """
    y = cast("Any", y_true).reshape(-1)
    p = cast("Any", y_pred).reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_positive_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    negatives = (y == 0).sum()
    # 0/0 yields NaN when there are no negative samples; a Python branch would break jax.jit tracing.
    return ((p == 1) & (y == 0)).sum() / negatives


@flexdispatch
def false_negative_rate(y_pred: object, y_true: object) -> object:
    """Compute the false negative rate ``FN / (FN + TP)`` of binary predictions.

    Args:
        y_pred: Predicted binary labels (0 or 1) of shape ``(n,)``.
        y_true: Ground-truth binary labels (0 or 1) of shape ``(n,)``.

    Returns:
        The fraction of true positives that are predicted negative, or NaN if there are no positive samples.

    Raises:
        ValueError: If ``y_pred`` and ``y_true`` differ in batch size.
    """
    y = cast("Any", y_true).reshape(-1)
    p = cast("Any", y_pred).reshape(-1)
    if p.shape[0] != y.shape[0]:
        msg = f"false_negative_rate y_true and y_pred must match in batch size. Got {y.shape[0]} and {p.shape[0]}."
        raise ValueError(msg)
    positives = (y == 1).sum()
    # 0/0 yields NaN when there are no positive samples; a Python branch would break jax.jit tracing.
    return ((p == 0) & (y == 1)).sum() / positives


# --- Predicted-set metrics ----------------------------------------------------
#
# Three top-level dispatched functions used to evaluate predicted-set
# representations such as conformal sets and credal sets:
#
# * :func:`coverage` reports the fraction of samples whose true value is
#   contained in the predicted set.
# * :func:`efficiency` reports the average size of the predicted set.
# * :func:`average_interval_width` reports the mean width of per-class
#   probability intervals (a geometric companion to :func:`efficiency` for
#   envelope-based credal sets).
#
# Concrete semantics depend on the dispatched type. Conformal sets follow the
# classical conformal-prediction definitions (cardinality of a one-hot set,
# width of an interval). Credal-set semantics specialize per subtype. The
# implementations below are registered on the backend-independent set classes
# and written against the array operations NumPy, PyTorch and JAX share, so
# every backend class of a set type is covered and results keep the backend's
# scalar type and gradients.


@flexdispatch
def classwise_ece(y_prob: object, y_true: object, *, num_bins: int = 15) -> object:
    """Compute the classwise expected calibration error (classwise-ECE) :cite:`kullBeyondTemperatureScaling2019`.

    For every class ``j``, the predicted probabilities ``p_j`` are grouped into ``num_bins`` equal-width bins;
    the per-class error is the bin-size-weighted mean absolute difference between the empirical frequency of
    class ``j`` and its mean predicted probability within each bin, and the classwise-ECE averages these errors
    over all classes. In contrast to the confidence-based :func:`expected_calibration_error`, it is also
    sensitive to miscalibration on non-maximal class probabilities.

    Args:
        y_prob: Predicted class probabilities of shape ``(n, k)``, or a categorical distribution.
        y_true: Integer ground-truth class labels of shape ``(n,)``.
        num_bins: Number of equal-width probability bins per class.

    Returns:
        The classwise expected calibration error in ``[0, 1]``.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``y_prob``.
    """
    msg = f"No classwise_ece implementation registered for type {type(y_prob)}"
    raise NotImplementedError(msg)


@flexdispatch
def coverage(y_pred: Representation, y_true: object, **kwargs: object) -> _MetricScalar:
    """Compute the empirical coverage of a predicted set against true labels.

    Coverage is the fraction of samples whose true value lies inside the
    predicted set. Higher is better; a well-calibrated conformal predictor
    targeting confidence ``1 - alpha`` should yield coverage close to
    ``1 - alpha``.

    Per-credal-set semantics:

    * Singleton: degenerates to top-1 accuracy on the single distribution.
    * Discrete: covered iff some vertex distribution puts its argmax on the
      true class. Discrete coverage is always less than or equal to Convex
      coverage on the same vertices: a true class with probability strictly
      between two vertex argmaxes is included by the convex hull's
      interval-dominance set but missed by the vertex-only rule. Use
      Discrete when the modeled object is the finite vertex set (e.g. an
      ensemble); use Convex when the convex hull is the modeled object.
    * Convex / DistanceBased / ProbabilityIntervals / DirichletLevelSet:
      interval-dominance prediction set built from
      ``lower()`` / ``upper()`` envelopes. For distribution-valued targets,
      Convex and ProbabilityIntervals sets count a target as covered when
      its L1 distance to ``{p : lower <= p <= upper, sum(p) = 1}`` is at most
      ``epsilon`` (keyword argument, default ``CREDAL_COVERAGE_EPSILON``),
      the same tolerance :func:`convex_hull_coverage` uses. Note that the DistanceBased
      envelope (``clip(nominal +- radius, 0, 1)``) ignores the simplex
      constraint and is therefore loose; the resulting coverage is
      consistent with the declared envelope but over-conservative relative
      to the tight L1-ball envelope.

    Args:
        y_pred: A predicted-set representation (conformal set or credal set).
            See module docstring for the registered types.
        y_true: Ground-truth values aligned with ``y_pred``. Integer class
            labels for classification, or scalar targets for regression.
        **kwargs: Forwarded to the implementation, e.g. ``epsilon`` for Convex
            and ProbabilityIntervals credal sets with distribution-valued targets.

    Returns:
        The empirical coverage in ``[0, 1]``, as a scalar of the set's backend.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"coverage is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def efficiency(y_pred: Representation) -> _MetricScalar:
    """Compute the average size of a predicted set.

    Efficiency reports how compact the predicted set is on average. Smaller
    is better unless noted otherwise. The exact semantics depend on the
    dispatched type:

    * One-hot conformal sets: average cardinality (number of selected
      classes).
    * Interval conformal sets: average interval width (``upper - lower``).
    * Singleton credal sets: ``1`` by definition (one distribution yields
      one argmax class per sample).
    * Discrete credal sets: average number of distinct argmax classes
      across the vertex set. Differs from the convex hull's interval-dominance
      cardinality (which is always ``>=`` the discrete count).
    * Convex / DistanceBased / ProbabilityIntervals credal sets:
      ``1 - mean(upper - lower)``, one minus the mean width of the per-class
      probability intervals; here larger is better.
    * DirichletLevelSet credal sets: cardinality of the interval-dominance
      prediction set.

    Args:
        y_pred: A predicted-set representation. See module docstring for the
            registered types.

    Returns:
        The efficiency, as a scalar of the set's backend.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"efficiency is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def average_interval_width(y_pred: CategoricalCredalSet) -> _MetricScalar:
    """Compute the mean width of per-class probability intervals.

    A geometric companion to :func:`efficiency` for credal sets that expose a
    meaningful per-class interval (probability-intervals, distance-based, and
    Dirichlet-level-set credal sets). Smaller is better. Defined as the mean
    of ``upper - lower`` over both samples and classes.

    Args:
        y_pred: An envelope-based credal set with ``lower()`` / ``upper()``
            methods.

    Returns:
        The mean per-class interval width, as a scalar of the set's backend.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"average_interval_width is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def convex_hull_coverage[T](
    y_pred: T, y_true: object, *, epsilon: float = CREDAL_COVERAGE_EPSILON, **linprog_kwargs: object
) -> float:
    """Empirical convex-hull coverage for distribution-valued targets.

    For each instance, the LP feasibility test
    ``V^T lambda = t, sum(lambda) = 1, lambda in [0, 1]``
    asks whether the target distribution ``t`` can be expressed as a convex
    combination of the credal set's vertex distributions ``V``. Coverage is
    the fraction of instances where the LP is feasible — i.e. where the
    target lies in the convex hull of the vertices.

    With ``epsilon > 0`` the relaxed slack-variable LP
    ``V^T lambda + s+ - s- = t`` is solved with objective
    ``min sum(s+ + s-)``; coverage counts instances whose optimal L1
    distance from the hull is at most ``epsilon``.

    Differs from :func:`coverage` on credal sets, which tests the per-class
    ``[lower, upper]`` envelope instead of the hull (with the same L1
    tolerance for distribution-valued targets) and uses interval dominance
    for integer class labels.

    Args:
        y_pred: A vertex-based credal-set representation
            (``NumpyConvexCredalSet`` / ``NumpyDiscreteCredalSet`` /
            ``NumpySingletonCredalSet`` / ``TorchConvexCredalSet``).
        y_true: A wrapped categorical distribution per instance (shape
            ``(N, K)``). ``NumpyCategoricalDistribution`` for the numpy
            handlers, ``TorchCategoricalDistribution`` for the torch handler.
        epsilon: L1-distance tolerance for relaxed coverage. ``epsilon=0.0``
            (the default) runs the strict feasibility LP, which is faster
            (no slack variables). ``epsilon > 0`` runs the slack LP.
        **linprog_kwargs: Forwarded to :func:`scipy.optimize.linprog` (e.g.
            ``method="highs"`` or solver tolerances). Passing an unsupported
            kwarg raises ``TypeError``.

    Returns:
        Fraction of instances whose target distribution lies in (or within
        ``epsilon`` of) the credal set's convex hull, as ``np.float64``.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.

    Note:
        Only vertex-based credal sets are registered. For
        ``NumpyProbabilityIntervalsCredalSet`` and
        ``TorchProbabilityIntervalsCredalSet``, use the type's own
        :meth:`contains` method to check whether a target distribution lies
        inside the (axis-aligned) credal set; that is a tighter and cheaper
        test than the LP feasibility check used here. Distance-based credal
        sets do not have explicit vertices and are intentionally not
        registered. ``TorchDirichletLevelSetCredalSet`` is also skipped:
        its envelope is estimated by Monte-Carlo sampling, which makes
        LP-based hull membership stochastic.
    """
    msg = f"convex_hull_coverage is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


def _validate_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon < 0.0:
        msg = f"epsilon must be a non-negative finite float, got {epsilon!r}."
        raise ValueError(msg)


# The predicted-set metrics need four array operations that the backends spell differently.
@flexdispatch
def _asarray_like[A: NativeArray](reference: A, values: object) -> A:
    """Convert ``values`` to an array of the backend, and on the device, of the array ``reference``."""
    msg = f"Predicted-set metrics are not implemented for arrays of type {type(reference).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _label_values[A: NativeArray](values: A, y_true: object) -> A:
    """Return the entry of ``values`` ``(..., K)`` at the class label ``y_true`` ``(...)`` of each instance."""
    msg = f"Predicted-set metrics are not implemented for arrays of type {type(values).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _floating[A: NativeArray](values: A) -> A:
    """Cast an array to a floating dtype of at least single precision."""
    msg = f"Predicted-set metrics are not implemented for arrays of type {type(values).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _machine_epsilon(values: NativeArray) -> float:
    """Return the machine epsilon of the dtype of a floating array."""
    msg = f"Predicted-set metrics are not implemented for arrays of type {type(values).__name__}."
    raise NotImplementedError(msg)


class _ProtectedPayload[T](Protocol):
    def protected_values(self) -> dict[str, T]: ...


@overload
def _payload[T](y_pred: OneHotConformalSet[T] | IntervalConformalSet[T]) -> T: ...


@overload
def _payload[T: CategoricalDistribution](y_pred: DiscreteCredalSet[T]) -> T: ...


def _payload(y_pred: object) -> object:
    """Return the single array or distribution a set wraps."""
    (payload,) = cast("_ProtectedPayload[object]", y_pred).protected_values().values()
    return payload


def _mean(values: NativeArray) -> NativeArray:
    """Average all entries, including boolean and integer arrays in every backend."""
    return values.sum() / math.prod(values.shape)


def _interval_dominance_mask(lower: NativeArray, upper: NativeArray) -> NativeArray:
    """Return the classes whose upper probability reaches the largest lower probability."""
    threshold = _label_values(lower, lower.argmax(-1))
    # Both operands have the same backend, which ty cannot infer from the union.
    return upper >= threshold[..., None]  # ty: ignore[unsupported-operator]


@coverage.register(OneHotConformalSet)
def _onehot_coverage(y_pred: OneHotConformalSet[NativeArray], y_true: object) -> NativeArray:
    """Fraction of one-hot conformal sets that contain the true class."""
    return _mean(_label_values(_payload(y_pred), y_true))


@coverage.register(IntervalConformalSet)
def _interval_coverage(y_pred: IntervalConformalSet[NativeArray], y_true: object) -> NativeArray:
    """Fraction of conformal intervals that contain the target, endpoints included."""
    bounds = _payload(y_pred)
    target = _asarray_like(bounds, y_true)
    return _mean((target >= bounds[..., 0]) & (target <= bounds[..., 1]))  # ty: ignore[unsupported-operator]


@efficiency.register((OneHotConformalSet, IntervalConformalSet))
def _conformal_efficiency(y_pred: OneHotConformalSet | IntervalConformalSet) -> NativeArray:
    """Average size of a conformal set: the cardinality of a one-hot set or the width of an interval."""
    return _mean(y_pred.set_size)


@coverage.register(SingletonCredalSet)
def _singleton_coverage(
    y_pred: SingletonCredalSet[CategoricalDistribution[NativeArray]], y_true: object
) -> NativeArray:
    """Top-1 accuracy of the single distribution in a singleton credal set."""
    predicted = _payload(y_pred).probabilities.argmax(-1)
    return _mean(cast("NativeArray", predicted == _asarray_like(predicted, y_true)))


@efficiency.register(SingletonCredalSet)
def _singleton_efficiency(y_pred: SingletonCredalSet[CategoricalDistribution[NativeArray]]) -> NativeArray:
    """Average number of argmax classes of a singleton credal set, which is one for every instance."""
    one_hot = _one_hot_argmax(_payload(y_pred).probabilities)  # (..., K)
    return _mean(one_hot.sum(-1))


@coverage.register(DiscreteCredalSet)
def _discrete_coverage(y_pred: DiscreteCredalSet[CategoricalDistribution[NativeArray]], y_true: object) -> NativeArray:
    """Fraction of instances whose true class is the argmax of some distribution in a discrete credal set."""
    predicted = _payload(y_pred).probabilities.argmax(-1)  # (..., V)
    target = _asarray_like(predicted, y_true)
    matches = cast("NativeArray", predicted == target[..., None])
    return _mean(matches.sum(-1) > 0)


@efficiency.register(DiscreteCredalSet)
def _discrete_efficiency(y_pred: DiscreteCredalSet[CategoricalDistribution[NativeArray]]) -> NativeArray:
    """Average number of distinct argmax classes among the distributions of a discrete credal set."""
    one_hot = _one_hot_argmax(_payload(y_pred).probabilities)  # (..., V, K)
    return _mean((one_hot.sum(-2) > 0).sum(-1))


@coverage.register(DirichletLevelSetCredalSet)
def _interval_dominance_coverage(y_pred: CategoricalCredalSet, y_true: object) -> NativeArray:
    """Fraction of instances whose true class is selected by interval dominance on the envelope."""
    return _mean(_label_values(_interval_dominance_mask(y_pred.lower(), y_pred.upper()), y_true))


@coverage.register((ConvexCredalSet, ProbabilityIntervalsCredalSet))
def _envelope_coverage(
    y_pred: ConvexCredalSet | ProbabilityIntervalsCredalSet,
    y_true: object,
    *,
    epsilon: float = CREDAL_COVERAGE_EPSILON,
) -> NativeArray:
    """Coverage of the lower/upper envelope of a convex or probability-intervals credal set.

    Class labels of shape ``(...)`` are covered when interval dominance selects them. Target distributions of shape
    ``(..., K)`` are covered when their L1 distance to ``{p : lower <= p <= upper, sum(p) = 1}`` is at most
    ``epsilon``. For a target on the simplex this distance is ``2 * max(a, b)``, with ``a`` the target mass above
    ``upper`` and ``b`` the mass missing below ``lower``, the same L1 distance :func:`convex_hull_coverage` uses.

    Args:
        y_pred: Convex or probability-intervals credal set.
        y_true: Class labels of shape ``(...)`` or target distributions of shape ``(..., K)``.
        epsilon: L1 tolerance for target distributions.

    Returns:
        The fraction of covered instances.

    Raises:
        ValueError: If ``epsilon`` is negative or not finite.
    """
    lower, upper = y_pred.lower(), y_pred.upper()
    target = _asarray_like(lower, y_true)
    if target.ndim < lower.ndim:
        return _mean(_label_values(_interval_dominance_mask(lower, upper), target))
    _validate_epsilon(epsilon)
    lo, up, y = (_floating(array) for array in (lower, upper, target))
    above = (y - up).clip(0.0).sum(-1)  # ty: ignore[unsupported-operator]
    below = (lo - y).clip(0.0).sum(-1)  # ty: ignore[unsupported-operator]
    # Rounding the inputs to their floating precision shifts the distance by at most two machine epsilons, so the
    # slack keeps a distance that equals epsilon on the covered side.
    tolerance = epsilon + 4.0 * _machine_epsilon(lo)
    return _mean((2.0 * above <= tolerance) & (2.0 * below <= tolerance))


@coverage.register(DistanceBasedCredalSet)
def _distance_based_coverage(
    y_pred: DistanceBasedCredalSet[CategoricalDistribution[NativeArray]], y_true: object
) -> NativeArray:
    """Coverage of a distance-based credal set.

    Class labels of shape ``(...)`` are covered when interval dominance on the envelope selects them. Target
    distributions of shape ``(..., K)`` are covered when their total variation distance to the nominal distribution
    is at most the radius.
    """
    nominal = y_pred.nominal.probabilities
    target = _asarray_like(nominal, y_true)
    if target.ndim < nominal.ndim:
        return _interval_dominance_coverage(y_pred, target)
    total_variation = 0.5 * abs(nominal - target).sum(-1)  # ty: ignore[unsupported-operator]
    return _mean(total_variation <= y_pred.radius)


@efficiency.register((ConvexCredalSet, DistanceBasedCredalSet, ProbabilityIntervalsCredalSet))
def _envelope_efficiency(y_pred: CategoricalCredalSet) -> NativeArray:
    """Efficiency of an envelope-based credal set as ``1 - mean(upper - lower)``; higher means a tighter set."""
    return 1.0 - _envelope_average_interval_width(y_pred)


@efficiency.register(DirichletLevelSetCredalSet)
def _interval_dominance_efficiency(y_pred: DirichletLevelSetCredalSet) -> NativeArray:
    """Average number of classes selected by interval dominance on the envelope."""
    return _mean(_interval_dominance_mask(y_pred.lower(), y_pred.upper()).sum(-1))


@average_interval_width.register(CategoricalCredalSet)
def _envelope_average_interval_width(y_pred: CategoricalCredalSet) -> NativeArray:
    """Mean per-class width ``upper - lower`` of the envelope of a credal set."""
    lower, upper = y_pred.lower(), y_pred.upper()
    return _mean(upper - lower)  # ty: ignore[unsupported-operator]


# These primitives provide composed defaults. Individual rules can specialize dispatch where a backend needs
# a more efficient or numerically stable implementation.
@flexdispatch
def _probabilities(y_prob: object) -> object:
    """Return the probabilities ``(..., K)`` of a prediction, unwrapping categorical distributions."""
    msg = f"No probabilities implementation registered for type {type(y_prob).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _log_probabilities(y_prob: object) -> object:
    """Return the log probabilities ``(..., K)`` of a prediction, exact for logit-parameterized distributions."""
    msg = f"No log probabilities implementation registered for type {type(y_prob).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _one_hot_argmax(probabilities: object) -> NativeArray:
    """One-hot encode the first maximum of each probability vector ``(..., K)`` in the probabilities' dtype."""
    msg = f"No one-hot argmax implementation registered for type {type(probabilities).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def _expected_loss(loss_vector: object, y_true: object) -> object:
    """Reduce a per-label loss vector of shape ``(..., K)`` to the loss of each instance under ``y_true``."""
    msg = f"Scoring rules are not implemented for type {type(loss_vector).__name__}."
    raise NotImplementedError(msg)


def _validate_labels(labels_shape: tuple[int, ...], loss_shape: tuple[int, ...], *, is_integer: bool) -> None:
    """Raise unless ``y_true`` holds integer labels for per-label losses of shape ``loss_shape``."""
    if labels_shape != loss_shape[:-1]:
        msg = (
            f"y_true must be integer labels of shape {loss_shape[:-1]} or target probabilities of shape "
            f"{loss_shape}, got shape {labels_shape}."
        )
        raise ValueError(msg)
    if not is_integer:
        msg = "y_true of the predictions' batch shape must hold integer class labels."
        raise ValueError(msg)


@flexdispatch
def log_loss_per_instance(y_prob: object, y_true: object) -> object:
    """Compute the log loss ``-log(p_y)`` of each prediction.

    No clipping is applied, so the loss is ``inf`` where the true class has probability zero.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution, whose log
            probabilities are used directly so that logit-parameterized distributions stay exact.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)`` (array or
            categorical distribution) under which the expected loss is taken.

    Returns:
        The loss of each prediction, of shape ``(...)``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return _expected_loss(-cast("Any", _log_probabilities(y_prob)), y_true)


def brier_loss_per_instance(y_prob: object, y_true: object) -> object:
    """Compute the Brier loss ``sum_k (p_k - [k == y])^2`` of each prediction.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)`` (array or
            categorical distribution) under which the expected loss is taken.

    Returns:
        The loss of each prediction in ``[0, 2]``, of shape ``(...)``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    probabilities = cast("Any", _probabilities(y_prob))
    loss_vector = (probabilities**2).sum(axis=-1, keepdims=True) - 2.0 * probabilities + 1.0
    return _expected_loss(loss_vector, y_true)


def zero_one_loss_per_instance(y_prob: object, y_true: object) -> object:
    """Compute the zero-one loss ``1 - [argmax_k p_k == y]`` of each prediction.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)`` (array or
            categorical distribution) under which the expected loss is taken.

    Returns:
        The loss of each prediction in ``[0, 1]``, of shape ``(...)``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return _expected_loss(1.0 - _one_hot_argmax(_probabilities(y_prob)), y_true)


def spherical_loss_per_instance(y_prob: object, y_true: object) -> object:
    """Compute the spherical loss ``1 - p_y / ||p||_2`` of each prediction.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)`` (array or
            categorical distribution) under which the expected loss is taken.

    Returns:
        The loss of each prediction in ``[0, 1]``, of shape ``(...)``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    probabilities = cast("Any", _probabilities(y_prob))
    norm = (probabilities**2).sum(axis=-1, keepdims=True) ** 0.5
    return _expected_loss(1.0 - probabilities / norm, y_true)


def log_loss(y_prob: object, y_true: object) -> object:
    """Compute the mean log loss of the predictions, the average of :func:`log_loss_per_instance`.

    No clipping is applied, so the mean is ``inf`` if any true class has probability zero.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)``.

    Returns:
        The mean log loss.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return cast("Any", log_loss_per_instance(y_prob, y_true)).mean()


def brier_loss(y_prob: object, y_true: object) -> object:
    """Compute the mean Brier loss of the predictions, the average of :func:`brier_loss_per_instance`.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)``.

    Returns:
        The mean Brier loss in ``[0, 2]``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return cast("Any", brier_loss_per_instance(y_prob, y_true)).mean()


def zero_one_loss(y_prob: object, y_true: object) -> object:
    """Compute the mean zero-one loss of the predictions, the average of :func:`zero_one_loss_per_instance`.

    With integer labels this is the error rate, ``1 - accuracy``.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)``.

    Returns:
        The mean zero-one loss in ``[0, 1]``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return cast("Any", zero_one_loss_per_instance(y_prob, y_true)).mean()


def spherical_loss(y_prob: object, y_true: object) -> object:
    """Compute the mean spherical loss of the predictions, the average of :func:`spherical_loss_per_instance`.

    Args:
        y_prob: Predicted class probabilities of shape ``(..., K)``, or a categorical distribution.
        y_true: Integer class labels of shape ``(...)``, or target probabilities of shape ``(..., K)``.

    Returns:
        The mean spherical loss in ``[0, 1]``.

    Raises:
        ValueError: If ``y_true`` has neither shape or its labels are not integers.
        TypeError: If ``y_prob`` is a sample of predictions.
    """
    return cast("Any", spherical_loss_per_instance(y_prob, y_true)).mean()
