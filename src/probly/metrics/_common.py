"""Dispatched metric factories for classification evaluation."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def auc(x: object, y: object) -> object:
    """Compute area under a curve using the trapezoid rule.

    Args:
        x: x-coordinates (must be monotonic).
        y: y-coordinates.

    Returns:
        Area under the curve.
    """
    msg = f"No auc implementation registered for type {type(x)}"
    raise NotImplementedError(msg)


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
    msg = f"No roc_auc_score implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


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
    msg = f"No average_precision_score implementation registered for type {type(y_true)}"
    raise NotImplementedError(msg)


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
# width of an interval). Credal-set semantics specialize per subtype; see the
# implementations in :mod:`probly.metrics.array` and :mod:`probly.metrics.torch`.
#
# Currently registered types
# --------------------------
# * Conformal: ``ArrayOneHotConformalSet``, ``ArrayIntervalConformalSet``,
#   ``TorchOneHotConformalSet``, ``TorchIntervalConformalSet``.
# * Credal (numpy): ``ArraySingletonCredalSet``, ``ArrayDiscreteCredalSet``,
#   ``ArrayConvexCredalSet``, ``ArrayDistanceBasedCredalSet``,
#   ``ArrayProbabilityIntervalsCredalSet``.
# * Credal (torch): ``TorchConvexCredalSet``, ``TorchDistanceBasedCredalSet``,
#   ``TorchProbabilityIntervalsCredalSet``, ``TorchDirichletLevelSetCredalSet``.
#   Singleton and Discrete torch counterparts do not yet exist; constructing
#   one numpy-side is the supported path for those semantics.


@flexdispatch
def coverage[T](y_pred: T, y_true: object) -> float:
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
      ``lower()`` / ``upper()`` envelopes. Note that the DistanceBased
      envelope (``clip(nominal +- radius, 0, 1)``) ignores the simplex
      constraint and is therefore loose; the resulting coverage is
      consistent with the declared envelope but over-conservative relative
      to the tight L1-ball envelope.

    Args:
        y_pred: A predicted-set representation (conformal set or credal set).
            See module docstring for the registered types.
        y_true: Ground-truth values aligned with ``y_pred``. Integer class
            labels for classification, or scalar targets for regression.

    Returns:
        The empirical coverage as a float in ``[0, 1]``.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"coverage is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def efficiency[T](y_pred: T) -> float:
    """Compute the average size of a predicted set.

    Efficiency reports how compact the predicted set is on average. Smaller
    is better. The exact semantics depend on the dispatched type:

    * One-hot conformal sets: average cardinality (number of selected
      classes).
    * Interval conformal sets: average interval width (``upper - lower``).
    * Singleton credal sets: ``1`` by definition (one distribution yields
      one argmax class per sample).
    * Discrete credal sets: average number of distinct argmax classes
      across the vertex set. Differs from the convex hull's interval-dominance
      cardinality (which is always ``>=`` the discrete count).
    * Convex / DistanceBased / ProbabilityIntervals / DirichletLevelSet
      credal sets: cardinality of the interval-dominance prediction set.

    Args:
        y_pred: A predicted-set representation. See module docstring for the
            registered types.

    Returns:
        The mean set size as a float.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"efficiency is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def average_interval_width[T](y_pred: T) -> float:
    """Compute the mean width of per-class probability intervals.

    A geometric companion to :func:`efficiency` for credal sets that expose a
    meaningful per-class interval (probability-intervals, distance-based, and
    Dirichlet-level-set credal sets). Smaller is better. Defined as the mean
    of ``upper - lower`` over both samples and classes.

    Args:
        y_pred: An envelope-based credal set with ``lower()`` / ``upper()``
            methods.

    Returns:
        The mean per-class interval width as a float.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"average_interval_width is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def convex_hull_coverage[T](y_pred: T, y_true: object, *, epsilon: float = 0.0, **linprog_kwargs: object) -> float:
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

    Differs from :func:`coverage` on credal sets, which takes integer class
    labels and uses interval dominance. This function takes wrapped
    distribution-valued targets and uses exact convex-hull membership.

    Args:
        y_pred: A vertex-based credal-set representation
            (``ArrayConvexCredalSet`` / ``ArrayDiscreteCredalSet`` /
            ``ArraySingletonCredalSet`` / ``TorchConvexCredalSet``).
        y_true: A wrapped categorical distribution per instance (shape
            ``(N, K)``). ``ArrayCategoricalDistribution`` for the numpy
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
        ``ArrayProbabilityIntervalsCredalSet`` and
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
