"""Coverage and efficiency evaluation metrics.

Three top-level dispatched functions used to evaluate predicted-set
representations such as conformal sets and credal sets:

* :func:`coverage` reports the fraction of samples whose true value is
  contained in the predicted set.
* :func:`efficiency` reports the average size of the predicted set.
* :func:`average_interval_width` reports the mean width of per-class
  probability intervals (a geometric companion to :func:`efficiency` for
  envelope-based credal sets).

Concrete semantics depend on the dispatched type. Conformal sets follow the
classical conformal-prediction definitions (cardinality of a one-hot set,
width of an interval). Credal-set semantics specialize per subtype; see the
implementations in :mod:`probly.evaluation.array` and
:mod:`probly.evaluation.torch`.

Currently registered types
--------------------------
* Conformal: :class:`~probly.representation.conformal_set.array.ArrayOneHotConformalSet`,
  :class:`~probly.representation.conformal_set.array.ArrayIntervalConformalSet`,
  :class:`~probly.representation.conformal_set.torch.TorchOneHotConformalSet`,
  :class:`~probly.representation.conformal_set.torch.TorchIntervalConformalSet`.
* Credal (numpy): all of
  :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`,
  :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet`,
  :class:`~probly.representation.credal_set.array.ArrayConvexCredalSet`,
  :class:`~probly.representation.credal_set.array.ArrayDistanceBasedCredalSet`,
  :class:`~probly.representation.credal_set.array.ArrayProbabilityIntervalsCredalSet`.
* Credal (torch): all of
  :class:`~probly.representation.credal_set.torch.TorchConvexCredalSet`,
  :class:`~probly.representation.credal_set.torch.TorchDistanceBasedCredalSet`,
  :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`,
  :class:`~probly.representation.credal_set.torch.TorchDirichletLevelSetCredalSet`.
  Singleton and Discrete torch counterparts do not yet exist; constructing one
  numpy-side is the supported path for those semantics.
"""

from __future__ import annotations

from flextype import flexdispatch


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
