"""Coverage and efficiency evaluation metrics.

Two top-level dispatched functions used to evaluate predicted-set
representations such as conformal sets and credal sets:

* :func:`coverage` reports the fraction of samples whose true value is
  contained in the predicted set.
* :func:`efficiency` reports the average size of the predicted set.

Concrete semantics depend on the dispatched type. Conformal sets follow the
classical conformal-prediction definitions (cardinality of a one-hot set,
width of an interval). Credal-set semantics specialize per subtype; see the
implementations in :mod:`probly.evaluation.array` and
:mod:`probly.evaluation.torch`.
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

    Args:
        y_pred: A predicted-set representation (conformal set or credal set)
            or a raw array. The exact semantics depend on the dispatched type.
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
    * Categorical credal sets: type-specific. Singletons collapse to ``1``;
      finite-vertex credal sets count distinct argmax classes; envelope-based
      credal sets use the cardinality of the interval-dominance prediction
      set.

    Args:
        y_pred: A predicted-set representation or a raw array.

    Returns:
        The mean set size as a float.

    Raises:
        NotImplementedError: If no implementation is registered for the type
            of ``y_pred``.
    """
    msg = f"efficiency is not implemented for type {type(y_pred).__name__}."
    raise NotImplementedError(msg)
