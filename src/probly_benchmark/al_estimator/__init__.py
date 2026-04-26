"""Active-learning estimators for the benchmark experiment.

Two estimator classes mirror the two paths the AL experiment takes:

- :class:`BaselineALEstimator` -- ``plain`` and ``ensemble`` trained with
  vanilla cross-entropy. Pairs with the traditional AL strategies
  (``random``, ``margin``, ``badge``). Exposes ``predict_proba`` and
  ``embed``.
- :class:`UQALEstimator` -- any probly UQ method. Trained via probly's
  ``train_model`` flexdispatch. Pairs with ``random`` and ``uncertainty``.
  Exposes ``predict_proba`` and ``uncertainty_scores``.

The driver (:mod:`probly_benchmark.active_learning`) selects between them by
``(method, strategy)``.
"""

from probly_benchmark.al_estimator.baseline import BaselineALEstimator
from probly_benchmark.al_estimator.uq import UQALEstimator

__all__ = ["BaselineALEstimator", "UQALEstimator"]
