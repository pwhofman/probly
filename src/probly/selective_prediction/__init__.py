"""Selective prediction: predict or abstain based on the uncertainty of an uncertainty-aware model.

A :func:`selective_predictor` wraps a model that probly can represent, e.g. an ensemble, an MC-dropout model, or a
plain classifier declared with :func:`~probly.method.cast`, and accepts or rejects each prediction with a
:class:`~probly.selective_prediction.selectors.Selector`, a rule on the uncertainty criterion. :func:`selector` builds
it from the keywords that name its guarantee:

- ``threshold``: a fixed threshold; on the default criterion, Chow's rule :cite:`chowOptimumRecognition1970`.
- ``coverage``: a target coverage, fitted without labels with a split-conformal guarantee
  :cite:`angelopoulosGentleIntroduction2021`.
- ``risk`` and ``delta``: a bound on the risk of the accepted predictions that holds with probability ``1 - delta``
  :cite:`geifmanSelectiveClassification2017`.

A selector sees only the criterion, so it also works for any other model on a criterion computed by hand.

The guarantees of the fitted selectors are marginal, over the draw of the calibration set and the test instance, and
cover one quantity each: the coverage or the risk. They hold if

- the calibration data is independent of the model and of any recalibration of it, e.g. temperature scaling;
- the criterion is computed in the same way for calibration and test data, as
  ``calibrate`` and ``predict`` of the selective predictor do;
- a random model, e.g. with MC dropout, draws its randomness independently per instance; with a dropout mask shared
  across the batch, they hold only approximately.
"""

from __future__ import annotations

from ._common import SelectivePrediction, SelectivePredictor, selective_predictor
from .selectors import Selector, selector

__all__ = [
    "SelectivePrediction",
    "SelectivePredictor",
    "Selector",
    "selective_predictor",
    "selector",
]
