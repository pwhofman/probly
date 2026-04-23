"""Tests for sklearn-specific predictor dispatch behavior."""

from __future__ import annotations

import numpy as np
import pytest

from probly.predictor import LogitClassifier, predict, predict_raw

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator


class DummySklearnLogitPredictor(BaseEstimator):
    """Minimal sklearn-style predictor emitting probability-like outputs via predict."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = np.array([0.8, 0.2], dtype=float)
        return np.repeat(probs[None, :], len(x), axis=0)


def test_sklearn_logit_predictor_outputs_are_treated_as_probabilities() -> None:
    """Sklearn logit predictors should avoid an extra logits-to-probabilities conversion."""
    predictor = DummySklearnLogitPredictor()
    LogitClassifier.register_instance(predictor)

    x = np.zeros((6, 2), dtype=float)
    raw = predict_raw(predictor, x)
    output = predict(predictor, x)

    np.testing.assert_allclose(output.probabilities, raw)
