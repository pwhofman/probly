"""Reference test suite comparing probly.metrics against sklearn.metrics.

Every test uses truly random data (no seed). If a test is flaky, it indicates
a real correctness bug that needs fixing.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics as sm

from probly.metrics import (
    accuracy,
    auc,
    average_precision_score,
    false_negative_rate,
    false_positive_rate,
    roc_auc_score,
    roc_curve,
)

# Each test runs 3 times with independent random data to increase confidence.
_ROUNDS = pytest.mark.parametrize("_round", range(3), ids=lambda i: f"round{i}")


class ReferenceSuite:
    """Compare probly.metrics against sklearn on random data across all backends."""

    @_ROUNDS
    def test_auc_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """Auc matches sklearn on FPR/TPR from a random classifier."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=20).astype(float)
        y_score = rng.random(size=20)

        # auc needs monotonic x, so we feed it FPR/TPR from roc_curve
        fpr_sk, tpr_sk, _ = sm.roc_curve(y_true, y_score, drop_intermediate=False)
        expected = sm.auc(fpr_sk, tpr_sk)

        fpr, tpr, _ = roc_curve(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float))
        actual = float(auc(fpr, tpr))

        assert actual == pytest.approx(expected, abs=1e-4)

    @_ROUNDS
    def test_roc_auc_score_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """roc_auc_score matches sklearn on random data."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=20).astype(float)
        y_score = rng.random(size=20)

        expected = sm.roc_auc_score(y_true, y_score)

        actual = float(roc_auc_score(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float)))

        assert actual == pytest.approx(expected, abs=1e-4)

    @_ROUNDS
    def test_average_precision_score_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """average_precision_score matches sklearn on random data."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=20).astype(float)
        y_score = rng.random(size=20)

        expected = sm.average_precision_score(y_true, y_score)

        actual = float(average_precision_score(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float)))

        assert actual == pytest.approx(expected, abs=1e-4)

    @_ROUNDS
    def test_accuracy_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """Accuracy matches sklearn on random multiclass labels and probabilities."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 3, size=20)
        y_pred = rng.integers(0, 3, size=20)
        y_prob = rng.random(size=(20, 3))

        expected_labels = sm.accuracy_score(y_true, y_pred)
        expected_probs = sm.accuracy_score(y_true, y_prob.argmax(axis=-1))

        actual_labels = float(accuracy(array_fn(y_pred), array_fn(y_true)))
        actual_probs = float(accuracy(array_fn(y_prob, dtype=float), array_fn(y_true)))

        assert actual_labels == pytest.approx(expected_labels, abs=1e-4)
        assert actual_probs == pytest.approx(expected_probs, abs=1e-4)

    @_ROUNDS
    def test_error_rates_match_sklearn(self, _round, array_fn):  # noqa: PT019
        """false_positive_rate and false_negative_rate match sklearn's confusion matrix."""
        rng = np.random.default_rng()
        # Prepend one sample of each class so both rates are defined.
        y_true = np.concatenate([[0, 1], rng.integers(0, 2, size=18)])
        y_pred = rng.integers(0, 2, size=20)

        tn, fp, fn, tp = sm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        expected_fpr = fp / (fp + tn)
        expected_fnr = fn / (fn + tp)

        actual_fpr = float(false_positive_rate(array_fn(y_pred), array_fn(y_true)))
        actual_fnr = float(false_negative_rate(array_fn(y_pred), array_fn(y_true)))

        assert actual_fpr == pytest.approx(expected_fpr, abs=1e-4)
        assert actual_fnr == pytest.approx(expected_fnr, abs=1e-4)
