"""Reference test suite comparing probly.metrics against sklearn.metrics.

Every test uses truly random data (no seed). If a test is flaky, it indicates
a real correctness bug that needs fixing.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.metrics as sm

from probly.metrics import auc, average_precision_score, roc_auc_score, roc_curve

# Each test runs 3 times with independent random data to increase confidence.
_ROUNDS = pytest.mark.parametrize("_round", range(3), ids=lambda i: f"round{i}")


class ReferenceSuite:
    """Compare probly.metrics against sklearn on random data across all backends."""

    @_ROUNDS
    def test_auc_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """Auc matches sklearn on FPR/TPR from a random classifier."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_score = rng.random(size=200)

        # auc needs monotonic x, so we feed it FPR/TPR from roc_curve
        fpr_sk, tpr_sk, _ = sm.roc_curve(y_true, y_score, drop_intermediate=False)
        expected = sm.auc(fpr_sk, tpr_sk)

        fpr, tpr, _ = roc_curve(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float))
        actual = float(auc(fpr, tpr))

        assert actual == pytest.approx(expected, abs=1e-5)

    @_ROUNDS
    def test_roc_auc_score_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """roc_auc_score matches sklearn on random data."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_score = rng.random(size=200)

        expected = sm.roc_auc_score(y_true, y_score)

        actual = float(roc_auc_score(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float)))

        assert actual == pytest.approx(expected, abs=1e-5)

    @_ROUNDS
    def test_average_precision_score_matches_sklearn(self, _round, array_fn):  # noqa: PT019
        """average_precision_score matches sklearn on random data."""
        rng = np.random.default_rng()
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_score = rng.random(size=200)

        expected = sm.average_precision_score(y_true, y_score)

        actual = float(average_precision_score(array_fn(y_true, dtype=float), array_fn(y_score, dtype=float)))

        assert actual == pytest.approx(expected, abs=1e-5)
