"""Average precision score test suite."""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import average_precision_score


class APScoreSuite:
    """Test suite for average_precision_score."""

    def test_perfect_classifier(self, array_fn):
        """A perfectly separating classifier achieves an AP of 1.0."""
        y_true = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)
        result = average_precision_score(y_true, y_score)
        assert float(result) == pytest.approx(1.0)

    def test_inverse_classifier(self, array_fn):
        """Worst-case ranking gives AP well below 1."""
        y_true = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = array_fn([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=float)
        result = float(average_precision_score(y_true, y_score))
        assert result < 0.5

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's scalar type."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        result = average_precision_score(y_true, y_score)
        assert isinstance(result, array_type)

    @pytest.mark.parametrize(
        ("y_true", "y_score"),
        [
            pytest.param(
                [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                [0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35],
                id="unbatched",
            ),
            pytest.param(
                [[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]],
                [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
                id="batched",
            ),
        ],
    )
    def test_bounded(self, y_true, y_score, array_fn):
        """Average precision is always in [0, 1]."""
        y_true = array_fn(y_true, dtype=float)
        y_score = array_fn(y_score, dtype=float)
        result = np.asarray(average_precision_score(y_true, y_score))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_better_ranking_higher_ap(self, array_fn):
        """A better ranking produces a higher AP than a worse one."""
        y_true_good = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score_good = array_fn([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)
        y_true_bad = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score_bad = array_fn([0.5, 0.6, 0.4, 0.7, 0.3, 0.9], dtype=float)
        ap_good = float(average_precision_score(y_true_good, y_score_good))
        ap_bad = float(average_precision_score(y_true_bad, y_score_bad))
        assert ap_good > ap_bad

    def test_batched_shape(self, array_fn):
        """2D input produces one score per row."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        result = average_precision_score(y_true, y_score)
        assert result.shape == (2,)

    def test_batched_matches_unbatched(self, array_fn):
        """Batched scores match computing each row independently."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        batched = average_precision_score(y_true, y_score)
        for i in range(batched.shape[0]):
            unbatched = average_precision_score(y_true[i], y_score[i])
            assert float(batched[i]) == pytest.approx(float(unbatched), abs=1e-6)

    def test_higher_rank_equivalence(self, array_fn):
        """Higher-rank inputs are treated as additional leading batch dims.

        The metric's contract is that scores live on the last axis and every other
        axis is a batch dim. Running average_precision_score on a (3, 2, n) input
        must produce the same per-row result as running it on the equivalent flat
        (6, n) input. Since average_precision_score is a composition over
        precision_recall_curve, this test also gives defense in depth against
        leading-dim regressions in that primitive.
        """
        rng = np.random.default_rng(0)
        y_true_flat = rng.integers(0, 2, size=(6, 8)).astype(float)
        y_score_flat = rng.random(size=(6, 8))
        y_true_nested = y_true_flat.reshape(3, 2, 8)
        y_score_nested = y_score_flat.reshape(3, 2, 8)

        flat = np.asarray(
            average_precision_score(array_fn(y_true_flat, dtype=float), array_fn(y_score_flat, dtype=float))
        )
        nested = np.asarray(
            average_precision_score(array_fn(y_true_nested, dtype=float), array_fn(y_score_nested, dtype=float))
        )

        np.testing.assert_allclose(nested.reshape(flat.shape), flat)
