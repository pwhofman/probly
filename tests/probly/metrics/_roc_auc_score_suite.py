"""ROC AUC score test suite."""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import auc, roc_auc_score, roc_curve


class RocAucScoreSuite:
    """Test suite for roc_auc_score."""

    @pytest.mark.parametrize(
        ("y_true", "y_score"),
        [
            pytest.param(
                [0, 0, 0, 1, 1, 1],
                [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
                id="unbatched",
            ),
            pytest.param(
                [[0, 0, 1, 1], [0, 0, 1, 1]],
                [[0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 0.8, 0.9]],
                id="batched",
            ),
        ],
    )
    def test_perfect_classifier(self, y_true, y_score, array_fn):
        """A perfectly separating classifier achieves an AUC of 1.0."""
        y_true = array_fn(y_true, dtype=float)
        y_score = array_fn(y_score, dtype=float)
        result = roc_auc_score(y_true, y_score)
        np.testing.assert_allclose(np.asarray(result), 1.0)

    def test_inverse_classifier(self, array_fn):
        """A perfectly inverse classifier scores 0.0."""
        y_true = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = array_fn([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=float)
        result = roc_auc_score(y_true, y_score)
        assert float(result) == pytest.approx(0.0)

    def test_random_classifier(self, array_fn):
        """A random classifier scores approximately 0.5."""
        rng = np.random.default_rng(42)
        y_true = array_fn(rng.integers(0, 2, size=1000).tolist(), dtype=float)
        y_score = array_fn(rng.random(size=1000).tolist(), dtype=float)
        result = roc_auc_score(y_true, y_score)
        assert float(result) == pytest.approx(0.5, abs=0.05)

    def test_uninformative_scores_give_half(self, array_fn):
        """Constant (all-tied) scores give an AUC of exactly 0.5."""
        y_true = array_fn([0, 1, 0, 1, 0, 1], dtype=float)
        y_score = array_fn([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        result = roc_auc_score(y_true, y_score)
        np.testing.assert_allclose(np.asarray(result), 0.5)

    def test_partial_ties_hand_computed(self, array_fn):
        """A tie block spanning both classes adds a diagonal segment; hand-computed AUC is 0.875."""
        y_true = array_fn([0, 1, 0, 1], dtype=float)
        y_score = array_fn([0.5, 0.5, 0.2, 0.8], dtype=float)
        result = roc_auc_score(y_true, y_score)
        np.testing.assert_allclose(np.asarray(result), 0.875)

    @pytest.mark.parametrize("label", [0.0, 1.0], ids=["all-negative", "all-positive"])
    def test_single_class_returns_nan(self, label, array_fn):
        """AUC is undefined for single-class y_true; matches sklearn by warning and returning NaN."""
        y_true = array_fn([label, label, label], dtype=float)
        y_score = array_fn([0.2, 0.5, 0.9], dtype=float)
        with pytest.warns(UserWarning, match="Only one class"):
            result = roc_auc_score(y_true, y_score)
        assert np.isnan(np.asarray(result))

    def test_single_class_batched_nan_per_row(self, array_fn):
        """Batched input: degenerate rows are NaN, well-defined rows keep their value."""
        y_true = array_fn([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=float)
        y_score = array_fn([[0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 0.8, 0.9]], dtype=float)
        with pytest.warns(UserWarning, match="Only one class"):
            result = np.asarray(roc_auc_score(y_true, y_score))
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(1.0)

    def test_bounded(self, array_fn):
        """ROC AUC is always in [0, 1]."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        result = float(roc_auc_score(y_true, y_score))
        assert 0.0 <= result <= 1.0

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's scalar type."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        result = roc_auc_score(y_true, y_score)
        assert isinstance(result, array_type)

    def test_consistent_with_auc_roc_curve(self, array_fn):
        """roc_auc_score matches auc(roc_curve(...))."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        expected = float(auc(fpr, tpr))
        result = float(roc_auc_score(y_true, y_score))
        assert result == pytest.approx(expected)

    def test_batched_shape(self, array_fn):
        """2D input produces one score per row."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        result = roc_auc_score(y_true, y_score)
        assert result.shape == (2,)

    def test_batched_matches_unbatched(self, array_fn):
        """Batched scores match computing each row independently."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        batched = roc_auc_score(y_true, y_score)
        for i in range(batched.shape[0]):
            unbatched = roc_auc_score(y_true[i], y_score[i])
            assert float(batched[i]) == pytest.approx(float(unbatched), abs=1e-6)

    def test_higher_rank_equivalence(self, array_fn):
        """Higher-rank inputs are treated as additional leading batch dims.

        The metric's contract is that scores live on the last axis and every other
        axis is a batch dim. Running roc_auc_score on a (3, 2, n) input must
        produce the same per-row result as running it on the equivalent flat
        (6, n) input. Since roc_auc_score is a composition of roc_curve and auc,
        this test also gives defense in depth against leading-dim regressions in
        either of those primitives.
        """
        rng = np.random.default_rng(0)
        y_true_flat = rng.integers(0, 2, size=(6, 8)).astype(float)
        y_score_flat = rng.random(size=(6, 8))
        y_true_nested = y_true_flat.reshape(3, 2, 8)
        y_score_nested = y_score_flat.reshape(3, 2, 8)

        flat = np.asarray(roc_auc_score(array_fn(y_true_flat, dtype=float), array_fn(y_score_flat, dtype=float)))
        nested = np.asarray(roc_auc_score(array_fn(y_true_nested, dtype=float), array_fn(y_score_nested, dtype=float)))

        np.testing.assert_allclose(nested.reshape(flat.shape), flat)
