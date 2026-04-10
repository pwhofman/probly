"""ROC curve test suite."""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import roc_curve


class RocCurveSuite:
    """Test suite for roc_curve."""

    def test_perfect_classifier(self, array_fn):
        """ROC curve spans from (0,0) to (1,1) for a perfect classifier."""
        y_true = array_fn([0, 0, 0, 1, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        assert float(fpr[0]) == 0.0
        assert float(tpr[0]) == 0.0
        assert float(fpr[-1]) == pytest.approx(1.0)
        assert float(tpr[-1]) == pytest.approx(1.0)

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
    def test_starts_at_origin(self, y_true, y_score, array_fn):
        """FPR and TPR both start at 0."""
        y_true = array_fn(y_true, dtype=float)
        y_score = array_fn(y_score, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        np.testing.assert_array_equal(np.asarray(fpr)[..., 0], 0.0)
        np.testing.assert_array_equal(np.asarray(tpr)[..., 0], 0.0)

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
    def test_ends_at_corner(self, y_true, y_score, array_fn):
        """FPR and TPR both end at 1."""
        y_true = array_fn(y_true, dtype=float)
        y_score = array_fn(y_score, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        np.testing.assert_allclose(np.asarray(fpr)[..., -1], 1.0)
        np.testing.assert_allclose(np.asarray(tpr)[..., -1], 1.0)

    def test_fpr_monotonically_increasing(self, array_fn):
        """FPR values are monotonically non-decreasing."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        fpr, _, _ = roc_curve(y_true, y_score)
        fpr = np.asarray(fpr)
        assert np.all(np.diff(fpr) >= 0)

    def test_tpr_monotonically_increasing(self, array_fn):
        """TPR values are monotonically non-decreasing."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        _, tpr, _ = roc_curve(y_true, y_score)
        tpr = np.asarray(tpr)
        assert np.all(np.diff(tpr) >= 0)

    def test_fpr_tpr_bounded(self, array_fn):
        """FPR and TPR values are in [0, 1]."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr = np.asarray(fpr)
        tpr = np.asarray(tpr)
        assert np.all(fpr >= 0.0)
        assert np.all(fpr <= 1.0)
        assert np.all(tpr >= 0.0)
        assert np.all(tpr <= 1.0)

    def test_unbatched_shape(self, array_fn):
        """1D input produces 1D fpr, tpr, and thresholds."""
        y_true = array_fn([0, 0, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.4, 0.8, 0.9], dtype=float)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        assert fpr.ndim == 1
        assert tpr.ndim == 1
        assert thresholds.ndim == 1

    def test_batched_shape(self, array_fn):
        """2D input of shape (2, n) produces outputs of shape (2, n+1)."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        n = y_true.shape[1]
        assert fpr.shape == (2, n + 1)
        assert tpr.shape == (2, n + 1)
        assert thresholds.shape == (2, n + 1)

    def test_all_negatives(self, array_fn):
        """All-negative labels produce tpr=0 everywhere."""
        y_true = array_fn([0, 0, 0, 0], dtype=float)
        y_score = array_fn([0.1, 0.4, 0.8, 0.9], dtype=float)
        _, tpr, _ = roc_curve(y_true, y_score)
        tpr = np.asarray(tpr)
        np.testing.assert_allclose(tpr, 0.0)

    def test_all_positives(self, array_fn):
        """All-positive labels produce fpr=0 everywhere."""
        y_true = array_fn([1, 1, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.4, 0.8, 0.9], dtype=float)
        fpr, _, _ = roc_curve(y_true, y_score)
        fpr = np.asarray(fpr)
        np.testing.assert_allclose(fpr, 0.0)

    def test_higher_rank_equivalence(self, array_fn):
        """Higher-rank inputs are treated as additional leading batch dims.

        The metric's contract is that scores live on the last axis and every other
        axis is a batch dim. Running roc_curve on a (3, 2, n) input must produce
        the same per-row result as running it on the equivalent flat (6, n) input.
        This locks in the any-leading-dims contract and guards against a regression
        back to an ``ndim == 2`` special case that hardcodes the batch dim.
        """
        rng = np.random.default_rng(0)
        y_true_flat = rng.integers(0, 2, size=(6, 8)).astype(float)
        y_score_flat = rng.random(size=(6, 8))
        y_true_nested = y_true_flat.reshape(3, 2, 8)
        y_score_nested = y_score_flat.reshape(3, 2, 8)

        flat = roc_curve(array_fn(y_true_flat, dtype=float), array_fn(y_score_flat, dtype=float))
        nested = roc_curve(array_fn(y_true_nested, dtype=float), array_fn(y_score_nested, dtype=float))

        for out_flat, out_nested in zip(flat, nested, strict=True):
            flat_arr = np.asarray(out_flat)
            nested_arr = np.asarray(out_nested)
            np.testing.assert_allclose(nested_arr.reshape(flat_arr.shape), flat_arr)
