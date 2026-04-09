"""Precision-recall curve test suite."""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import precision_recall_curve


class PRCurveSuite:
    """Test suite for precision_recall_curve."""

    def test_recall_decreasing(self, array_fn):
        """Recall values are monotonically non-increasing."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        _, recall, _ = precision_recall_curve(y_true, y_score)
        recall = np.asarray(recall)
        assert np.all(np.diff(recall) <= 0)

    def test_precision_bounded(self, array_fn):
        """All precision values are in [0, 1]."""
        y_true = array_fn([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        y_score = array_fn([0.1, 0.3, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6, 0.85, 0.35], dtype=float)
        precision, _, _ = precision_recall_curve(y_true, y_score)
        precision = np.asarray(precision)
        assert np.all(precision >= 0.0)
        assert np.all(precision <= 1.0)

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
    def test_ends_with_sentinel(self, y_true, y_score, array_fn):
        """Curve ends with the sentinel point recall=0, precision=1."""
        y_true = array_fn(y_true, dtype=float)
        y_score = array_fn(y_score, dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        np.testing.assert_allclose(np.asarray(recall)[..., -1], 0.0)
        np.testing.assert_allclose(np.asarray(precision)[..., -1], 1.0)

    def test_unbatched_shape(self, array_fn):
        """1D input produces 1D precision, recall, and thresholds."""
        y_true = array_fn([0, 0, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.4, 0.8, 0.9], dtype=float)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        assert precision.ndim == 1
        assert recall.ndim == 1
        assert thresholds.ndim == 1

    def test_batched_shape(self, array_fn):
        """2D input produces precision/recall of shape (batch, n+1) and thresholds of shape (batch, n)."""
        y_true = array_fn([[0, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=float)
        y_score = array_fn(
            [[0.1, 0.3, 0.9, 0.8, 0.2, 0.7], [0.6, 0.2, 0.8, 0.3, 0.1, 0.9]],
            dtype=float,
        )
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        n = y_true.shape[1]
        assert precision.shape == (2, n + 1)
        assert recall.shape == (2, n + 1)
        assert thresholds.shape == (2, n)

    def test_perfect_classifier(self, array_fn):
        """Perfect separation achieves precision=1.0 at recall=1.0."""
        y_true = array_fn([0, 0, 1, 1], dtype=float)
        y_score = array_fn([0.1, 0.2, 0.8, 0.9], dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        precision = np.asarray(precision)
        recall = np.asarray(recall)
        # There exists a threshold where recall=1 and precision=1.
        full_recall_mask = recall >= 1.0 - 1e-9
        best_precision_at_full_recall = precision[full_recall_mask].max()
        assert best_precision_at_full_recall == pytest.approx(1.0)

    def test_all_negatives(self, array_fn):
        """All-negative labels produce recall=0 everywhere."""
        y_true = array_fn([0, 0, 0, 0], dtype=float)
        y_score = array_fn([0.1, 0.4, 0.8, 0.9], dtype=float)
        _, recall, _ = precision_recall_curve(y_true, y_score)
        recall = np.asarray(recall)
        np.testing.assert_allclose(recall, 0.0)
