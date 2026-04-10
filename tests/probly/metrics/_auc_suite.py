"""AUC test suite."""

from __future__ import annotations

import numpy as np
import pytest

from probly.metrics import auc


class AUCSuite:
    """Test suite for auc."""

    @pytest.mark.parametrize(
        ("x", "y", "expected"),
        [
            pytest.param([0.0, 1.0], [1.0, 1.0], 1.0, id="unit_square"),
            pytest.param([0.0, 1.0], [0.0, 1.0], 0.5, id="triangle"),
            pytest.param([0.0, 0.5, 1.0], [0.0, 1.0, 0.0], 0.5, id="tent"),
        ],
    )
    def test_value(self, x, y, expected, array_fn):
        """Trapezoid rule gives the correct area for known shapes."""
        result = auc(array_fn(x), array_fn(y))
        assert float(result) == pytest.approx(expected)

    def test_returns_backend_type(self, array_fn, array_type):
        """Result is an instance of the input backend's scalar type."""
        x = array_fn([0.0, 1.0])
        y = array_fn([0.0, 1.0])
        result = auc(x, y)
        assert isinstance(result, array_type)

    def test_single_point(self, array_fn):
        """A single point has no interval, so the area is zero."""
        x = array_fn([0.0])
        y = array_fn([5.0])
        result = auc(x, y)
        assert float(result) == pytest.approx(0.0)

    def test_negative_y(self, array_fn):
        """Negative y-values produce negative area."""
        x = array_fn([0.0, 1.0])
        y = array_fn([-1.0, -1.0])
        result = auc(x, y)
        assert float(result) == pytest.approx(-1.0)

    def test_non_uniform_spacing(self, array_fn):
        """Non-uniform x-spacing weights intervals correctly."""
        x = array_fn([0.0, 0.1, 1.0])
        y = array_fn([1.0, 1.0, 1.0])
        result = auc(x, y)
        assert float(result) == pytest.approx(1.0)

    def test_batched_shape(self, array_fn):
        """2D input produces one AUC value per row."""
        x = array_fn([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        y = array_fn([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
        result = auc(x, y)
        assert result.shape == (2,)

    def test_batched_matches_unbatched(self, array_fn):
        """Batched AUC matches computing each row independently."""
        x = array_fn([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        y = array_fn([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]])
        batched = auc(x, y)
        for i in range(batched.shape[0]):
            expected = float(auc(x[i], y[i]))
            assert float(batched[i]) == pytest.approx(expected)

    def test_higher_rank_equivalence(self, array_fn):
        """Higher-rank inputs are treated as additional leading batch dims.

        The metric's contract is that curve coordinates live on the last axis and
        every other axis is a batch dim. Running auc on a (3, 2, n) input must
        produce the same per-row result as running it on the equivalent flat
        (6, n) input. This locks in the any-leading-dims contract and guards
        against a regression back to an ``ndim == 2`` special case that hardcodes
        the batch dim.
        """
        rng = np.random.default_rng(0)
        x_flat = np.sort(rng.random(size=(6, 8)), axis=-1)
        y_flat = rng.random(size=(6, 8))
        x_nested = x_flat.reshape(3, 2, 8)
        y_nested = y_flat.reshape(3, 2, 8)

        flat = np.asarray(auc(array_fn(x_flat, dtype=float), array_fn(y_flat, dtype=float)))
        nested = np.asarray(auc(array_fn(x_nested, dtype=float), array_fn(y_nested, dtype=float)))

        np.testing.assert_allclose(nested.reshape(flat.shape), flat)
