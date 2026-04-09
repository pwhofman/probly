"""AUC test suite."""

from __future__ import annotations

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
