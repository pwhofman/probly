"""Tests for the SciPy/NumPy implementation of efficient credal prediction calibration."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
import numpy as np

from probly.method.efficient_credal_prediction import compute_efficient_credal_prediction_bounds


class TestNumpyCredalBounds:
    """Tests the basic properties of the SciPy/NumPy SLSQP solver."""

    @pytest.fixture
    def dummy_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        n_samples, n_classes = 100, 5
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((n_samples, n_classes))
        targets = rng.integers(0, n_classes, size=n_samples)
        return logits, targets, n_classes

    def test_bounds_shape_and_type(self, dummy_data: tuple[np.ndarray, np.ndarray, int]) -> None:
        """Ensure the outputs are float64 arrays of length C."""
        logits, targets, num_classes = dummy_data
        lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes=num_classes, alpha=0.5)

        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (num_classes,)
        assert upper.shape == (num_classes,)
        assert lower.dtype == np.float64
        assert upper.dtype == np.float64

    def test_bounds_sign_invariants(self, dummy_data: tuple[np.ndarray, np.ndarray, int]) -> None:
        """The unperturbed logits always satisfy alpha=0; so lower <= 0 <= upper."""
        logits, targets, num_classes = dummy_data
        lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes=num_classes, alpha=0.5)

        assert np.all(lower <= 0.0), "Lower bounds must be <= 0"
        assert np.all(upper >= 0.0), "Upper bounds must be >= 0"

    def test_alpha_monotonicity(self, dummy_data: tuple[np.ndarray, np.ndarray, int]) -> None:
        """Smaller alpha implies a looser constraint, meaning wider bounds."""
        logits, targets, num_classes = dummy_data

        lower_loose, upper_loose = compute_efficient_credal_prediction_bounds(logits, targets, num_classes, 0.1)
        lower_strict, upper_strict = compute_efficient_credal_prediction_bounds(logits, targets, num_classes, 0.9)

        assert np.all(lower_loose <= lower_strict + 1e-5)
        assert np.all(upper_loose >= upper_strict - 1e-5)
