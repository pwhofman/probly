"""Tests for the SciPy/NumPy implementation of efficient credal prediction calibration."""

from __future__ import annotations

import importlib.util
from unittest.mock import patch
import warnings

import pytest

pytest.importorskip("scipy")
import numpy as np
import scipy.special

from probly.method.efficient_credal_prediction import compute_efficient_credal_prediction_bounds
from probly.method.efficient_credal_prediction._common import compute_efficient_credal_bounds


def _is_torch_available() -> bool:
    """Check if torch is installed without triggering an ImportError."""
    return importlib.util.find_spec("torch") is not None


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

    def test_numpy_fallback_warning_without_torch(self, dummy_data: tuple[np.ndarray, np.ndarray, int]) -> None:
        """Verify that if torch is missing, the scipy backend runs and warns the user."""
        logits, targets, num_classes = dummy_data

        # We "patch" our helper function to pretend torch doesn't exist
        with patch("probly.method.efficient_credal_prediction.numpy._is_torch_available", return_value=False):
            with pytest.warns(UserWarning, match="massive speedup"):
                lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes, 0.1)

            assert isinstance(lower, np.ndarray)
            assert isinstance(upper, np.ndarray)

    @pytest.mark.skipif(not _is_torch_available(), reason="Requires PyTorch to test the silent upgrade.")
    def test_numpy_upgrades_to_torch_silently(self, dummy_data: tuple[np.ndarray, np.ndarray, int]) -> None:
        """Verify that if torch is present, the array is upgraded without a warning."""
        logits, targets, num_classes = dummy_data

        # We patch it to explicitly return True (even though it likely is True anyway)
        with patch("probly.method.efficient_credal_prediction.numpy._is_torch_available", return_value=True):
            with warnings.catch_warnings():
                # This will FAIL the test if ANY warning is raised
                warnings.simplefilter("error")

                lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes, 0.1)

            # Ensure the re-dispatch properly converted the result back to numpy
            assert isinstance(lower, np.ndarray)
            assert isinstance(upper, np.ndarray)


class TestNumpyCredalIntervalBounds:
    """Tests the NumPy inference step turning logits and offsets into packed interval bounds."""

    @pytest.fixture
    def inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((6, 4))
        lower = -rng.uniform(0.1, 2.0, size=4)
        upper = rng.uniform(0.1, 2.0, size=4)
        return logits, lower, upper

    def test_zero_offsets_collapse_to_softmax(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        logits, _, _ = inputs
        zeros = np.zeros(4)

        packed = compute_efficient_credal_bounds(logits, zeros, zeros)

        probabilities = scipy.special.softmax(logits, axis=-1)
        np.testing.assert_allclose(packed, np.concatenate([probabilities, probabilities], axis=-1))

    def test_matches_explicit_perturbations(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        logits, lower, upper = inputs
        perturbed = []
        for offsets in (lower, upper):
            for k in range(4):
                shifted = logits.copy()
                shifted[:, k] += offsets[k]
                perturbed.append(scipy.special.softmax(shifted, axis=-1))
        stacked = np.stack(perturbed)  # (2K, B, K)

        packed = compute_efficient_credal_bounds(logits, lower, upper)

        assert packed.shape == (6, 8)
        np.testing.assert_allclose(packed[:, :4], stacked.min(axis=0))
        np.testing.assert_allclose(packed[:, 4:], stacked.max(axis=0))

    @pytest.mark.skipif(not _is_torch_available(), reason="Requires PyTorch for the cross-backend comparison.")
    def test_agrees_with_torch(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        import torch  # noqa: PLC0415

        logits, lower, upper = inputs

        packed = compute_efficient_credal_bounds(logits, lower, upper)
        torch_packed = compute_efficient_credal_bounds(
            torch.as_tensor(logits), torch.as_tensor(lower), torch.as_tensor(upper)
        )

        np.testing.assert_allclose(packed, torch_packed.numpy(), rtol=1e-12)
