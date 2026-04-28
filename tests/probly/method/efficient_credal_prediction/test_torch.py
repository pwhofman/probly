"""Tests for the PyTorch implementation of efficient credal prediction calibration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from probly.method.efficient_credal_prediction import compute_efficient_credal_prediction_bounds  # noqa: E402


class TestTorchCredalBounds:
    """Tests the basic properties of the optimized PyTorch bisection solver."""

    @pytest.fixture
    def dummy_data(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        torch.manual_seed(42)
        n_samples, n_classes = 100, 5
        logits = torch.randn(n_samples, n_classes)
        targets = torch.randint(0, n_classes, (n_samples,))
        return logits, targets, n_classes

    def test_bounds_shape_and_type(self, dummy_data: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        """Ensure the outputs correctly return numpy arrays from the torch input."""
        logits, targets, num_classes = dummy_data
        lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes=num_classes, alpha=0.5)

        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape == (num_classes,)
        assert upper.shape == (num_classes,)
        assert lower.dtype == np.float64
        assert upper.dtype == np.float64

    def test_bounds_sign_invariants(self, dummy_data: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        """The unperturbed logits always satisfy alpha=0; so lower <= 0 <= upper."""
        logits, targets, num_classes = dummy_data
        lower, upper = compute_efficient_credal_prediction_bounds(logits, targets, num_classes=num_classes, alpha=0.5)

        assert np.all(lower <= 0.0), "Lower bounds must be <= 0"
        assert np.all(upper >= 0.0), "Upper bounds must be >= 0"

    def test_chunk_size_invariance(self, dummy_data: tuple[torch.Tensor, torch.Tensor, int]) -> None:
        """Chunk size is a memory optimization; it should not affect the math."""
        logits, targets, num_classes = dummy_data

        lower_1, upper_1 = compute_efficient_credal_prediction_bounds(logits, targets, num_classes, 0.5, chunk_size=1)
        lower_all, upper_all = compute_efficient_credal_prediction_bounds(
            logits, targets, num_classes, 0.5, chunk_size=num_classes
        )

        np.testing.assert_allclose(lower_1, lower_all, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(upper_1, upper_all, rtol=1e-12, atol=1e-12)
