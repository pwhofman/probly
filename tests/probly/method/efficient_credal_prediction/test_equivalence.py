"""Tests verifying equivalence between the Torch and SciPy/NumPy implementations."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from probly.method.efficient_credal_prediction import compute_efficient_credal_prediction_bounds  # noqa: E402


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_implementations_are_equivalent(alpha: float) -> None:
    """Ensure the fast Torch bisection matches the legacy SciPy SLSQP bounds."""
    n_samples, n_classes = 150, 4

    # Generate fixed random data
    rng = np.random.default_rng(123)
    logits_np = rng.standard_normal((n_samples, n_classes))
    targets_np = rng.integers(0, n_classes, size=n_samples)

    # Cast to Torch
    logits_t = torch.tensor(logits_np)
    targets_t = torch.tensor(targets_np)

    # Compute bounds using NumPy dispatch
    lower_np, upper_np = compute_efficient_credal_prediction_bounds(
        logits_np, targets_np, num_classes=n_classes, alpha=alpha
    )

    # Compute bounds using Torch dispatch
    lower_t, upper_t = compute_efficient_credal_prediction_bounds(
        logits_t, targets_t, num_classes=n_classes, alpha=alpha, n_iter=60
    )

    # SLSQP has a default tolerance around ~1e-6, while the bisection exhaust
    # float64 precision (~1e-11). We test at atol=1e-3 to account for SLSQP noise.
    np.testing.assert_allclose(lower_np, lower_t, atol=1e-3)
    np.testing.assert_allclose(upper_np, upper_t, atol=1e-3)
