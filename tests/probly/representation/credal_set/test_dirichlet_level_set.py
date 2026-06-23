"""Tests for Dirichlet level set credal sets."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from probly.representation.credal_set._common import create_dirichlet_level_set_credal_set
from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet


def _make_credal_set(alphas: list[float], threshold: float) -> TorchDirichletLevelSetCredalSet:
    return TorchDirichletLevelSetCredalSet(
        alphas=torch.tensor(alphas, dtype=torch.float64),
        threshold=torch.tensor(threshold, dtype=torch.float64),
    )


def test_construction() -> None:
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.5)
    assert isinstance(cs, TorchDirichletLevelSetCredalSet)
    assert cs.num_classes == 3


def test_barycenter_is_dirichlet_mean() -> None:
    cs = _make_credal_set([6.0, 3.0, 1.0], 0.5)
    expected = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    assert torch.allclose(cs.barycenter.probabilities, expected, atol=1e-6)


def test_lower_upper_valid() -> None:
    """Lower bounds should be <= upper bounds and within [0, 1]."""
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.5)
    lower = cs.lower()
    upper = cs.upper()
    assert torch.all(lower >= 0.0)
    assert torch.all(upper <= 1.0)
    assert torch.all(lower <= upper + 1e-6)


def test_high_threshold_tight_bounds() -> None:
    """With threshold near 1, bounds should be tight around the mode."""
    cs = _make_credal_set([10.0, 5.0, 3.0], 0.99)
    lower = cs.lower()
    upper = cs.upper()
    width = upper - lower
    # Bounds should be relatively tight
    assert torch.all(width < 0.3)


def test_low_threshold_wide_bounds() -> None:
    """With threshold near 0, bounds should cover most of the simplex."""
    cs = _make_credal_set([5.0, 3.0, 2.0], 0.01)
    lower = cs.lower()
    upper = cs.upper()
    width = upper - lower
    # At least some classes should have wide bounds
    assert torch.any(width > 0.3)


def test_batch_shape_preserved() -> None:
    """Batch dimensions should be preserved."""
    alphas = torch.tensor([[5.0, 3.0, 2.0], [10.0, 1.0, 1.0]], dtype=torch.float64)
    threshold = torch.tensor([0.5, 0.5], dtype=torch.float64)
    cs = TorchDirichletLevelSetCredalSet(alphas=alphas, threshold=threshold)
    assert cs.lower().shape == (2, 3)
    assert cs.upper().shape == (2, 3)
    assert cs.barycenter.shape == (2,)


def test_factory_creates_correct_type() -> None:
    """Factory function should create TorchDirichletLevelSetCredalSet."""
    alphas = torch.tensor([5.0, 3.0, 2.0], dtype=torch.float64)
    result = create_dirichlet_level_set_credal_set(alphas, 0.5)
    assert isinstance(result, TorchDirichletLevelSetCredalSet)
