"""Tests for torch credal set uncertainty measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import numpy as np
from scipy.stats import entropy as scipy_entropy
import torch

from probly.quantification.measure.credal_set import (
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _intervals_credal_set(lower: list, upper: list) -> TorchProbabilityIntervalsCredalSet:
    return TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.tensor(lower, dtype=torch.float64),
        upper_bounds=torch.tensor(upper, dtype=torch.float64),
    )


def _convex_credal_set(vertices: list) -> TorchConvexCredalSet:
    t = torch.tensor(vertices, dtype=torch.float64)
    return TorchConvexCredalSet(tensor=TorchCategoricalDistribution(t))


# ---------------------------------------------------------------------------
# TorchProbabilityIntervalsCredalSet
# ---------------------------------------------------------------------------


def test_intervals_upper_entropy_singleton_returns_exact_entropy() -> None:
    """When lower == upper the set is a singleton and upper == lower entropy."""
    probs = [0.2, 0.5, 0.3]
    cs = _intervals_credal_set(probs, probs)
    ue = upper_entropy(cs)
    le = lower_entropy(cs)
    expected = float(scipy_entropy(probs))
    assert float(ue) == pytest.approx(expected, abs=1e-5)
    assert float(le) == pytest.approx(expected, abs=1e-5)


def test_intervals_upper_ge_lower_entropy() -> None:
    """Upper entropy must be >= lower entropy for any valid credal set."""
    lower = torch.tensor([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]], dtype=torch.float64)
    upper = torch.tensor([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue = upper_entropy(cs)
    le = lower_entropy(cs)
    assert torch.all(ue >= le - 1e-6)


def test_intervals_upper_entropy_base2() -> None:
    """Upper entropy with base=2 equals natural upper entropy / ln(2)."""
    lower = torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)
    upper = torch.tensor([0.4, 0.5, 0.5], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue_nat = upper_entropy(cs)
    ue_2 = upper_entropy(cs, base=2.0)
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=1e-5)


def test_intervals_upper_entropy_normalize() -> None:
    """Normalized upper entropy is in [0, 1]."""
    lower = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue = upper_entropy(cs, base="normalize")
    assert float(ue) == pytest.approx(1.0, abs=1e-5)


def test_intervals_lower_entropy_degenerate_is_zero() -> None:
    """A distribution concentrated on one class has zero lower entropy."""
    lower = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    upper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    le = lower_entropy(cs)
    assert float(le) == pytest.approx(0.0, abs=1e-5)


def test_intervals_batch_shape_preserved() -> None:
    """Upper/lower entropy output shape matches batch dims of the credal set."""
    lower = torch.zeros(4, 3, dtype=torch.float64)
    upper = torch.ones(4, 3, dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    assert upper_entropy(cs).shape == (4,)
    assert lower_entropy(cs).shape == (4,)


# ---------------------------------------------------------------------------
# TorchConvexCredalSet
# ---------------------------------------------------------------------------


def test_convex_upper_entropy_single_vertex_equals_entropy() -> None:
    """A singleton convex credal set (one vertex) gives exact entropy."""
    probs = [[0.2, 0.5, 0.3]]  # single vertex
    cs = _convex_credal_set(probs)
    ue = upper_entropy(cs)
    expected = float(scipy_entropy(probs[0]))
    assert float(ue) == pytest.approx(expected, abs=1e-5)


def test_convex_upper_ge_lower_entropy() -> None:
    """Upper entropy >= lower entropy for convex credal sets."""
    vertices = [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.4],
    ]
    cs = _convex_credal_set(vertices)
    assert float(upper_entropy(cs)) >= float(lower_entropy(cs)) - 1e-6


def test_convex_batch_shape_preserved() -> None:
    """upper/lower entropy output shape matches batch dims of the credal set."""
    vertices = torch.rand(5, 4, 3, dtype=torch.float64)
    vertices = vertices / vertices.sum(dim=-1, keepdim=True)
    cs = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(vertices))
    assert upper_entropy(cs).shape == (5,)
    assert lower_entropy(cs).shape == (5,)


# ---------------------------------------------------------------------------
# Generalized Hartley
# ---------------------------------------------------------------------------


def test_generalized_hartley_single_vertex_is_zero() -> None:
    """A credal set with a single vertex (singleton) has zero Hartley measure."""
    probs = [[0.3, 0.5, 0.2]]
    cs = _convex_credal_set(probs)
    gh = generalized_hartley(cs)
    assert float(gh) == pytest.approx(0.0, abs=1e-5)


def test_generalized_hartley_corner_vertices_known_value() -> None:
    """GH for the 3-class corner-vertex credal set equals the known Möbius value.

    With all three unit-basis vertices, the upper probability of every non-empty
    subset is 1. The Möbius inversion gives:
        GH_nat = -3*ln(2) + ln(3)   (in natural log)
    """
    vertices = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cs = _convex_credal_set(vertices)
    gh_nat = generalized_hartley(cs)
    gh_2 = generalized_hartley(cs, base=2.0)
    expected_nat = np.log(3)
    expected_2 = np.log2(3)
    assert float(gh_nat) == pytest.approx(expected_nat, abs=1e-4)
    assert float(gh_2) == pytest.approx(expected_2, abs=1e-4)


def test_generalized_hartley_base_consistency() -> None:
    """GH with base=2 equals GH with natural log divided by ln(2)."""
    vertices = torch.tensor(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]],
        dtype=torch.float64,
    )
    cs = TorchConvexCredalSet(tensor=TorchCategoricalDistribution(vertices))
    gh_nat = generalized_hartley(cs)
    gh_2 = generalized_hartley(cs, base=2.0)
    assert float(gh_2) == pytest.approx(float(gh_nat) / np.log(2), abs=1e-5)
