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
    TorchDistanceBasedCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.torch_categorical import (
    TorchProbabilityCategoricalDistribution,
)

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
    return TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(t))


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
    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
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
    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    gh_nat = generalized_hartley(cs)
    gh_2 = generalized_hartley(cs, base=2.0)
    assert float(gh_2) == pytest.approx(float(gh_nat) / np.log(2), abs=1e-5)


# ---------------------------------------------------------------------------
# TorchDistanceBasedCredalSet
# ---------------------------------------------------------------------------


def _distance_credal_set(nominal: list[float], radius: float) -> TorchDistanceBasedCredalSet:
    return TorchDistanceBasedCredalSet(
        nominal=TorchProbabilityCategoricalDistribution(torch.tensor(nominal, dtype=torch.float64)),
        radius=torch.tensor(radius, dtype=torch.float64),
    )


def test_distance_upper_entropy_singleton_returns_exact_entropy() -> None:
    """When radius is 0, upper == lower == nominal entropy."""
    probs = [0.2, 0.5, 0.3]
    cs = _distance_credal_set(probs, 0.0)
    ue = upper_entropy(cs)
    le = lower_entropy(cs)
    expected = float(scipy_entropy(probs))
    assert float(ue) == pytest.approx(expected, abs=1e-5)
    assert float(le) == pytest.approx(expected, abs=1e-5)


def test_distance_upper_ge_lower_entropy() -> None:
    """Upper entropy must be >= lower entropy."""
    cs = _distance_credal_set([0.6, 0.3, 0.1], 0.2)
    ue = upper_entropy(cs)
    le = lower_entropy(cs)
    assert float(ue) >= float(le) - 1e-6


def test_distance_matches_equivalent_intervals() -> None:
    """Distance-based entropy must match the equivalent probability-intervals credal set.

    A TV ball with nominal p and radius r implies:
        lower_i = max(0, p_i - r)
        upper_i = min(1, p_i + r)
    """
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs_dist = _distance_credal_set(nominal, radius)

    lower = [max(0.0, p - radius) for p in nominal]
    upper = [min(1.0, p + radius) for p in nominal]
    cs_int = TorchProbabilityIntervalsCredalSet(
        lower_bounds=torch.tensor(lower, dtype=torch.float64),
        upper_bounds=torch.tensor(upper, dtype=torch.float64),
    )

    ue_dist = upper_entropy(cs_dist)
    ue_int = upper_entropy(cs_int)
    le_dist = lower_entropy(cs_dist)
    le_int = lower_entropy(cs_int)

    assert float(ue_dist) == pytest.approx(float(ue_int), abs=1e-5)
    assert float(le_dist) == pytest.approx(float(le_int), abs=1e-5)


def test_distance_batch_shape_preserved() -> None:
    """Entropy output shape matches batch dims."""
    nominal = torch.rand(4, 3, dtype=torch.float64)
    nominal = nominal / nominal.sum(dim=-1, keepdim=True)
    radius = torch.full((4,), 0.1, dtype=torch.float64)
    cs = TorchDistanceBasedCredalSet(
        nominal=TorchProbabilityCategoricalDistribution(nominal),
        radius=radius,
    )
    assert upper_entropy(cs).shape == (4,)
    assert lower_entropy(cs).shape == (4,)


def test_distance_upper_entropy_base2() -> None:
    """Upper entropy with base=2 equals natural upper entropy / ln(2)."""
    cs = _distance_credal_set([0.5, 0.3, 0.2], 0.1)
    ue_nat = upper_entropy(cs)
    ue_2 = upper_entropy(cs, base=2.0)
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=1e-5)


def _torch_modules():
    """Skip the calling test if torch is unavailable; otherwise return the module."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestQuantificationCredalSetTorchMeasures:
    """Upper / lower entropy on a Dirichlet level set credal set."""

    def test_upper_entropy_finite(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.credal_set._common import upper_entropy  # noqa: PLC0415

        # Force the torch dispatch to load.
        import probly.quantification.measure.credal_set.torch  # noqa: F401, PLC0415
        from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet  # noqa: PLC0415

        torch.manual_seed(0)
        cred = TorchDirichletLevelSetCredalSet(
            alphas=torch.tensor([[2.0, 5.0, 3.0]]),
            threshold=torch.tensor(0.5),
        )
        result = upper_entropy(cred)
        # Returns a finite tensor of the right batch shape.
        assert torch.isfinite(result).all()
        assert result.shape == (1,)

    def test_lower_entropy_finite(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.credal_set._common import lower_entropy  # noqa: PLC0415
        import probly.quantification.measure.credal_set.torch  # noqa: F401, PLC0415
        from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet  # noqa: PLC0415

        torch.manual_seed(0)
        cred = TorchDirichletLevelSetCredalSet(
            alphas=torch.tensor([[2.0, 5.0, 3.0]]),
            threshold=torch.tensor(0.5),
        )
        result = lower_entropy(cred)
        assert torch.isfinite(result).all()
        assert result.shape == (1,)

    def test_upper_entropy_with_explicit_base(self) -> None:
        torch = _torch_modules()
        from probly.quantification.measure.credal_set._common import upper_entropy  # noqa: PLC0415
        import probly.quantification.measure.credal_set.torch  # noqa: F401, PLC0415
        from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet  # noqa: PLC0415

        torch.manual_seed(0)
        cred = TorchDirichletLevelSetCredalSet(
            alphas=torch.tensor([[2.0, 5.0, 3.0]]),
            threshold=torch.tensor(0.5),
        )
        result_nat = upper_entropy(cred, base=None)
        result_normalized = upper_entropy(cred, base="normalize")
        # Normalised entropy is in [0, 1].
        assert (result_normalized <= 1.0 + 1e-5).all()
        assert torch.isfinite(result_nat).all()


# ---------------------------------------------------------------------------
# return_distribution=True: distributions accompany entropies
# ---------------------------------------------------------------------------


def _assert_simplex(p: torch.Tensor, atol: float = 1e-5) -> None:
    """Check that the last axis of ``p`` is a probability simplex element."""
    assert (p >= -atol).all(), p
    sums = p.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=atol), sums


def _assert_entropy_matches(entropy: torch.Tensor, p: torch.Tensor, atol: float = 1e-5) -> None:
    """The returned entropy equals ``torch_entropy(p)`` (natural log, no base rescaling)."""
    from probly.utils.torch import torch_entropy  # noqa: PLC0415

    assert torch.allclose(entropy, torch_entropy(p), atol=atol)


# ---- Intervals ----


def test_intervals_upper_entropy_return_distribution() -> None:
    lower = torch.tensor([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]], dtype=torch.float64)
    upper = torch.tensor([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue_default = upper_entropy(cs)
    ue, p = upper_entropy(cs, return_distribution=True)
    assert torch.allclose(ue, ue_default)
    assert p.shape == (2, 3)
    _assert_simplex(p)
    assert (p >= lower - 1e-6).all()
    assert (p <= upper + 1e-6).all()
    _assert_entropy_matches(ue, p)


def test_intervals_lower_entropy_return_distribution() -> None:
    lower = torch.tensor([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]], dtype=torch.float64)
    upper = torch.tensor([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    le_default = lower_entropy(cs)
    le, p = lower_entropy(cs, return_distribution=True)
    assert torch.allclose(le, le_default)
    assert p.shape == (2, 3)
    _assert_simplex(p)
    assert (p >= lower - 1e-6).all()
    assert (p <= upper + 1e-6).all()
    _assert_entropy_matches(le, p)


def test_intervals_singleton_returns_the_singleton() -> None:
    probs_list = [0.2, 0.5, 0.3]
    cs = _intervals_credal_set(probs_list, probs_list)
    _, p_up = upper_entropy(cs, return_distribution=True)
    _, p_lo = lower_entropy(cs, return_distribution=True)
    expected = torch.tensor(probs_list, dtype=torch.float64)
    assert torch.allclose(p_up, expected, atol=1e-5)
    assert torch.allclose(p_lo, expected, atol=1e-5)


def test_intervals_full_simplex_lower_entropy_is_extreme_point() -> None:
    """With no bounds (lower=0, upper=1) the lower-entropy minimizer is a corner."""
    lower = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    upper = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    le, p = lower_entropy(cs, return_distribution=True)
    assert float(le) == pytest.approx(0.0, abs=1e-5)
    # One coordinate is 1, others 0.
    assert float(p.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(p.min()) == pytest.approx(0.0, abs=1e-6)


def test_intervals_upper_entropy_distribution_unchanged_by_base() -> None:
    lower = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    upper = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue_nat, p_nat = upper_entropy(cs, return_distribution=True)
    ue_2, p_2 = upper_entropy(cs, base=2.0, return_distribution=True)
    # Distribution does not depend on the log base.
    assert torch.allclose(p_nat, p_2)
    # Entropy rescales by 1/ln(base).
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=1e-5)


# ---- Distance-based ----


def test_distance_upper_entropy_return_distribution() -> None:
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs = _distance_credal_set(nominal, radius)
    ue_default = upper_entropy(cs)
    ue, p = upper_entropy(cs, return_distribution=True)
    assert torch.allclose(ue, ue_default)
    _assert_simplex(p)
    lower = torch.tensor([max(0.0, x - radius) for x in nominal], dtype=torch.float64)
    upper = torch.tensor([min(1.0, x + radius) for x in nominal], dtype=torch.float64)
    assert (p >= lower - 1e-6).all()
    assert (p <= upper + 1e-6).all()
    _assert_entropy_matches(ue, p)


def test_distance_lower_entropy_return_distribution() -> None:
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs = _distance_credal_set(nominal, radius)
    le_default = lower_entropy(cs)
    le, p = lower_entropy(cs, return_distribution=True)
    assert torch.allclose(le, le_default)
    _assert_simplex(p)
    lower = torch.tensor([max(0.0, x - radius) for x in nominal], dtype=torch.float64)
    upper = torch.tensor([min(1.0, x + radius) for x in nominal], dtype=torch.float64)
    assert (p >= lower - 1e-6).all()
    assert (p <= upper + 1e-6).all()
    _assert_entropy_matches(le, p)


# ---- Convex ----


def test_convex_upper_entropy_return_distribution() -> None:
    vertices = [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.4],
    ]
    cs = _convex_credal_set(vertices)
    ue_default = upper_entropy(cs)
    ue, p = upper_entropy(cs, return_distribution=True)
    assert torch.allclose(ue, ue_default)
    assert p.shape == (3,)
    _assert_simplex(p)
    _assert_entropy_matches(ue, p)


def test_convex_lower_entropy_return_distribution_is_a_vertex() -> None:
    vertices = [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.4],
    ]
    cs = _convex_credal_set(vertices)
    le_default = lower_entropy(cs)
    le, p = lower_entropy(cs, return_distribution=True)
    assert torch.allclose(le, le_default)
    assert p.shape == (3,)
    # The minimizer must equal one of the vertices.
    v_tensor = torch.tensor(vertices, dtype=torch.float64)
    assert ((v_tensor - p).norm(dim=-1) < 1e-6).any()
    _assert_entropy_matches(le, p)


def test_convex_upper_entropy_return_distribution_batched() -> None:
    torch.manual_seed(0)
    vertices = torch.rand(5, 4, 3, dtype=torch.float64)
    vertices = vertices / vertices.sum(dim=-1, keepdim=True)
    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    ue, p = upper_entropy(cs, return_distribution=True)
    assert p.shape == (5, 3)
    _assert_simplex(p)
    _assert_entropy_matches(ue, p)


def test_convex_lower_entropy_return_distribution_batched() -> None:
    torch.manual_seed(0)
    vertices = torch.rand(5, 4, 3, dtype=torch.float64)
    vertices = vertices / vertices.sum(dim=-1, keepdim=True)
    cs = TorchConvexCredalSet(tensor=TorchProbabilityCategoricalDistribution(vertices))
    le, p = lower_entropy(cs, return_distribution=True)
    assert p.shape == (5, 3)
    _assert_simplex(p)
    # Each batch element's returned p matches one of that batch element's vertices.
    diffs = (vertices - p.unsqueeze(-2)).norm(dim=-1)  # (5, 4)
    assert (diffs.min(-1).values < 1e-6).all()
    _assert_entropy_matches(le, p)


# ---- Dirichlet level set ----


def test_dirichlet_level_set_upper_entropy_return_distribution() -> None:
    from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet  # noqa: PLC0415

    cred = TorchDirichletLevelSetCredalSet(
        alphas=torch.tensor([[2.0, 5.0, 3.0]], dtype=torch.float64),
        threshold=torch.tensor(0.5, dtype=torch.float64),
    )
    # Replay the MC draws used inside the call (lower() then upper()).
    torch.manual_seed(0)
    expected_lower = cred.lower()
    expected_upper = cred.upper()
    torch.manual_seed(0)
    ue_default = upper_entropy(cred)
    torch.manual_seed(0)
    ue, p = upper_entropy(cred, return_distribution=True)
    assert torch.allclose(ue, ue_default)
    assert p.shape == (1, 3)
    _assert_simplex(p)
    _assert_entropy_matches(ue, p)
    assert (p >= expected_lower - 1e-6).all()
    assert (p <= expected_upper + 1e-6).all()


def test_dirichlet_level_set_lower_entropy_return_distribution() -> None:
    from probly.representation.credal_set.torch import TorchDirichletLevelSetCredalSet  # noqa: PLC0415

    cred = TorchDirichletLevelSetCredalSet(
        alphas=torch.tensor([[2.0, 5.0, 3.0]], dtype=torch.float64),
        threshold=torch.tensor(0.5, dtype=torch.float64),
    )
    torch.manual_seed(0)
    expected_lower = cred.lower()
    expected_upper = cred.upper()
    torch.manual_seed(0)
    le_default = lower_entropy(cred)
    torch.manual_seed(0)
    le, p = lower_entropy(cred, return_distribution=True)
    assert torch.allclose(le, le_default)
    assert p.shape == (1, 3)
    _assert_simplex(p)
    _assert_entropy_matches(le, p)
    assert (p >= expected_lower - 1e-6).all()
    assert (p <= expected_upper + 1e-6).all()


# ---- Quantification regression: default path stays the same ----


def test_credal_set_entropy_decomposition_unchanged() -> None:
    """``CredalSetEntropyDecomposition`` never sets ``return_distribution``; values must match direct calls."""
    from probly.quantification.decomposition.entropy._common import CredalSetEntropyDecomposition  # noqa: PLC0415

    lower = torch.tensor([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]], dtype=torch.float64)
    upper = torch.tensor([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]], dtype=torch.float64)
    cs = TorchProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    dec: CredalSetEntropyDecomposition[torch.Tensor] = CredalSetEntropyDecomposition(credal_set=cs)
    assert torch.allclose(dec.total, upper_entropy(cs))
    assert torch.allclose(dec.aleatoric, lower_entropy(cs))
