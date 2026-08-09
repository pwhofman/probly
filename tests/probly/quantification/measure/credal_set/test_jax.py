"""Tests for jax credal set uncertainty measures.

Mirrors ``test_torch.py``. JAX defaults to float32, so the tolerances here are
looser than the torch (float64) ones. The convex upper entropy uses BFGS rather
than torch's L-BFGS, so it is checked against mathematical invariants
(``>= max vertex entropy``, ``<= log K``) instead of exact values.

Torch is deliberately never imported here: the CI jax job installs no torch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import entropy as scipy_entropy

from probly.quantification.measure.credal_set import (
    generalized_hartley,
    lower_entropy,
    upper_entropy,
)
from probly.representation.credal_set.jax import (
    JaxConvexCredalSet,
    JaxDirichletLevelSetCredalSet,
    JaxDistanceBasedCredalSet,
    JaxProbabilityIntervalsCredalSet,
)
from probly.representation.distribution.jax_categorical import (
    JaxProbabilityCategoricalDistribution,
)
from probly.utils.jax import jax_entropy

_ATOL = 1e-4


def _intervals_credal_set(lower: list, upper: list) -> JaxProbabilityIntervalsCredalSet:
    return JaxProbabilityIntervalsCredalSet(
        lower_bounds=jnp.asarray(lower),
        upper_bounds=jnp.asarray(upper),
    )


def _convex_credal_set(vertices: list) -> JaxConvexCredalSet:
    return JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(jnp.asarray(vertices)))


def _distance_credal_set(nominal: list[float], radius: float) -> JaxDistanceBasedCredalSet:
    return JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(jnp.asarray(nominal)),
        radius=jnp.asarray(radius),
    )


def test_intervals_upper_entropy_singleton_returns_exact_entropy() -> None:
    """When lower == upper the set is a singleton and upper == lower entropy."""
    probs = [0.2, 0.5, 0.3]
    cs = _intervals_credal_set(probs, probs)
    expected = float(scipy_entropy(probs))
    assert float(upper_entropy(cs)) == pytest.approx(expected, abs=_ATOL)
    assert float(lower_entropy(cs)) == pytest.approx(expected, abs=_ATOL)


def test_intervals_upper_ge_lower_entropy() -> None:
    """Upper entropy must be >= lower entropy for any valid credal set."""
    cs = JaxProbabilityIntervalsCredalSet(
        lower_bounds=jnp.array([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]]),
        upper_bounds=jnp.array([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]]),
    )
    assert bool(jnp.all(upper_entropy(cs) >= lower_entropy(cs) - _ATOL))


def test_intervals_upper_entropy_base2() -> None:
    """Upper entropy with base=2 equals natural upper entropy / ln(2)."""
    cs = _intervals_credal_set([0.1, 0.2, 0.1], [0.4, 0.5, 0.5])
    ue_nat = upper_entropy(cs)
    ue_2 = upper_entropy(cs, base=2.0)
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=_ATOL)


def test_intervals_upper_entropy_normalize() -> None:
    """Normalized upper entropy is in [0, 1]."""
    cs = _intervals_credal_set([[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]])
    assert bool(jnp.allclose(upper_entropy(cs, base="normalize"), 1.0, atol=_ATOL))


def test_intervals_lower_entropy_degenerate_is_zero() -> None:
    """A distribution concentrated on one class has zero lower entropy."""
    cs = _intervals_credal_set([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert float(lower_entropy(cs)) == pytest.approx(0.0, abs=_ATOL)


def test_intervals_batch_shape_preserved() -> None:
    """Upper/lower entropy output shape matches batch dims of the credal set."""
    cs = JaxProbabilityIntervalsCredalSet(lower_bounds=jnp.zeros((4, 3)), upper_bounds=jnp.ones((4, 3)))
    assert upper_entropy(cs).shape == (4,)
    assert lower_entropy(cs).shape == (4,)


def test_convex_upper_entropy_single_vertex_equals_entropy() -> None:
    """A singleton convex credal set (one vertex) gives exact entropy."""
    probs = [[0.2, 0.5, 0.3]]
    cs = _convex_credal_set(probs)
    expected = float(scipy_entropy(probs[0]))
    assert float(upper_entropy(cs)) == pytest.approx(expected, abs=_ATOL)


def test_convex_upper_entropy_invariants() -> None:
    """Upper entropy is between the largest vertex entropy and ``log K``.

    The BFGS optimizer differs from torch's L-BFGS, so only the mathematical
    invariants of the maximum over the hull are asserted.
    """
    vertices = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.3, 0.4],
        ]
    )
    cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))
    ue = float(upper_entropy(cs))
    assert ue >= float(jnp.max(jax_entropy(vertices))) - _ATOL
    assert ue <= np.log(3) + _ATOL


def test_convex_upper_entropy_with_a_class_that_is_zero_in_every_vertex() -> None:
    """Regression: the nan gradient through log(0) made BFGS fall back to uniform weights."""
    cs = _convex_credal_set([[0.9, 0.1, 0.0], [0.5, 0.5, 0.0]])

    assert float(upper_entropy(cs)) == pytest.approx(float(np.log(2)), abs=_ATOL)


def test_jax_entropy_gradient_is_finite_at_exact_zeros() -> None:
    p = jnp.array([0.5, 0.5, 0.0])

    gradient = jax.grad(lambda x: jax_entropy(x).sum())(p)

    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_convex_upper_ge_lower_entropy() -> None:
    """Upper entropy >= lower entropy for convex credal sets."""
    cs = _convex_credal_set(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.3, 0.4],
        ]
    )
    assert float(upper_entropy(cs)) >= float(lower_entropy(cs)) - _ATOL


def test_convex_lower_entropy_is_min_vertex_entropy() -> None:
    """Lower entropy over a hull is exactly the smallest vertex entropy."""
    vertices = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
    cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))
    assert float(lower_entropy(cs)) == pytest.approx(float(jnp.min(jax_entropy(vertices))), abs=_ATOL)


def test_convex_batch_shape_preserved() -> None:
    """Upper/lower entropy output shape matches batch dims of the credal set."""
    vertices = jax.random.uniform(jax.random.key(0), (5, 4, 3))
    vertices = vertices / jnp.sum(vertices, axis=-1, keepdims=True)
    cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))
    assert upper_entropy(cs).shape == (5,)
    assert lower_entropy(cs).shape == (5,)


def test_generalized_hartley_single_vertex_is_zero() -> None:
    """A credal set with a single vertex (singleton) has zero Hartley measure."""
    cs = _convex_credal_set([[0.3, 0.5, 0.2]])
    assert float(generalized_hartley(cs)) == pytest.approx(0.0, abs=_ATOL)


def test_generalized_hartley_corner_vertices_known_value() -> None:
    """GH for the 3-class corner-vertex credal set equals the known Moebius value.

    With all three unit-basis vertices, the upper probability of every non-empty
    subset is 1. The Moebius inversion gives ``GH_nat = ln(3)``.
    """
    cs = _convex_credal_set([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert float(generalized_hartley(cs)) == pytest.approx(np.log(3), abs=_ATOL)
    assert float(generalized_hartley(cs, base=2.0)) == pytest.approx(np.log2(3), abs=_ATOL)


def test_generalized_hartley_base_consistency() -> None:
    """GH with base=2 equals GH with natural log divided by ln(2)."""
    cs = _convex_credal_set([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    gh_nat = generalized_hartley(cs)
    gh_2 = generalized_hartley(cs, base=2.0)
    assert float(gh_2) == pytest.approx(float(gh_nat) / np.log(2), abs=_ATOL)


def test_distance_upper_entropy_singleton_returns_exact_entropy() -> None:
    """When radius is 0, upper == lower == nominal entropy."""
    probs = [0.2, 0.5, 0.3]
    cs = _distance_credal_set(probs, 0.0)
    expected = float(scipy_entropy(probs))
    assert float(upper_entropy(cs)) == pytest.approx(expected, abs=_ATOL)
    assert float(lower_entropy(cs)) == pytest.approx(expected, abs=_ATOL)


def test_distance_upper_ge_lower_entropy() -> None:
    """Upper entropy must be >= lower entropy."""
    cs = _distance_credal_set([0.6, 0.3, 0.1], 0.2)
    assert float(upper_entropy(cs)) >= float(lower_entropy(cs)) - _ATOL


def test_distance_matches_equivalent_intervals() -> None:
    """Distance-based entropy must match the equivalent probability-intervals credal set.

    A TV ball with nominal p and radius r implies
    ``lower_i = max(0, p_i - r)`` and ``upper_i = min(1, p_i + r)``.
    """
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs_dist = _distance_credal_set(nominal, radius)
    cs_int = _intervals_credal_set(
        [max(0.0, p - radius) for p in nominal],
        [min(1.0, p + radius) for p in nominal],
    )
    assert float(upper_entropy(cs_dist)) == pytest.approx(float(upper_entropy(cs_int)), abs=_ATOL)
    assert float(lower_entropy(cs_dist)) == pytest.approx(float(lower_entropy(cs_int)), abs=_ATOL)


def test_distance_batch_shape_preserved() -> None:
    """Entropy output shape matches batch dims."""
    nominal = jax.random.uniform(jax.random.key(1), (4, 3))
    nominal = nominal / jnp.sum(nominal, axis=-1, keepdims=True)
    cs = JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(nominal),
        radius=jnp.full((4,), 0.1),
    )
    assert upper_entropy(cs).shape == (4,)
    assert lower_entropy(cs).shape == (4,)


def test_distance_upper_entropy_base2() -> None:
    """Upper entropy with base=2 equals natural upper entropy / ln(2)."""
    cs = _distance_credal_set([0.5, 0.3, 0.2], 0.1)
    ue_nat = upper_entropy(cs)
    ue_2 = upper_entropy(cs, base=2.0)
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=_ATOL)


def _dirichlet_credal_set() -> JaxDirichletLevelSetCredalSet:
    return JaxDirichletLevelSetCredalSet(
        alphas=jnp.array([[2.0, 5.0, 3.0]]),
        threshold=jnp.array(0.5),
    )


def test_dirichlet_upper_entropy_finite() -> None:
    """Upper entropy of a Dirichlet level set is finite with the right batch shape."""
    result = upper_entropy(_dirichlet_credal_set())
    assert bool(jnp.isfinite(result).all())
    assert result.shape == (1,)


def test_dirichlet_lower_entropy_finite() -> None:
    """Lower entropy of a Dirichlet level set is finite with the right batch shape."""
    result = lower_entropy(_dirichlet_credal_set())
    assert bool(jnp.isfinite(result).all())
    assert result.shape == (1,)


def test_dirichlet_upper_entropy_with_explicit_base() -> None:
    """Normalized Dirichlet-level-set upper entropy stays in [0, 1]."""
    cs = _dirichlet_credal_set()
    assert bool(jnp.isfinite(upper_entropy(cs, base=None)).all())
    assert bool((upper_entropy(cs, base="normalize") <= 1.0 + _ATOL).all())


def _assert_simplex(p: jax.Array, atol: float = _ATOL) -> None:
    """Check that the last axis of ``p`` is a probability simplex element."""
    assert bool((p >= -atol).all()), p
    sums = jnp.sum(p, axis=-1)
    assert bool(jnp.allclose(sums, jnp.ones_like(sums), atol=atol)), sums


def _assert_entropy_matches(entropy: jax.Array, p: jax.Array, atol: float = _ATOL) -> None:
    """The returned entropy equals ``jax_entropy(p)`` (natural log, no base rescaling)."""
    assert bool(jnp.allclose(entropy, jax_entropy(p), atol=atol))


def test_intervals_upper_entropy_return_distribution() -> None:
    lower = jnp.array([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]])
    upper = jnp.array([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]])
    cs = JaxProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    ue, p = upper_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(ue, upper_entropy(cs)))
    assert p.shape == (2, 3)
    _assert_simplex(p)
    assert bool((p >= lower - _ATOL).all())
    assert bool((p <= upper + _ATOL).all())
    _assert_entropy_matches(ue, p)


def test_intervals_lower_entropy_return_distribution() -> None:
    lower = jnp.array([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]])
    upper = jnp.array([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]])
    cs = JaxProbabilityIntervalsCredalSet(lower_bounds=lower, upper_bounds=upper)
    le, p = lower_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(le, lower_entropy(cs)))
    assert p.shape == (2, 3)
    _assert_simplex(p)
    assert bool((p >= lower - _ATOL).all())
    assert bool((p <= upper + _ATOL).all())
    _assert_entropy_matches(le, p)


def test_intervals_singleton_returns_the_singleton() -> None:
    probs_list = [0.2, 0.5, 0.3]
    cs = _intervals_credal_set(probs_list, probs_list)
    _, p_up = upper_entropy(cs, return_distribution=True)
    _, p_lo = lower_entropy(cs, return_distribution=True)
    expected = jnp.asarray(probs_list)
    assert bool(jnp.allclose(p_up, expected, atol=_ATOL))
    assert bool(jnp.allclose(p_lo, expected, atol=_ATOL))


def test_intervals_full_simplex_lower_entropy_is_extreme_point() -> None:
    """With no bounds (lower=0, upper=1) the lower-entropy minimizer is a corner."""
    cs = _intervals_credal_set([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    le, p = lower_entropy(cs, return_distribution=True)
    assert float(le) == pytest.approx(0.0, abs=_ATOL)
    assert float(jnp.max(p)) == pytest.approx(1.0, abs=_ATOL)
    assert float(jnp.min(p)) == pytest.approx(0.0, abs=_ATOL)


def test_intervals_upper_entropy_distribution_unchanged_by_base() -> None:
    cs = _intervals_credal_set([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ue_nat, p_nat = upper_entropy(cs, return_distribution=True)
    ue_2, p_2 = upper_entropy(cs, base=2.0, return_distribution=True)
    assert bool(jnp.allclose(p_nat, p_2))
    assert float(ue_2) == pytest.approx(float(ue_nat) / np.log(2), abs=_ATOL)


def test_distance_upper_entropy_return_distribution() -> None:
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs = _distance_credal_set(nominal, radius)
    ue, p = upper_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(ue, upper_entropy(cs)))
    _assert_simplex(p)
    lower = jnp.asarray([max(0.0, x - radius) for x in nominal])
    upper = jnp.asarray([min(1.0, x + radius) for x in nominal])
    assert bool((p >= lower - _ATOL).all())
    assert bool((p <= upper + _ATOL).all())
    _assert_entropy_matches(ue, p)


def test_distance_lower_entropy_return_distribution() -> None:
    nominal = [0.5, 0.3, 0.2]
    radius = 0.15
    cs = _distance_credal_set(nominal, radius)
    le, p = lower_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(le, lower_entropy(cs)))
    _assert_simplex(p)
    lower = jnp.asarray([max(0.0, x - radius) for x in nominal])
    upper = jnp.asarray([min(1.0, x + radius) for x in nominal])
    assert bool((p >= lower - _ATOL).all())
    assert bool((p <= upper + _ATOL).all())
    _assert_entropy_matches(le, p)


def test_convex_upper_entropy_return_distribution() -> None:
    cs = _convex_credal_set(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.3, 0.3, 0.4],
        ]
    )
    ue, p = upper_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(ue, upper_entropy(cs)))
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
    le, p = lower_entropy(cs, return_distribution=True)
    assert bool(jnp.allclose(le, lower_entropy(cs)))
    assert p.shape == (3,)
    v = jnp.asarray(vertices)
    assert bool((jnp.linalg.norm(v - p, axis=-1) < _ATOL).any())
    _assert_entropy_matches(le, p)


def test_convex_upper_entropy_return_distribution_batched() -> None:
    vertices = jax.random.uniform(jax.random.key(2), (5, 4, 3))
    vertices = vertices / jnp.sum(vertices, axis=-1, keepdims=True)
    cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))
    ue, p = upper_entropy(cs, return_distribution=True)
    assert p.shape == (5, 3)
    _assert_simplex(p)
    _assert_entropy_matches(ue, p)


def test_convex_lower_entropy_return_distribution_batched() -> None:
    vertices = jax.random.uniform(jax.random.key(3), (5, 4, 3))
    vertices = vertices / jnp.sum(vertices, axis=-1, keepdims=True)
    cs = JaxConvexCredalSet(tensor=JaxProbabilityCategoricalDistribution(vertices))
    le, p = lower_entropy(cs, return_distribution=True)
    assert p.shape == (5, 3)
    _assert_simplex(p)
    diffs = jnp.linalg.norm(vertices - jnp.expand_dims(p, axis=-2), axis=-1)
    assert bool((jnp.min(diffs, axis=-1) < _ATOL).all())
    _assert_entropy_matches(le, p)


def test_dirichlet_level_set_upper_entropy_return_distribution() -> None:
    cred = _dirichlet_credal_set()
    ue, p = upper_entropy(cred, return_distribution=True)
    assert bool(jnp.allclose(ue, upper_entropy(cred)))
    assert p.shape == (1, 3)
    _assert_simplex(p)
    _assert_entropy_matches(ue, p)
    assert bool((p >= cred.lower() - _ATOL).all())
    assert bool((p <= cred.upper() + _ATOL).all())


def test_dirichlet_level_set_lower_entropy_return_distribution() -> None:
    cred = _dirichlet_credal_set()
    le, p = lower_entropy(cred, return_distribution=True)
    assert bool(jnp.allclose(le, lower_entropy(cred)))
    assert p.shape == (1, 3)
    _assert_simplex(p)
    _assert_entropy_matches(le, p)
    assert bool((p >= cred.lower() - _ATOL).all())
    assert bool((p <= cred.upper() + _ATOL).all())


def test_credal_set_entropy_decomposition_unchanged() -> None:
    """``CredalSetEntropyDecomposition`` never sets ``return_distribution``; values must match."""
    from probly.quantification.decomposition.entropy._common import CredalSetEntropyDecomposition  # noqa: PLC0415

    cs = JaxProbabilityIntervalsCredalSet(
        lower_bounds=jnp.array([[0.1, 0.2, 0.1], [0.0, 0.3, 0.2]]),
        upper_bounds=jnp.array([[0.4, 0.6, 0.5], [0.5, 0.6, 0.5]]),
    )
    dec: CredalSetEntropyDecomposition[jax.Array] = CredalSetEntropyDecomposition(credal_set=cs)
    assert bool(jnp.allclose(dec.total, upper_entropy(cs)))
    assert bool(jnp.allclose(dec.aleatoric, lower_entropy(cs)))
