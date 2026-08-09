"""Tests for Jax distribution measures."""

from __future__ import annotations

from typing import Literal

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp
from scipy.stats import entropy as scipy_entropy

from probly.quantification.measure.distribution import (
    conditional_entropy,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
    mutual_information,
)
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistributionSample,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_expand_dims, jax_mean, jax_moveaxis, jax_take_along_axis

CATEGORICAL_BASES: tuple[None | float | Literal["normalize"], ...] = (None, 2.0, "normalize")
NUMERIC_BASES: tuple[None | float, ...] = (None, 2.0, 10.0)


def _resolve_categorical_base(base: None | float | Literal["normalize"], num_classes: int) -> None | float:
    if base == "normalize":
        return float(num_classes)
    return base


def _change_base_natural_log(values: jax.Array, base: None | float) -> jax.Array:
    if base is None or base == jnp.e:
        return values
    return values / jnp.log(base)


@pytest.mark.parametrize(
    "probabilities",
    [
        jnp.array([[0.25, 0.25, 0.5]], dtype=float),
        jnp.array([[0.1, 0.2, 0.7], [0.4, 0.1, 0.5]], dtype=float),
    ],
)
@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_array_categorical_entropy_matches_scipy(
    probabilities: jax.Array, base: None | float | Literal["normalize"]
) -> None:
    distribution = JaxProbabilityCategoricalDistribution(probabilities)

    measured = entropy(distribution, base=base)
    expected = scipy_entropy(probabilities, axis=-1, base=_resolve_categorical_base(base, probabilities.shape[-1]))

    assert jnp.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_jax_categorical_entropy_normalize_maps_to_unit_interval() -> None:
    probabilities = jnp.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    measured = entropy(JaxProbabilityCategoricalDistribution(probabilities), base="normalize")

    assert jnp.allclose(measured[0], 1.0, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(measured[1], 0.0, rtol=1e-12, atol=1e-12)
    assert jnp.all(measured >= 0.0)
    assert jnp.all(measured <= 1.0)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_jax_sample_second_order_measures_match_scipy(
    sample_axis: int, base: None | float | Literal["normalize"]
) -> None:
    base_probabilities = jnp.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = jax_moveaxis(base_probabilities, 0, sample_axis)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    measured_entropy_of_expected = entropy_of_expected_predictive_distribution(sample, base=base)
    measured_conditional_entropy = conditional_entropy(sample, base=base)
    measured_mutual_information = mutual_information(sample, base=base)

    expected_mean = jax_mean(probabilities, axis=sample_axis)
    scipy_base = _resolve_categorical_base(base, probabilities.shape[-1])
    expected_entropy_of_expected = scipy_entropy(expected_mean, axis=-1, base=scipy_base)
    expected_conditional_entropy = jax_mean(scipy_entropy(probabilities, axis=-1, base=scipy_base), axis=sample_axis)
    expected_mutual_information = jax_mean(
        scipy_entropy(
            probabilities,
            jax_expand_dims(expected_mean, sample_axis),
            axis=-1,
            base=scipy_base,
        ),
        axis=sample_axis,
    )

    assert jnp.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(measured_mutual_information, expected_mutual_information, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_jax_categorical_sample(sample_axis: int, base: None | float | str) -> None:
    base_probabilities = jnp.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = jax_moveaxis(base_probabilities, 0, sample_axis)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    expected_entropy = entropy_of_expected_predictive_distribution(sample, base=base)
    decomposition_sum = conditional_entropy(sample, base=base) + mutual_information(sample, base=base)

    assert jnp.allclose(expected_entropy, decomposition_sum, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_jax_sample_zero_one_measures_match_manual(sample_axis: int) -> None:
    base_probabilties = jnp.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = jax_moveaxis(base_probabilties, 0, sample_axis)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    measured_total = max_probability_complement_of_expected(sample)
    measured_aleatoric = expected_max_probability_complement(sample)
    measured_epistemic = max_disagreement(sample)

    expected_mean = jax_mean(probabilities, axis=sample_axis)
    expected_total = 1.0 - jnp.max(expected_mean, axis=-1)
    expected_aleatoric = jax_mean(1.0 - jnp.max(probabilities, axis=-1), axis=sample_axis)
    bma_argmax_expanded = jax_expand_dims(jnp.argmax(expected_mean, axis=-1), axis=(sample_axis, -1))
    per_sample_bma_prob = jax_take_along_axis(probabilities, bma_argmax_expanded, axis=-1).squeeze(-1)
    expected_epistemic = jax_mean(jnp.max(probabilities, axis=-1) - per_sample_bma_prob, axis=sample_axis)

    assert jnp.allclose(measured_total, expected_total, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(measured_aleatoric, expected_aleatoric, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(measured_epistemic, expected_epistemic, rtol=1e-12, atol=1e-12)


def test_jax_sample_zero_one_known_value() -> None:
    probabilities = jnp.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ],
        dtype=float,
    )
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    assert jnp.allclose(max_probability_complement_of_expected(sample), 0.45, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(expected_max_probability_complement(sample), 0.15, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(max_disagreement(sample), 0.30, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_zero_one_identity_holds_for_jax_categorical_sample(sample_axis: int) -> None:
    base_probabilities = jnp.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.5]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    probabilities = jax_moveaxis(base_probabilities, 0, sample_axis)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=sample_axis,
    )

    total = max_probability_complement_of_expected(sample)
    aleatoric = expected_max_probability_complement(sample)
    epistemic = max_disagreement(sample)

    assert jnp.allclose(total, aleatoric + epistemic, rtol=1e-12, atol=1e-12)


def test_jax_sample_min_expected_total_variation_known_value_binary() -> None:
    """EU = 1/2 min_q E||p - q||_1 for the K=2 example where it differs from TU - AU.

    Q puts equal mass on (0.9, 0.1) and (0.5, 0.5). The optimal q is (0.7, 0.3), so EU = 0.2,
    while the zero-one epistemic term TU - AU is 0.0.
    """
    probabilties = jnp.array([[0.90, 0.10], [0.50, 0.50]], dtype=float)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilties),
        sample_axis=0,
    )

    assert jnp.allclose(min_expected_total_variation(sample), 0.2, rtol=1e-7, atol=1e-7)


def test_jax_sample_min_expected_total_variation_known_value_ternary_contstrained() -> None:
    """EU for a K=3 case where the simplex constraint binds (the per-class medians sum to 0.8)."""
    probabilities = jnp.array(
        [[0.70, 0.20, 0.10], [0.50, 0.40, 0.10], [0.10, 0.10, 0.80]],
        dtype=float,
    )
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    assert jnp.allclose(min_expected_total_variation(sample), 0.3, rtol=1e-9, atol=1e-9)


def test_jax_sample_min_expected_total_variation_is_zero_for_no_second_order_spread() -> None:
    """A second-order Dirac (all samples identical) has no epistemic uncertainty."""
    probabilities = jnp.tile(jnp.array([1 / 3, 1 / 3, 1 / 3], dtype=float), (5, 1))
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    assert jnp.allclose(min_expected_total_variation(sample), 0.0, atol=1e-9)


def test_jax_sample_min_expected_total_variation_is_maximal_for_uniform_diracs() -> None:
    """EU attains its upper bound (K-1)/K for a uniform mixture of first-order Diracs."""
    probabilities = jnp.eye(3, dtype=float)  # one-hot samples on each vertex
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    assert jnp.allclose(min_expected_total_variation(sample), 2.0 / 3.0, rtol=1e-7, atol=1e-7)


def test_array_sample_min_expected_total_variation_differs_from_zero_one_epistemic() -> None:
    """The OT epistemic measure is genuinely distinct from the additive zero-one EU (TU - AU)."""
    probabilities = jnp.array([[0.90, 0.10], [0.50, 0.50]], dtype=float)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    wasserstein_eu = min_expected_total_variation(sample)
    zero_one_eu = max_disagreement(sample)

    assert not jnp.allclose(wasserstein_eu, zero_one_eu)
    assert jnp.allclose(zero_one_eu, 0.0, atol=1e-12)
