"""Tests for Jax distribution measures."""

from __future__ import annotations

from typing import Literal

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp
from scipy.stats import dirichlet, entropy as scipy_entropy, norm

from probly.quantification.measure.distribution import (
    conditional_entropy,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    min_expected_total_variation,
    mutual_information,
    vacuity,
)
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistributionSample,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.distribution.jax_gaussian import JaxGaussianDistribution
from probly.representation.jax_functions import jax_expand_dims, jax_mean, jax_moveaxis, jax_sum, jax_take_along_axis

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

    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_categorical_entropy_normalize_maps_to_unit_interval() -> None:
    probabilities = jnp.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    measured = entropy(JaxProbabilityCategoricalDistribution(probabilities), base="normalize")

    assert jnp.allclose(measured[0], 1.0, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(measured[1], 0.0, rtol=1e-7, atol=1e-7)
    assert jnp.all(measured >= 0.0)
    assert jnp.all(measured <= 1.0)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_jax_dirichlet_entropy_matches_scipy(base: None | float) -> None:
    alphas = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5],
        ],
        dtype=float,
    )
    distribution = JaxDirichletDistribution(alphas)

    measured = entropy(distribution, base=base)
    expected_natural = jnp.array([dirichlet(alpha).entropy() for alpha in alphas], dtype=float)
    expected = _change_base_natural_log(expected_natural, base)

    assert jnp.allclose(measured, expected, rtol=1e-10, atol=1e-7)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_jax_gaussian_entropy_matches_scipy_norm(base: None | float) -> None:
    mean = jnp.array([0.0, 3.5, -1.0], dtype=float)
    var = jnp.array([1.0, 0.25, 2.0], dtype=float)
    distribution = JaxGaussianDistribution(mean=mean, var=var)

    measured = entropy(distribution, base=base)
    expected_natural = norm(scale=jnp.sqrt(var)).entropy()
    expected = _change_base_natural_log(expected_natural, base)

    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


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

    assert jnp.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(measured_mutual_information, expected_mutual_information, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_jax_dirichlet_entropy_of_expected_predictive_distribution_matches_scipy(
    base: None | float | Literal["normalize"],
) -> None:
    alphas = jnp.array(
        [
            [2.0, 3.0, 5.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    distribution = JaxDirichletDistribution(alphas)

    measured = entropy_of_expected_predictive_distribution(distribution, base=base)
    expected_mean = alphas / jax_sum(alphas, axis=-1, keepdims=True)
    expected = scipy_entropy(expected_mean, axis=-1, base=_resolve_categorical_base(base, expected_mean.shape[-1]))

    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_jax_dirichlet_conditional_entropy_and_mutual_information_known_points(base: None | float) -> None:
    num_classes = 3
    expected_uniform_entropy = _change_base_natural_log(jnp.asarray(jnp.log(num_classes), dtype=float), base)

    concentrated = JaxDirichletDistribution(jnp.array([1000.0, 1000.0, 1000.0], dtype=float))
    concentrated_conditional = conditional_entropy(concentrated, base=base)
    concentrated_mutual_information = mutual_information(concentrated, base=base)
    concentrated_entropy_of_expected = entropy_of_expected_predictive_distribution(concentrated, base=base)

    assert jnp.allclose(concentrated_entropy_of_expected, expected_uniform_entropy, atol=1e-7)
    assert concentrated_conditional == pytest.approx(expected_uniform_entropy, abs=2e-3)
    assert concentrated_mutual_information >= 0.0
    assert concentrated_mutual_information < 1e-3

    corner_like = JaxDirichletDistribution(jnp.array([1e-3, 1e-3, 1e-3], dtype=float))
    corner_like_conditional = conditional_entropy(corner_like, base=base)
    corner_like_mutual_information = mutual_information(corner_like, base=base)
    corner_like_entropy_of_expected = entropy_of_expected_predictive_distribution(corner_like, base=base)

    assert jnp.allclose(corner_like_entropy_of_expected, expected_uniform_entropy, atol=1e-7)
    assert corner_like_conditional >= 0.0
    assert corner_like_conditional < 1e-2
    assert corner_like_mutual_information > 0.95 * expected_uniform_entropy
    assert corner_like_mutual_information <= corner_like_entropy_of_expected


def test_jax_normalize_base_unsupported_for_non_categorical_entropies() -> None:
    dirichlet_distribution = JaxDirichletDistribution(jnp.array([2.0, 3.0, 5.0], dtype=float))
    gaussian_distribution = JaxGaussianDistribution(
        mean=jnp.array([0.0], dtype=float), var=jnp.array([1.0], dtype=float)
    )

    with pytest.raises(ValueError, match="normalization is not supported"):
        entropy(dirichlet_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        entropy(gaussian_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        conditional_entropy(dirichlet_distribution, base="normalize")

    with pytest.raises(ValueError, match="normalization is not supported"):
        mutual_information(dirichlet_distribution, base="normalize")


@pytest.mark.parametrize("base", NUMERIC_BASES)
def test_identity_holds_for_jax_dirichlet(base: None | float) -> None:
    alphas = jnp.array(
        [
            [1.5, 2.0, 3.5],
            [10.0, 10.0, 10.0],
            [1e-2, 2e-2, 3e-2],
        ],
        dtype=float,
    )
    distribution = JaxDirichletDistribution(alphas)

    expected_entropy = entropy_of_expected_predictive_distribution(distribution, base=base)
    decomposition_sum = conditional_entropy(distribution, base=base) + mutual_information(distribution, base=base)

    assert jnp.allclose(expected_entropy, decomposition_sum, rtol=1e-10, atol=1e-7)


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

    assert jnp.allclose(measured_total, expected_total, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(measured_aleatoric, expected_aleatoric, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(measured_epistemic, expected_epistemic, rtol=1e-7, atol=1e-7)


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

    assert jnp.allclose(max_probability_complement_of_expected(sample), 0.45, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(expected_max_probability_complement(sample), 0.15, rtol=1e-7, atol=1e-7)
    assert jnp.allclose(max_disagreement(sample), 0.30, rtol=1e-7, atol=1e-7)


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

    assert jnp.allclose(total, aleatoric + epistemic, rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_vacuity_known_values() -> None:
    alphas = jnp.array(
        [
            [1.0, 1.0, 1.0],  # uniform Dir(1,1,1): K=3, alpha_0=3 -> vacuity=1
            [10.0, 10.0, 10.0],  # K=3, alpha_0=30 -> vacuity=0.1
            [2.0, 3.0, 5.0],  # K=3, alpha_0=10 -> vacuity=0.3
        ],
        dtype=float,
    )
    distribution = JaxDirichletDistribution(alphas)

    measured = vacuity(distribution)

    assert jnp.allclose(measured, jnp.array([1.0, 0.1, 0.3]), rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_vacuity_lies_in_unit_interval() -> None:
    prng = jax.random.key(0)
    alphas = jax.random.uniform(key=prng, minval=1.0, maxval=20.0, shape=(50, 4))
    distribution = JaxDirichletDistribution(alphas)

    measured = vacuity(distribution)

    assert jnp.all(measured > 0.0)
    assert jnp.all(measured <= 1.0)


def test_jax_dirichlet_vacutiry_decreses_with_evidence() -> None:
    weak = JaxDirichletDistribution(jnp.array([1.0, 1.0, 1.0], dtype=float))
    strong = JaxDirichletDistribution(jnp.array([100.0, 100.0, 100.0], dtype=float))

    assert vacuity(weak) > vacuity(strong)


def test_jax_dirichlet_max_probability_complement_of_expected_known_values() -> None:
    alphas = jnp.array(
        [
            [1.0, 1.0, 1.0],  # uniform: max(1/3) -> 1 - 1/3 = 2/3
            [10.0, 1.0, 1.0],  # max = 10/12 -> 1 - 5/6 = 1/6
            [2.0, 3.0, 5.0],  # max = 5/10 -> 1 - 1/2 = 1/2
        ],
        dtype=float,
    )
    distribution = JaxDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected = jnp.array([2.0 / 3.0, 1.0 / 6.0, 0.5])
    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_max_probability_complement_of_expected_matches_explicit_formula() -> None:
    prng = jax.random.key(1)
    alphas = jax.random.uniform(key=prng, minval=0.5, maxval=20.0, shape=(50, 5))
    distribution = JaxDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected_mean = alphas / alphas.sum(axis=-1, keepdims=True)
    expected = 1.0 - jnp.max(expected_mean, axis=-1)
    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_max_probability_complement_of_expected_lies_in_unit_interval() -> None:
    prng = jax.random.key(1)
    alphas = jax.random.uniform(key=prng, minval=0.1, maxval=50.0, shape=(50, 6))
    distribution = JaxDirichletDistribution(alphas)

    measured = max_probability_complement_of_expected(distribution)

    assert jnp.all(measured >= 0.0)
    assert jnp.all(measured < 1.0)


def test_jax_dirichlet_max_probability_complement_of_expected_invariant_to_scaling() -> None:
    """Scaling the alphas by a constant leaves the predictive mean (and thus the score) unchanged."""
    base = jnp.array([1.0, 2.0, 3.0], dtype=float)
    weak = JaxDirichletDistribution(base)
    strong = JaxDirichletDistribution(100.0 * base)

    assert jnp.allclose(
        max_probability_complement_of_expected(weak),
        max_probability_complement_of_expected(strong),
        rtol=1e-7,
        atol=1e-7,
    )


def test_jax_gaussian_dempster_shafer_uniform_logits_with_default_factor() -> None:
    """Uniform-zero logits should give a vacuity = K / (K + K * exp(0)) = 1/2."""
    mean = jnp.zeros((3, 5), dtype=float)
    var = jnp.ones_like(mean)
    distribution = JaxGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    assert jnp.allclose(measured, 0.5, rtol=1e-7, atol=1e-7)


def test_jax_gaussian_dempster_shafer_matches_explicit_formula() -> None:
    import math  # noqa: PLC0415

    prng = jax.random.key(1)
    mean = jax.random.normal(key=prng, shape=(20, 5))
    var = jax.random.uniform(key=prng, minval=0.01, maxval=4.0, shape=(20, 5))
    distribution = JaxGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    num_classes = mean.shape[-1]
    adjusted = mean / jnp.sqrt(1.0 + (math.pi / 8.0) * var)
    expected = num_classes / (num_classes + jax_sum(jnp.exp(adjusted), axis=-1))
    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_gaussian_dempster_shafer_lies_in_unit_interval() -> None:
    prng = jax.random.key(1)
    mean = jax.random.normal(key=prng, shape=(50, 4))
    var = jax.random.uniform(key=prng, minval=0.01, maxval=10.0, shape=(50, 4))
    distribution = JaxGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    assert jnp.all(measured > 0.0)
    assert jnp.all(measured <= 1.0)


def test_jax_gaussian_dempster_shafer_high_variance_increased_uncertainty() -> None:
    """Mean-field correction shrinks logits when variance is large -> vacuity goes up."""
    mean = jnp.array([[10.0, -10.0, 0.0, 0.0]], dtype=float)
    low_var = jnp.full_like(mean, 1e-3)
    high_var = jnp.full_like(mean, 1000.0)

    low_var_score = dempster_shafer_uncertainty(JaxGaussianDistribution(mean=mean, var=low_var))
    high_var_score = dempster_shafer_uncertainty(JaxGaussianDistribution(mean=mean, var=high_var))

    assert high_var_score[0] > low_var_score[0]


def test_jax_gaussian_dempster_shafer_zero_factor_disables_mean_field() -> None:
    """``mean_field_factor=0`` should reduce to the variance-free formula K / (K + sum exp(h))."""
    mean = jnp.array([[1.0, 2.0, 3.0]], dtype=float)
    var = jnp.array([[100.0, 100.0, 100.0]], dtype=float)
    distribution = JaxGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution, mean_field_factor=0.0)

    expected = 3.0 / (3.0 + jax_sum(jnp.exp(mean), axis=-1))
    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


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
    assert jnp.allclose(zero_one_eu, 0.0, atol=1e-7)


def test_jax_dirichlet_min_expected_total_variation_delegates_to_sampling() -> None:
    """The Dirichlet EU draws Monte-Carlo samples and reuses the sample estimator."""
    alphas = jnp.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=float)
    distribution = JaxDirichletDistribution(alphas)

    measured = min_expected_total_variation(distribution, num_samples=500, generator=jax.random.key(0))
    reference_sample = distribution.sample(500, prng_key=jax.random.key(0))
    expected = min_expected_total_variation(reference_sample)

    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_expected_max_probability_complement_delegates_to_sampling() -> None:
    """The Dirichlet aleatoric uncertainty draws Monte-Carlo samples and reuses the sample estimator."""
    alphas = jnp.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=float)
    distribution = JaxDirichletDistribution(alphas)

    measured = expected_max_probability_complement(distribution, num_samples=500, generator=jax.random.key(0))
    reference_sample = distribution.sample(500, prng_key=jax.random.key(0))
    expected = expected_max_probability_complement(reference_sample)

    assert jnp.allclose(measured, expected, rtol=1e-7, atol=1e-7)


def test_jax_dirichlet_distance_measures_concentrated_limits() -> None:
    """A near-uniform Dirichlet has EU ~ 0 and AU ~ (K-1)/K. A near-vertex one has both near 0."""
    near_uniform = JaxDirichletDistribution(jnp.array([1000.0, 1000.0, 1000.0], dtype=float))
    eu_uniform = min_expected_total_variation(near_uniform, num_samples=4000, generator=jax.random.key(0))
    au_uniform = expected_max_probability_complement(near_uniform, num_samples=4000, generator=jax.random.key(1))
    assert eu_uniform == pytest.approx(0.0, abs=2e-2)
    assert au_uniform == pytest.approx(2.0 / 3.0, abs=2e-2)

    near_vertex = JaxDirichletDistribution(jnp.array([1000.0, 1.0, 1.0], dtype=float))
    eu_vertex = min_expected_total_variation(near_vertex, num_samples=4000, generator=jax.random.key(2))
    au_vertex = expected_max_probability_complement(near_vertex, num_samples=4000, generator=jax.random.key(3))
    assert eu_vertex == pytest.approx(0.0, abs=2e-2)
    assert au_vertex == pytest.approx(0.0, abs=2e-2)


def test_jax_dirichlet_min_expected_total_variation_in_range() -> None:
    prng = jax.random.key(1)
    alphas = jax.random.uniform(key=prng, minval=0.5, maxval=20.0, shape=(8, 4))
    distribution = JaxDirichletDistribution(alphas)

    measured = min_expected_total_variation(distribution, num_samples=1000, generator=jax.random.key(0))

    assert jnp.all(measured >= 0.0)
    assert jnp.all(measured <= 3.0 / 4.0 + 1e-9)


def test_jax_categorical_entropy_stays_in_jax() -> None:
    """Regression: the entropy measures used to round-trip through scipy on the host."""
    from probly.quantification.measure.distribution.jax import jax_categorical_entropy  # noqa: PLC0415

    probabilities = jnp.array([[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]], dtype=float)

    measured = jax_categorical_entropy(probabilities)
    jitted = jax.jit(jax_categorical_entropy)(probabilities)

    assert isinstance(measured, jax.Array)
    assert jnp.allclose(measured, jitted, rtol=1e-6, atol=1e-6)


def test_jax_mutual_information_stays_in_jax() -> None:
    probabilities = jnp.array([[0.7, 0.2, 0.1], [0.15, 0.35, 0.5]], dtype=float)
    sample = JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    assert isinstance(mutual_information(sample), jax.Array)


def test_vacuity_and_dempster_shafer_are_not_registered_for_jax() -> None:
    """The jax backend has no Dirichlet/Gaussian representation, so these have no jax handler."""
    from probly.quantification.measure.distribution import dempster_shafer_uncertainty, vacuity  # noqa: PLC0415

    alphas = jnp.array([[1.0, 2.0, 3.0]], dtype=float)

    with pytest.raises(NotImplementedError, match="Vacuity"):
        vacuity(alphas)
    with pytest.raises(NotImplementedError):
        dempster_shafer_uncertainty(alphas)
