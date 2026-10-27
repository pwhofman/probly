"""Tests for the scoring-rule decomposition on JAX representations."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp
from jax.experimental import checkify

from probly.metrics import (
    brier_loss,
    brier_loss_per_instance,
    log_loss_per_instance,
    spherical_loss_per_instance,
    zero_one_loss_per_instance,
)
from probly.quantification import (
    SecondOrderEntropyDecomposition,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
)
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistributionSample,
    JaxProbabilityCategoricalDistribution,
)

_PROBS = jnp.array(
    [
        [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
        [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
        [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
    ],
    dtype=float,
)


def _sample() -> JaxCategoricalDistributionSample:
    return JaxCategoricalDistributionSample(
        array=JaxProbabilityCategoricalDistribution(_PROBS),
        sample_axis=0,
    )


def test_jax_log_loss_matches_entropy_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), log_loss_per_instance)
    ent = SecondOrderEntropyDecomposition(_sample())
    assert jnp.allclose(sr.total, ent.total, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.aleatoric, ent.aleatoric, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.epistemic, ent.epistemic, rtol=1e-6, atol=1e-6)


def test_jax_log_loss_matches_entropy_with_zero_probabilities() -> None:
    # theta_bar stays positive, but individual members put exact zero mass on some classes;
    # the 0 log 0 = 0 convention must keep the decomposition finite and equal to entropy.
    probs = jnp.array(
        [
            [[1.00, 0.00, 0.00], [0.50, 0.50, 0.00]],
            [[0.20, 0.50, 0.30], [0.10, 0.30, 0.60]],
        ],
        dtype=float,
    )
    sample = JaxCategoricalDistributionSample(array=JaxProbabilityCategoricalDistribution(probs), sample_axis=0)
    sr = SecondOrderScoringRuleDecomposition(sample, log_loss_per_instance)
    ent = SecondOrderEntropyDecomposition(sample)
    assert jnp.all(jnp.isfinite(sr.total))
    assert jnp.all(jnp.isfinite(sr.aleatoric))
    assert jnp.allclose(sr.total, ent.total, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.aleatoric, ent.aleatoric, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.epistemic, ent.epistemic, rtol=1e-6, atol=1e-6)


def test_jax_zero_one_loss_matches_zero_one_decomposition() -> None:
    sr = SecondOrderScoringRuleDecomposition(_sample(), zero_one_loss_per_instance)
    zo = SecondOrderZeroOneDecomposition(_sample())
    assert jnp.allclose(sr.total, zo.total, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.aleatoric, zo.aleatoric, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.epistemic, zo.epistemic, rtol=1e-6, atol=1e-6)


def test_jax_brier_loss_matches_gini_closed_form() -> None:
    mean = _PROBS.mean(axis=0)
    tu = 1.0 - jnp.sum(mean**2, axis=-1)
    au = jnp.mean(1.0 - jnp.sum(_PROBS**2, axis=-1), axis=0)
    sr = SecondOrderScoringRuleDecomposition(_sample(), brier_loss_per_instance)
    assert jnp.allclose(sr.total, tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.aleatoric, au, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.epistemic, tu - au, rtol=1e-6, atol=1e-6)


def test_jax_spherical_loss_matches_closed_form_and_is_additive() -> None:
    mean = _PROBS.mean(axis=0)
    tu = 1.0 - jnp.sqrt(jnp.sum(mean**2, axis=-1))
    au = jnp.mean(1.0 - jnp.sqrt(jnp.sum(_PROBS**2, axis=-1)), axis=0)
    sr = SecondOrderScoringRuleDecomposition(_sample(), spherical_loss_per_instance)
    assert jnp.allclose(sr.total, tu, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.aleatoric, au, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(sr.total, sr.aleatoric + sr.epistemic, rtol=1e-6, atol=1e-6)
    assert jnp.all(sr.epistemic >= -1e-6)


def test_jax_averaged_rule_is_rejected() -> None:
    # An averaged metric collapses the batch to one number, which would silently broadcast through the decomposition.
    sr = SecondOrderScoringRuleDecomposition(_sample(), brier_loss)
    with pytest.raises(ValueError, match="one loss per instance"):
        _ = sr.total


@pytest.mark.parametrize(
    "rule", [log_loss_per_instance, brier_loss_per_instance, spherical_loss_per_instance, zero_one_loss_per_instance]
)
def test_jax_decomposition_supports_jit_and_grad(rule):
    def components(probabilities):
        sample = JaxCategoricalDistributionSample(JaxProbabilityCategoricalDistribution(probabilities), sample_axis=0)
        decomposition = SecondOrderScoringRuleDecomposition(sample, rule)
        return decomposition.total, decomposition.aleatoric, decomposition.epistemic

    error, actual = jax.jit(checkify.checkify(components))(_PROBS)
    error.throw()
    expected = components(_PROBS)
    assert all(jnp.allclose(a, e, atol=1e-6) for a, e in zip(actual, expected, strict=True))
    error, gradient = jax.jit(checkify.checkify(jax.grad(lambda p: components(p)[0].sum())))(_PROBS)
    error.throw()
    assert jnp.isfinite(gradient).all()
