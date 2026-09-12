"""JAX representation dispatch and multidimensional score regressions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from probly.conformal_scores import (
    APSScore,
    RAPSScore,
    SAPSScore,
    absolute_error_score,
    cqr_r_score,
    cqr_score,
    dirichlet_rl_score_func,
    inner_product_score_func,
    kl_divergence_score_func,
    lac_score,
    tv_score_func,
    uacqr_score,
    wasserstein_distance_score_func,
)
from probly.conformal_scores.inner_product.jax import compute_inner_product_score_jax
from probly.conformal_scores.kullback_leibler.jax import compute_kl_divergence_score_jax
from probly.representation.distribution.jax_categorical import (
    JaxLogitCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution
from probly.representation.sample.jax import JaxArraySample

from ._classification_target_suite import PREDICTIONS, SCORES, ClassificationTargetSuite, expected_score


@pytest.fixture
def classification_backend():
    return jnp.asarray, JaxProbabilityCategoricalDistribution, JaxLogitCategoricalDistribution


class TestClassificationTargets(ClassificationTargetSuite):
    """JAX target interpretation follows types and dtypes."""


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("integer", [False, True])
def test_target_interpretation_under_jit(score, integer):
    target = np.array([[0, 1], [1, 0]]) if integer else np.array([[0.1, 0.9], [0.7, 0.3]])
    probabilities = np.eye(2)[target] if integer else target
    result = jax.jit(score)(jnp.asarray(PREDICTIONS), jnp.asarray(target))
    np.testing.assert_allclose(result, expected_score(score, PREDICTIONS, probabilities), atol=1e-6)


def test_broadcast_target_gradients():
    predictions = jnp.asarray(PREDICTIONS)
    target = jnp.array([[0.1, 0.9], [0.7, 0.3]])
    gradients = jax.jit(jax.grad(lambda p, q: inner_product_score_func(p, q).sum(), argnums=(0, 1)))(
        predictions, target
    )
    np.testing.assert_allclose(gradients[0], -jnp.broadcast_to(target, predictions.shape))
    np.testing.assert_allclose(gradients[1], -predictions.sum(axis=0))


@pytest.mark.parametrize(
    "score",
    [
        APSScore(randomized=False),
        lac_score,
        RAPSScore(randomized=False, lambda_reg=0.3, k_reg=1),
        SAPSScore(randomized=False, lambda_val=0.3),
        tv_score_func,
        wasserstein_distance_score_func,
        inner_product_score_func,
        kl_divergence_score_func,
    ],
)
@pytest.mark.parametrize("wrapper", ["sample", "categorical", "nested"])
def test_classification_representations(score, wrapper: str) -> None:
    probabilities = jnp.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    labels = jnp.array([1, 2])
    distribution = JaxLogitCategoricalDistribution(jnp.log(probabilities))
    if wrapper == "sample":
        prediction = JaxArraySample(probabilities, sample_axis=0)
    elif wrapper == "categorical":
        prediction = distribution
    else:
        prediction = JaxArraySample(distribution, sample_axis=0)
    # Call the wrapper first to exercise lazy backend registration.
    result = score(prediction, labels)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(result, score(probabilities, labels), atol=1e-6)


@pytest.mark.parametrize("score", [absolute_error_score, cqr_score, cqr_r_score, uacqr_score])
def test_regression_samples(score) -> None:
    predictions = jnp.array([[[0.0, 2.0], [1.0, 4.0]], [[1.0, 4.0], [2.0, 5.0]]])
    labels = jnp.array([3.0, 0.0])
    if score is absolute_error_score:
        predictions = predictions[..., 0]
    result = score(JaxArraySample(predictions, sample_axis=0), labels)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(result, score(predictions, labels), atol=1e-6)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_dirichlet_relative_likelihood_samples(nested: bool, sample_axis: int) -> None:
    alphas = jnp.array([[[1.0, 2.0, 4.0], [3.0, 6.0, 2.0]], [[2.0, 4.0, 8.0], [6.0, 12.0, 4.0]]])
    labels = jnp.array([[1, 0], [2, 1]])
    values = JaxDirichletDistribution(alphas) if nested else alphas
    sample = JaxArraySample(values, sample_axis=sample_axis, weights=jnp.array([0.25, 0.75]))
    result = dirichlet_rl_score_func(sample, labels)
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(result, [[0.5, 0.5], [0.0, 0.0]])


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("score", [inner_product_score_func, kl_divergence_score_func])
def test_label_and_distribution_batching(score, batch_shape: tuple[int, ...]) -> None:
    probabilities = jnp.broadcast_to(jnp.array([0.2, 0.5, 0.3]), (*batch_shape, 3))
    labels = jnp.ones(batch_shape, dtype=int)
    one_hot = jax.nn.one_hot(labels, 3)
    expected = np.full(batch_shape, 0.5 if score is inner_product_score_func else -np.log(0.5))
    np.testing.assert_allclose(score(probabilities, labels), expected, atol=1e-6)
    np.testing.assert_allclose(score(probabilities, one_hot), expected, atol=1e-6)
    handler = compute_inner_product_score_jax if score is inner_product_score_func else compute_kl_divergence_score_jax
    np.testing.assert_allclose(jax.jit(handler)(probabilities, labels), expected, atol=1e-6)
