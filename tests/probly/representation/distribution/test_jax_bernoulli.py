"""Tests for Jax-based Bernoulli distribution representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp

from probly.predictor import BinaryLogitClassifier, BinaryProbabilisticClassifier, predict
from probly.representation.distribution import (
    BernoulliDistribution,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
)
from probly.representation.distribution.jax_bernoulli import (
    JaxBernoulliDistribution,
    JaxLogitBernoulliDistribution,
    JaxProbabilityBernoulliDistribution,
)
from probly.representation.distribution.jax_categorical import JaxCategoricalDistribution


def test_probability_bernoulli_exposes_categorical_fields_with_class_axis() -> None:
    positive = jnp.array([0.2, 0.8], dtype=float)

    dist = JaxProbabilityBernoulliDistribution(positive)

    assert isinstance(dist, BernoulliDistribution)
    assert dist.shape == (2,)
    assert dist.num_classes == 2
    assert jnp.allclose(dist.probabilities, jnp.array([[0.8, 0.2], [0.2, 0.8]], dtype=float))
    assert jnp.allclose(dist.unnormalized_probabilities, dist.probabilities)
    assert jnp.allclose(dist.log_probabilities, jnp.log(dist.probabilities))
    assert jnp.allclose(1.0 / (1.0 + jnp.exp(-dist.logits[..., 1])), positive)


def test_logit_bernoulli_exposes_true_log_odds_as_class_1_logit_gap() -> None:
    logits = jnp.array([-2.0, 0.0, 2.0], dtype=float)

    dist = JaxLogitBernoulliDistribution(logits)

    probabilities = 1.0 / (1.0 + jnp.exp(-logits))

    assert jnp.allclose(dist.logits[..., 1] - dist.logits[..., 0], logits)
    assert jnp.allclose(dist.probabilities[..., 1], probabilities)


def test_bernoulli_to_categoircal_returns_two_class_distribution() -> None:
    dist = JaxProbabilityBernoulliDistribution(jnp.array([[0.1, 0.9]], dtype=float))

    categorical = dist.to_categorical()

    assert isinstance(categorical, JaxCategoricalDistribution)
    assert categorical.shape == (1, 2)
    assert jnp.allclose(categorical.probabilities, dist.probabilities)


def test_bernoulli_factories_accept_class_axis_and_backing_probability() -> None:
    dist_from_backing = create_bernoulli_distribution(jnp.array([0.25, 0.75], dtype=float))
    dist_from_classes = create_bernoulli_distribution(jnp.array([[0.75, 0.25], [0.25, 0.75]], dtype=float))
    logit_dist = create_bernoulli_distribution_from_logits(jnp.array([[0.0, -1.0], [0.0, 1.0]], dtype=float))

    assert isinstance(dist_from_backing, JaxBernoulliDistribution)
    assert isinstance(dist_from_classes, JaxBernoulliDistribution)
    assert isinstance(logit_dist, JaxLogitBernoulliDistribution)
    assert jnp.allclose(dist_from_backing.probabilities, dist_from_classes.probabilities)
    assert jnp.allclose(logit_dist.logits[..., 1] - logit_dist.logits[..., 0], jnp.array([-1.0, 1.0]))


def test_probability_bernoulli_rejects_invalid_probabilites() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        JaxProbabilityBernoulliDistribution(jnp.array([1.2]))


def test_binary_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> jnp.ndarray:
            return jnp.array([0.2, 0.8], dtype=float)

    predictor = BinaryProbabilisticClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, JaxBernoulliDistribution)
    assert jnp.allclose(prediction.probabilities, jnp.array([[0.8, 0.2], [0.2, 0.8]], dtype=float))


def test_binary_logit_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> jnp.ndarray:
            return jnp.array([-1.0, 1.0], dtype=float)

    predictor = BinaryLogitClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, JaxBernoulliDistribution)
    assert jnp.allclose(prediction.logits[..., 1] - prediction.logits[..., 0], jnp.array([-1.0, 1.0]))


class TestJaxBernoulliDistribution:
    """Jax-based Bernoulli distribution validation (concrete implementations)."""

    def test_invalid_probabilities_raise(self) -> None:
        from probly.representation.distribution.jax_bernoulli import (  # noqa: PLC0415
            JaxProbabilityBernoulliDistribution,
        )

        with pytest.raises(ValueError, match="must be in"):
            JaxProbabilityBernoulliDistribution(array=jnp.array([1.5]))

    def test_negative_probabilities_raise(self) -> None:
        from probly.representation.distribution.jax_bernoulli import (  # noqa: PLC0415
            JaxProbabilityBernoulliDistribution,
        )

        with pytest.raises(ValueError, match="must be in"):
            JaxProbabilityBernoulliDistribution(array=jnp.array([-0.1]))

    def test_array_must_be_ndarray(self) -> None:
        from probly.representation.distribution.jax_bernoulli import (  # noqa: PLC0415
            JaxProbabilityBernoulliDistribution,
        )

        with pytest.raises(TypeError, match="jax Array"):
            JaxProbabilityBernoulliDistribution(array=[0.3])  # type: ignore[arg-type]

    def test_logit_array_must_be_ndarray(self) -> None:

        with pytest.raises(TypeError, match="jax Array"):
            JaxLogitBernoulliDistribution(array=[0.3])  # type: ignore[arg-type]
