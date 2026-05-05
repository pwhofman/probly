"""Tests for NumPy-based Bernoulli distribution representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.predictor import BinaryLogitClassifier, BinaryProbabilisticClassifier, predict
from probly.representation.distribution import (
    BernoulliDistribution,
    create_bernoulli_distribution,
    create_bernoulli_distribution_from_logits,
)
from probly.representation.distribution.array_bernoulli import (
    ArrayBernoulliDistribution,
    ArrayLogitBernoulliDistribution,
    ArrayProbabilityBernoulliDistribution,
)
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution


def test_probability_bernoulli_exposes_categorical_fields_with_class_axis() -> None:
    positive = np.array([0.2, 0.8], dtype=float)

    dist = ArrayProbabilityBernoulliDistribution(positive)

    assert isinstance(dist, BernoulliDistribution)
    assert dist.shape == (2,)
    assert dist.num_classes == 2
    np.testing.assert_allclose(dist.probabilities, np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float))
    np.testing.assert_allclose(dist.unnormalized_probabilities, dist.probabilities)
    np.testing.assert_allclose(dist.log_probabilities, np.log(dist.probabilities))
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-dist.logits[..., 1])), positive)


def test_logit_bernoulli_exposes_true_log_odds_as_class_1_logit_gap() -> None:
    logits = np.array([-2.0, 0.0, 2.0], dtype=float)

    dist = ArrayLogitBernoulliDistribution(logits)

    probabilities = 1.0 / (1.0 + np.exp(-logits))

    np.testing.assert_allclose(dist.logits[..., 1] - dist.logits[..., 0], logits)
    np.testing.assert_allclose(dist.probabilities[..., 1], probabilities)


def test_bernoulli_to_categorical_returns_two_class_distribution() -> None:
    dist = ArrayProbabilityBernoulliDistribution(np.array([[0.1, 0.9]], dtype=float))

    categorical = dist.to_categorical()

    assert isinstance(categorical, ArrayCategoricalDistribution)
    assert categorical.shape == (1, 2)
    np.testing.assert_allclose(categorical.probabilities, dist.probabilities)


def test_bernoulli_factories_accept_class_axis_and_backing_probability() -> None:
    dist_from_backing = create_bernoulli_distribution(np.array([0.25, 0.75], dtype=float))
    dist_from_classes = create_bernoulli_distribution(np.array([[0.75, 0.25], [0.25, 0.75]], dtype=float))
    logit_dist = create_bernoulli_distribution_from_logits(np.array([[0.0, -1.0], [0.0, 1.0]], dtype=float))

    assert isinstance(dist_from_backing, ArrayBernoulliDistribution)
    assert isinstance(dist_from_classes, ArrayBernoulliDistribution)
    assert isinstance(logit_dist, ArrayLogitBernoulliDistribution)
    np.testing.assert_allclose(dist_from_backing.probabilities, dist_from_classes.probabilities)
    np.testing.assert_allclose(logit_dist.logits[..., 1] - logit_dist.logits[..., 0], np.array([-1.0, 1.0]))


def test_probability_bernoulli_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ArrayProbabilityBernoulliDistribution(np.array([1.2]))


def test_binary_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> np.ndarray:
            return np.array([0.2, 0.8], dtype=float)

    predictor = BinaryProbabilisticClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, ArrayBernoulliDistribution)
    np.testing.assert_allclose(prediction.probabilities, np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float))


def test_binary_logit_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> np.ndarray:
            return np.array([-1.0, 1.0], dtype=float)

    predictor = BinaryLogitClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, ArrayBernoulliDistribution)
    np.testing.assert_allclose(prediction.logits[..., 1] - prediction.logits[..., 0], np.array([-1.0, 1.0]))
