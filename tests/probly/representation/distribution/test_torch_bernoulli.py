"""Tests for torch Bernoulli distribution representation."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from probly.predictor import BinaryLogitClassifier, BinaryProbabilisticClassifier, predict
from probly.representation.distribution import create_bernoulli_distribution, create_bernoulli_distribution_from_logits
from probly.representation.distribution.torch_bernoulli import (
    TorchBernoulliDistribution,
    TorchLogitBernoulliDistribution,
    TorchProbabilityBernoulliDistribution,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution


def test_probability_bernoulli_exposes_categorical_fields_with_class_axis() -> None:
    positive = torch.tensor([0.2, 0.8], dtype=torch.float64)

    dist = TorchProbabilityBernoulliDistribution(positive)

    assert dist.shape == (2,)
    assert dist.num_classes == 2
    torch.testing.assert_close(dist.probabilities, torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64))
    torch.testing.assert_close(dist.unnormalized_probabilities, dist.probabilities)
    torch.testing.assert_close(dist.log_probabilities, torch.log(dist.probabilities))
    torch.testing.assert_close(torch.sigmoid(dist.logits[..., 1]), positive)


def test_logit_bernoulli_exposes_true_log_odds_as_class_1_logit_gap() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)

    dist = TorchLogitBernoulliDistribution(logits)

    torch.testing.assert_close(dist.logits[..., 1] - dist.logits[..., 0], logits)
    torch.testing.assert_close(dist.probabilities[..., 1], torch.sigmoid(logits))


def test_bernoulli_to_categorical_returns_two_class_distribution() -> None:
    dist = TorchProbabilityBernoulliDistribution(torch.tensor([[0.1, 0.9]], dtype=torch.float64))

    categorical = dist.to_categorical()

    assert isinstance(categorical, TorchCategoricalDistribution)
    assert categorical.shape == (1, 2)
    torch.testing.assert_close(categorical.probabilities, dist.probabilities)


def test_bernoulli_factories_accept_class_axis_and_backing_probability() -> None:
    dist_from_backing = create_bernoulli_distribution(torch.tensor([0.25, 0.75], dtype=torch.float64))
    dist_from_classes = create_bernoulli_distribution(torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64))
    logit_dist = create_bernoulli_distribution_from_logits(torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=torch.float64))

    assert isinstance(dist_from_backing, TorchBernoulliDistribution)
    assert isinstance(dist_from_classes, TorchBernoulliDistribution)
    assert isinstance(logit_dist, TorchLogitBernoulliDistribution)
    torch.testing.assert_close(dist_from_backing.probabilities, dist_from_classes.probabilities)
    torch.testing.assert_close(
        logit_dist.logits[..., 1] - logit_dist.logits[..., 0], torch.tensor([-1.0, 1.0], dtype=torch.float64)
    )


def test_binary_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> torch.Tensor:
            return torch.tensor([0.2, 0.8], dtype=torch.float64)

    predictor = BinaryProbabilisticClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, TorchBernoulliDistribution)
    torch.testing.assert_close(prediction.probabilities, torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64))


def test_binary_logit_predictor_converts_to_bernoulli_distribution() -> None:
    class Model:
        def predict(self) -> torch.Tensor:
            return torch.tensor([-1.0, 1.0], dtype=torch.float64)

    predictor = BinaryLogitClassifier.register_instance(Model())

    prediction = predict(predictor)

    assert isinstance(prediction, TorchBernoulliDistribution)
    torch.testing.assert_close(
        prediction.logits[..., 1] - prediction.logits[..., 0], torch.tensor([-1.0, 1.0], dtype=torch.float64)
    )
