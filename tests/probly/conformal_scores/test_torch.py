"""Torch representation dispatch and multidimensional score regressions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from probly.conformal_scores import (
    dirichlet_rl_score_func,
    inner_product_score_func,
    kl_divergence_score_func,
    tv_score_func,
    wasserstein_distance_score_func,
)
from probly.representation.distribution.torch_categorical import (
    TorchLogitCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.sample.torch import TorchSample

from ._classification_target_suite import PREDICTIONS, ClassificationTargetSuite


@pytest.fixture
def classification_backend():
    return torch.as_tensor, TorchProbabilityCategoricalDistribution, TorchLogitCategoricalDistribution


class TestClassificationTargets(ClassificationTargetSuite):
    """Torch target interpretation follows types and dtypes."""


def test_broadcast_target_gradients():
    predictions = torch.tensor(PREDICTIONS, requires_grad=True)
    target = torch.tensor([[0.1, 0.9], [0.7, 0.3]], dtype=predictions.dtype, requires_grad=True)
    result = inner_product_score_func(predictions, target)
    result.sum().backward()
    torch.testing.assert_close(predictions.grad, -target.detach().expand_as(predictions))
    torch.testing.assert_close(target.grad, -predictions.detach().sum(dim=0))


@pytest.mark.parametrize(
    "score", [tv_score_func, wasserstein_distance_score_func, inner_product_score_func, kl_divergence_score_func]
)
@pytest.mark.parametrize("wrapper", ["sample", "categorical", "nested"])
def test_classification_representations(score, wrapper: str) -> None:
    probabilities = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    labels = torch.tensor([1, 2])
    distribution = TorchLogitCategoricalDistribution(torch.log(probabilities))
    if wrapper == "sample":
        prediction = TorchSample(probabilities, sample_dim=0)
    elif wrapper == "categorical":
        prediction = distribution
    else:
        prediction = TorchSample(distribution, sample_dim=0)
    result = score(prediction, labels)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, score(probabilities, labels))


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("score", [inner_product_score_func, kl_divergence_score_func])
def test_label_and_distribution_batching(score, batch_shape: tuple[int, ...]) -> None:
    probabilities = torch.tensor([0.2, 0.5, 0.3]).expand(*batch_shape, 3)
    labels = torch.ones(batch_shape, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels, 3).float()
    expected = torch.full(batch_shape, 0.5 if score is inner_product_score_func else -np.log(0.5))
    torch.testing.assert_close(score(probabilities, labels), expected)
    torch.testing.assert_close(score(probabilities, one_hot), expected)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("sample_dim", [0, 1])
def test_dirichlet_relative_likelihood_samples(nested: bool, sample_dim: int) -> None:
    alphas = torch.tensor([[[1.0, 2.0, 4.0], [3.0, 6.0, 2.0]], [[2.0, 4.0, 8.0], [6.0, 12.0, 4.0]]], requires_grad=True)
    labels = torch.tensor([[1, 0], [2, 1]])
    values = TorchDirichletDistribution(alphas) if nested else alphas
    sample = TorchSample(values, sample_dim=sample_dim, weights=torch.tensor([0.25, 0.75]))
    result = dirichlet_rl_score_func(sample, labels)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([[0.5, 0.5], [0.0, 0.0]]))
    result.sum().backward()
    assert alphas.grad is not None
    assert torch.isfinite(alphas.grad).all()
