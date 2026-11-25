"""Tests for the variance measures on torch representations."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.quantification.measure.variance import (  # noqa: E402
    expected_conditional_variance,
    variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)
from probly.representation.distribution.torch_gaussian import (  # noqa: E402
    TorchGaussianDistribution,
    TorchGaussianDistributionSample,
)
from probly.representation.sample.torch import TorchSample  # noqa: E402

# Three members for a batch of two: the member means have population variance 8/3 in both columns.
MEANS = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]], dtype=torch.float64)
VARIANCES = torch.tensor([[1.0, 1.0], [2.0, 3.0], [3.0, 5.0]], dtype=torch.float64)
EPISTEMIC = torch.tensor([8 / 3, 8 / 3], dtype=torch.float64)


def test_gaussian_variance_accepts_base() -> None:
    """The ``base`` argument of the facade is accepted, as it is for the other backends."""
    distribution = TorchGaussianDistribution(mean=MEANS, var=VARIANCES)
    assert torch.equal(variance(distribution, base=2.0), VARIANCES)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_gaussian_sample_variance_decomposes_by_the_law_of_total_variance(sample_axis: int) -> None:
    gaussians = TorchGaussianDistribution(mean=MEANS.movedim(0, sample_axis), var=VARIANCES.movedim(0, sample_axis))
    sample = TorchGaussianDistributionSample(tensor=gaussians, sample_dim=sample_axis)
    aleatoric = torch.tensor([2.0, 3.0], dtype=torch.float64)
    assert torch.allclose(expected_conditional_variance(sample), aleatoric)
    assert torch.allclose(variance_of_conditional_mean(sample), EPISTEMIC)
    assert torch.allclose(variance_of_expected_predictive_distribution(sample), aleatoric + EPISTEMIC)


def test_point_prediction_sample_variance_is_all_epistemic() -> None:
    sample = TorchSample(tensor=MEANS, sample_dim=0)
    assert torch.allclose(variance_of_expected_predictive_distribution(sample), EPISTEMIC)
    assert torch.allclose(variance_of_conditional_mean(sample), EPISTEMIC)
    assert torch.equal(expected_conditional_variance(sample), torch.zeros(2, dtype=torch.float64))


def test_gaussian_sample_variance_is_differentiable() -> None:
    means = MEANS.clone().requires_grad_(True)
    sample = TorchGaussianDistributionSample(tensor=TorchGaussianDistribution(mean=means, var=VARIANCES), sample_dim=0)
    (grad,) = torch.autograd.grad(variance_of_expected_predictive_distribution(sample).sum(), means)
    # d/dm_i of the population variance of the member means is 2 (m_i - mean) / M.
    assert torch.allclose(grad, 2 * (MEANS - MEANS.mean(dim=0)) / 3)
