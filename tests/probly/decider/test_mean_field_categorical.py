"""Tests for the mean_field_categorical decider."""

from __future__ import annotations

import pytest

from probly.decider import mean_field_categorical

torch = pytest.importorskip("torch")

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: E402


def _torch_gaussian(mean: list[list[float]], var: list[list[float]]) -> TorchGaussianDistribution:
    return TorchGaussianDistribution(
        mean=torch.tensor(mean, dtype=torch.float32),
        var=torch.tensor(var, dtype=torch.float32),
    )


def test_mean_field_categorical_returns_torch_categorical() -> None:
    gaussian = _torch_gaussian([[1.0, 2.0, 3.0]], [[0.1, 0.1, 0.1]])

    result = mean_field_categorical(gaussian, mean_field_factor=1.0)

    assert isinstance(result, TorchCategoricalDistribution)


def test_mean_field_categorical_factor_zero_recovers_softmax_of_mean() -> None:
    gaussian = _torch_gaussian([[1.0, 2.0, 3.0]], [[5.0, 5.0, 5.0]])

    result = mean_field_categorical(gaussian, mean_field_factor=0.0)

    expected = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    assert torch.allclose(result.probabilities, expected, atol=1e-6)


def test_mean_field_categorical_default_factor_is_one() -> None:
    gaussian = _torch_gaussian([[1.0, 2.0, 3.0]], [[0.5, 0.5, 0.5]])

    result_default = mean_field_categorical(gaussian)
    result_explicit = mean_field_categorical(gaussian, mean_field_factor=1.0)

    assert torch.allclose(result_default.probabilities, result_explicit.probabilities)


def test_mean_field_categorical_larger_factor_increases_entropy() -> None:
    gaussian = _torch_gaussian([[3.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]])

    low = mean_field_categorical(gaussian, mean_field_factor=0.1)
    high = mean_field_categorical(gaussian, mean_field_factor=10.0)

    def entropy(p: torch.Tensor) -> torch.Tensor:
        return -torch.sum(p * torch.log(p + 1e-12), dim=-1)

    assert (entropy(high.probabilities) > entropy(low.probabilities)).all()


def test_mean_field_categorical_matches_closed_form_formula() -> None:
    mean_t = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float32)
    var_t = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32)
    gaussian = TorchGaussianDistribution(mean=mean_t, var=var_t)

    factor = 0.5
    result = mean_field_categorical(gaussian, mean_field_factor=factor)

    scale = (1.0 + factor * var_t) ** 0.5
    expected = torch.softmax(mean_t / scale, dim=-1)
    assert torch.allclose(result.probabilities, expected, atol=1e-6)
