"""Tests for ordinal decompositions on PyTorch representations."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from probly.quantification import (  # noqa: E402
    CategoricalVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    SecondOrderVarianceDecomposition,
)
from probly.quantification.measure.ordinal import labelwise_entropy, labelwise_variance  # noqa: E402
from probly.representation.distribution.array_categorical import (  # noqa: E402
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.representation.distribution.torch_categorical import (  # noqa: E402
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.representation.distribution.torch_gaussian import (  # noqa: E402
    TorchGaussianDistribution,
    TorchGaussianDistributionSample,
)


def _categorical_sample() -> TorchCategoricalDistributionSample:
    probs = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    return TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probs),
        sample_dim=0,
    )


def _constant_categorical_sample() -> TorchCategoricalDistributionSample:
    probs = torch.tensor(
        [
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
            [[0.70, 0.20, 0.10]],
        ],
        dtype=torch.float64,
    )
    return TorchCategoricalDistributionSample(
        tensor=TorchCategoricalDistribution(probs),
        sample_dim=0,
    )


def _gaussian_sample() -> TorchGaussianDistributionSample:
    gaussians = [
        TorchGaussianDistribution(
            mean=torch.tensor([m], dtype=torch.float64),
            var=torch.tensor([0.5], dtype=torch.float64),
        )
        for m in [1.0, 2.0, 3.0]
    ]
    return TorchGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


def _identical_gaussian_sample() -> TorchGaussianDistributionSample:
    gaussians = [
        TorchGaussianDistribution(
            mean=torch.tensor([2.0], dtype=torch.float64),
            var=torch.tensor([0.5], dtype=torch.float64),
        )
    ] * 3
    return TorchGaussianDistributionSample.from_iterable(gaussians, sample_axis=0)


CATEGORICAL_DECOMP_CLASSES = [
    OrdinalEntropyDecomposition,
    OrdinalVarianceDecomposition,
    LabelwiseBinaryEntropyDecomposition,
    LabelwiseBinaryVarianceDecomposition,
    CategoricalVarianceDecomposition,
]


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_torch_categorical_decomposition_shapes(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert d.total.shape == (2,)
    assert d.aleatoric.shape == (2,)
    assert d.epistemic.shape == (2,)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_torch_categorical_decomposition_is_additive(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    torch.testing.assert_close(d.total, d.aleatoric + d.epistemic)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_torch_categorical_decomposition_epistemic_nonnegative(cls) -> None:
    sample = _categorical_sample()
    d = cls(sample)
    assert torch.all(d.epistemic >= -1e-10)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_torch_constant_ensemble_has_zero_epistemic(cls) -> None:
    sample = _constant_categorical_sample()
    d = cls(sample)
    torch.testing.assert_close(d.epistemic, torch.zeros_like(d.epistemic), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("cls", CATEGORICAL_DECOMP_CLASSES)
def test_torch_results_match_numpy(cls) -> None:
    np_sample_probs = np.array(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=float,
    )
    np_sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(np_sample_probs),
        sample_axis=0,
    )
    torch_sample = _categorical_sample()

    np_d = cls(np_sample)
    torch_d = cls(torch_sample)

    np.testing.assert_allclose(np_d.total, torch_d.total.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np_d.aleatoric, torch_d.aleatoric.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np_d.epistemic, torch_d.epistemic.numpy(), rtol=1e-10, atol=1e-10)


def test_torch_gaussian_variance_decomposition_values() -> None:
    sample = _gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)

    expected_au = torch.tensor([0.5], dtype=torch.float64)
    expected_eu = torch.tensor([np.var([1.0, 2.0, 3.0], ddof=0)], dtype=torch.float64)

    torch.testing.assert_close(d.aleatoric, expected_au)
    torch.testing.assert_close(d.epistemic, expected_eu)
    torch.testing.assert_close(d.total, expected_au + expected_eu)


def test_torch_gaussian_variance_decomposition_is_additive() -> None:
    sample = _gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)
    torch.testing.assert_close(d.total, d.aleatoric + d.epistemic)


def test_torch_identical_gaussian_has_zero_epistemic() -> None:
    sample = _identical_gaussian_sample()
    d = SecondOrderVarianceDecomposition(sample)
    torch.testing.assert_close(d.epistemic, torch.zeros_like(d.epistemic), atol=1e-12, rtol=0.0)


def test_torch_ordinal_entropy_with_log_base() -> None:
    sample = _categorical_sample()
    d_nats = OrdinalEntropyDecomposition(sample, base=None)
    d_bits = OrdinalEntropyDecomposition(sample, base=2)
    d_norm = OrdinalEntropyDecomposition(sample, base="normalize")

    torch.testing.assert_close(d_bits.total, d_nats.total / torch.log(torch.tensor(2.0, dtype=torch.float64)))
    torch.testing.assert_close(d_norm.total, d_bits.total)


def _torch_bh(p: torch.Tensor) -> torch.Tensor:
    p_c = torch.stack([p, 1.0 - p], dim=-1)
    return -torch.sum(torch.xlogy(p_c, p_c), dim=-1)


def test_torch_labelwise_entropy_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.tensor.probabilities  # (M=3, N=2, K=3)
    axis = sample.sample_axis
    p_bar = torch.mean(p, dim=axis)  # (N=2, K=3)

    expected_tu = torch.sum(_torch_bh(p_bar), dim=-1)
    expected_au = torch.mean(torch.sum(_torch_bh(p), dim=-1), dim=axis)

    d = LabelwiseBinaryEntropyDecomposition(sample)
    torch.testing.assert_close(d.total, expected_tu)
    torch.testing.assert_close(d.aleatoric, expected_au)


def test_torch_labelwise_variance_vs_manual_formula() -> None:
    sample = _categorical_sample()
    p = sample.tensor.probabilities  # (M, N, K)
    axis = sample.sample_axis
    p_bar = torch.mean(p, dim=axis)  # (N, K)

    expected_tu = torch.sum(p_bar * (1.0 - p_bar), dim=-1)
    expected_au = torch.mean(torch.sum(p * (1.0 - p), dim=-1), dim=axis)

    d = LabelwiseBinaryVarianceDecomposition(sample)
    torch.testing.assert_close(d.total, expected_tu)
    torch.testing.assert_close(d.aleatoric, expected_au)


def test_torch_labelwise_single_distribution_measures() -> None:
    probs = torch.tensor([[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]], dtype=torch.float64)
    dist = TorchCategoricalDistribution(probs)
    p = dist.probabilities  # (N=2, K=3)

    torch.testing.assert_close(labelwise_entropy(dist), torch.sum(_torch_bh(p), dim=-1))
    torch.testing.assert_close(labelwise_variance(dist), torch.sum(p * (1.0 - p), dim=-1))


def test_torch_labelwise_entropy_with_log_base() -> None:
    sample = _categorical_sample()
    d_nats = LabelwiseBinaryEntropyDecomposition(sample, base=None)
    d_bits = LabelwiseBinaryEntropyDecomposition(sample, base=2)
    d_norm = LabelwiseBinaryEntropyDecomposition(sample, base="normalize")

    torch.testing.assert_close(d_bits.total, d_nats.total / torch.log(torch.tensor(2.0, dtype=torch.float64)))
    torch.testing.assert_close(d_norm.total, d_bits.total)
