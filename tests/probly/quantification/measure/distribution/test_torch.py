"""Tests for PyTorch distribution measures."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

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
from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistributionSample,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.distribution.torch_mixture import TorchMixtureDistribution

CATEGORICAL_BASES: tuple[None | float | str, ...] = (None, 2.0, "normalize")


def _base_divisor(
    base: None | float | str, num_classes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if base is None or base == torch.e:
        return torch.tensor(1.0, dtype=dtype, device=device)
    resolved_base = float(num_classes) if base == "normalize" else float(base)
    return torch.log(torch.tensor(resolved_base, dtype=dtype, device=device))


def _tol(base: None | float | str) -> tuple[float, float]:
    if base is None:
        return 1e-12, 1e-12
    return 1e-7, 1e-7


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_torch_categorical_entropy_matches_torch_distribution(base: None | float | str) -> None:
    probabilities = torch.tensor(
        [[0.25, 0.25, 0.5], [0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    distribution = TorchProbabilityCategoricalDistribution(probabilities)

    measured = entropy(distribution, base=base)
    expected_natural = Categorical(probs=probabilities).entropy()
    expected = expected_natural / _base_divisor(
        base,
        probabilities.shape[-1],
        dtype=expected_natural.dtype,
        device=expected_natural.device,
    )

    rtol, atol = _tol(base)
    assert torch.allclose(measured, expected, rtol=rtol, atol=atol)


def test_torch_categorical_entropy_normalize_maps_to_unit_interval() -> None:
    probabilities = torch.tensor(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    measured = entropy(TorchProbabilityCategoricalDistribution(probabilities), base="normalize")

    assert measured[0] == pytest.approx(1.0, abs=1e-6)
    assert measured[1] == pytest.approx(0.0, abs=1e-6)
    assert torch.all(measured >= 0.0)
    assert torch.all(measured <= 1.0)


@pytest.mark.parametrize("base", [None, 2.0])
def test_torch_dirichlet_entropy_matches_torch_distribution(base: None | float) -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas)

    measured = entropy(distribution, base=base)
    expected = torch.distributions.Dirichlet(alphas).entropy()
    if base is not None:
        expected = expected / torch.log(torch.tensor(base, dtype=expected.dtype))

    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
def test_torch_dirichlet_second_order_measures(base: None | float | str) -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0], [10.0, 5.0, 1.0]], dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas)

    measured_entropy_of_expected = entropy_of_expected_predictive_distribution(distribution, base=base)

    expected_mean = alphas / alphas.sum(dim=-1, keepdim=True)
    expected_entropy_of_expected = Categorical(probs=expected_mean).entropy()
    expected_conditional_entropy = torch.digamma(alphas.sum(dim=-1) + 1.0) - torch.sum(
        expected_mean * torch.digamma(alphas + 1.0), dim=-1
    )

    divisor = _base_divisor(
        base,
        alphas.shape[-1],
        dtype=expected_entropy_of_expected.dtype,
        device=expected_entropy_of_expected.device,
    )
    expected_entropy_of_expected = expected_entropy_of_expected / divisor
    if base == "normalize":
        assert torch.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=1e-7, atol=1e-7)
        with pytest.raises(ValueError, match="Entropy normalization is not supported for Dirichlet"):
            conditional_entropy(distribution, base=base)
        with pytest.raises(ValueError, match="Entropy normalization is not supported for Dirichlet"):
            mutual_information(distribution, base=base)
        return
    measured_conditional_entropy = conditional_entropy(distribution, base=base)
    measured_mutual_information = mutual_information(distribution, base=base)
    expected_conditional_entropy = expected_conditional_entropy / divisor
    expected_mutual_information = expected_entropy_of_expected - expected_conditional_entropy

    rtol, atol = _tol(base)
    assert torch.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=rtol, atol=atol)
    assert torch.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=rtol, atol=atol)
    assert torch.allclose(measured_mutual_information, expected_mutual_information, rtol=rtol, atol=atol)


@pytest.mark.parametrize("base", [None, 2.0])
def test_torch_dirichlet_mixture_second_order_measures(base: None | float) -> None:
    alphas = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 3.0], [3.0, 1.0]],
            [[1.0, 5.0], [4.0, 2.0], [2.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]], dtype=torch.float64)
    distribution = TorchMixtureDistribution(components=TorchDirichletDistribution(alphas), mixture_weights=weights)

    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    component_means = alphas / alphas.sum(dim=-1, keepdim=True)
    expected_mean = torch.sum(component_means * normalized_weights.unsqueeze(-1), dim=1)
    expected_entropy_of_expected = Categorical(probs=expected_mean).entropy()
    component_conditional_entropy = torch.digamma(alphas.sum(dim=-1) + 1.0) - torch.sum(
        component_means * torch.digamma(alphas + 1.0), dim=-1
    )
    expected_conditional_entropy = torch.sum(component_conditional_entropy * normalized_weights, dim=-1)

    if base is not None:
        divisor = torch.log(torch.tensor(base, dtype=torch.float64))
        expected_entropy_of_expected = expected_entropy_of_expected / divisor
        expected_conditional_entropy = expected_conditional_entropy / divisor

    expected_mutual_information = expected_entropy_of_expected - expected_conditional_entropy

    assert torch.allclose(
        entropy_of_expected_predictive_distribution(distribution, base=base), expected_entropy_of_expected
    )
    assert torch.allclose(conditional_entropy(distribution, base=base), expected_conditional_entropy)
    assert torch.allclose(mutual_information(distribution, base=base), expected_mutual_information)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_torch_categorical_second_order_measures_match_torch(sample_axis: int, base: None | float | str) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    measured_entropy_of_expected = entropy_of_expected_predictive_distribution(sample, base=base)
    measured_conditional_entropy = conditional_entropy(sample, base=base)
    measured_mutual_information = mutual_information(sample, base=base)

    expected_mean = torch.mean(probabilities, dim=sample_axis)
    expected_entropy_of_expected_natural = Categorical(probs=expected_mean).entropy()
    expected_conditional_entropy_natural = torch.mean(Categorical(probs=probabilities).entropy(), dim=sample_axis)
    expected_mutual_information_natural = torch.mean(
        kl_divergence(Categorical(probs=probabilities), Categorical(probs=expected_mean.unsqueeze(sample_axis))),
        dim=sample_axis,
    )

    divisor = _base_divisor(
        base,
        probabilities.shape[-1],
        dtype=expected_entropy_of_expected_natural.dtype,
        device=expected_entropy_of_expected_natural.device,
    )
    expected_entropy_of_expected = expected_entropy_of_expected_natural / divisor
    expected_conditional_entropy = expected_conditional_entropy_natural / divisor
    expected_mutual_information = expected_mutual_information_natural / divisor

    rtol, atol = _tol(base)
    assert torch.allclose(measured_entropy_of_expected, expected_entropy_of_expected, rtol=rtol, atol=atol)
    assert torch.allclose(measured_conditional_entropy, expected_conditional_entropy, rtol=rtol, atol=atol)
    assert torch.allclose(measured_mutual_information, expected_mutual_information, rtol=rtol, atol=atol)


@pytest.mark.parametrize("base", CATEGORICAL_BASES)
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_identity_holds_for_torch_categorical_sample(sample_axis: int, base: None | float | str) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    expected_entropy = entropy_of_expected_predictive_distribution(sample, base=base)
    decomposition_sum = conditional_entropy(sample, base=base) + mutual_information(sample, base=base)

    rtol, atol = _tol(base)
    assert torch.allclose(expected_entropy, decomposition_sum, rtol=rtol, atol=atol)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_torch_sample_zero_one_measures_match_manual(sample_axis: int) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    measured_total = max_probability_complement_of_expected(sample)
    measured_aleatoric = expected_max_probability_complement(sample)
    measured_epistemic = max_disagreement(sample)

    expected_mean = torch.mean(probabilities, dim=sample_axis)
    expected_total = 1.0 - torch.max(expected_mean, dim=-1).values
    expected_aleatoric = torch.mean(1.0 - torch.max(probabilities, dim=-1).values, dim=sample_axis)
    bma_argmax = torch.argmax(expected_mean, dim=-1).unsqueeze(sample_axis).unsqueeze(-1)
    per_sample_bma_prob = torch.take_along_dim(probabilities, bma_argmax, dim=-1).squeeze(-1)
    expected_epistemic = torch.mean(torch.max(probabilities, dim=-1).values - per_sample_bma_prob, dim=sample_axis)

    assert torch.allclose(measured_total, expected_total, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_aleatoric, expected_aleatoric, rtol=1e-12, atol=1e-12)
    assert torch.allclose(measured_epistemic, expected_epistemic, rtol=1e-12, atol=1e-12)


def test_torch_sample_zero_one_known_values() -> None:
    probabilities = torch.tensor(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ],
        dtype=torch.float64,
    )
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert max_probability_complement_of_expected(sample).item() == pytest.approx(0.45, abs=1e-12)
    assert expected_max_probability_complement(sample).item() == pytest.approx(0.15, abs=1e-12)
    assert max_disagreement(sample).item() == pytest.approx(0.30, abs=1e-12)


@pytest.mark.parametrize("sample_axis", [0, 1])
def test_zero_one_identity_holds_for_torch_categorical_sample(sample_axis: int) -> None:
    base_probabilities = torch.tensor(
        [
            [[0.70, 0.20, 0.10], [0.15, 0.35, 0.50]],
            [[0.60, 0.30, 0.10], [0.20, 0.30, 0.50]],
            [[0.80, 0.10, 0.10], [0.10, 0.40, 0.50]],
        ],
        dtype=torch.float64,
    )
    probabilities = torch.moveaxis(base_probabilities, 0, sample_axis)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=sample_axis,
    )

    total = max_probability_complement_of_expected(sample)
    aleatoric = expected_max_probability_complement(sample)
    epistemic = max_disagreement(sample)

    assert torch.allclose(total, aleatoric + epistemic, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_vacuity_known_values() -> None:
    alphas = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # uniform Dir(1,1,1): K=3, alpha_0=3 -> vacuity=1
            [10.0, 10.0, 10.0],  # K=3, alpha_0=30 -> vacuity=0.1
            [2.0, 3.0, 5.0],  # K=3, alpha_0=10 -> vacuity=0.3
        ],
        dtype=torch.float64,
    )
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = vacuity(distribution)

    expected = torch.tensor([1.0, 0.1, 0.3], dtype=torch.float64)
    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_vacuity_lies_in_unit_interval() -> None:
    generator = torch.Generator().manual_seed(0)
    alphas = 1.0 + 19.0 * torch.rand((50, 4), generator=generator, dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = vacuity(distribution)

    assert torch.all(measured > 0.0)
    assert torch.all(measured <= 1.0)


def test_torch_dirichlet_vacuity_decreases_with_evidence() -> None:
    weak = TorchDirichletDistribution(alphas=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    strong = TorchDirichletDistribution(alphas=torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64))

    assert vacuity(weak).item() > vacuity(strong).item()


def test_torch_dirichlet_vacuity_propagates_gradients() -> None:
    alphas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = vacuity(distribution)
    measured.backward()

    # d(K/alpha_0)/d(alpha_c) = -K / alpha_0^2 for each c
    grad = alphas.grad
    assert grad is not None
    expected_grad = -torch.full_like(alphas, 3.0 / (10.0**2))
    assert torch.allclose(grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_max_probability_complement_of_expected_known_values() -> None:
    alphas = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # uniform: max(1/3) -> 1 - 1/3 = 2/3
            [10.0, 1.0, 1.0],  # max = 10/12 -> 1 - 5/6 = 1/6
            [2.0, 3.0, 5.0],  # max = 5/10 -> 1 - 1/2 = 1/2
        ],
        dtype=torch.float64,
    )
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected = torch.tensor([2.0 / 3.0, 1.0 / 6.0, 0.5], dtype=torch.float64)
    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_max_probability_complement_of_expected_matches_explicit_formula() -> None:
    generator = torch.Generator().manual_seed(0)
    alphas = 0.5 + 19.5 * torch.rand((50, 5), generator=generator, dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = max_probability_complement_of_expected(distribution)

    expected_mean = alphas / alphas.sum(dim=-1, keepdim=True)
    expected = 1.0 - torch.max(expected_mean, dim=-1).values
    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_max_probability_complement_of_expected_propagates_gradients() -> None:
    alphas = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    distribution = TorchDirichletDistribution(alphas=alphas)

    measured = max_probability_complement_of_expected(distribution)
    measured.backward()

    grad = alphas.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_torch_gaussian_dempster_shafer_uniform_logits_with_default_factor() -> None:
    """Uniform-zero logits should give vacuity = K / (K + K * exp(0)) = 1/2."""
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: PLC0415

    mean = torch.zeros((3, 5), dtype=torch.float64)
    var = torch.ones_like(mean)
    distribution = TorchGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    assert torch.allclose(measured, torch.full((3,), 0.5, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_torch_gaussian_dempster_shafer_matches_explicit_formula() -> None:
    import math  # noqa: PLC0415

    from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: PLC0415

    generator = torch.Generator().manual_seed(0)
    mean = 2.0 * torch.randn((20, 5), generator=generator, dtype=torch.float64)
    var = 0.01 + 4.0 * torch.rand((20, 5), generator=generator, dtype=torch.float64)
    distribution = TorchGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)

    num_classes = mean.shape[-1]
    adjusted = mean / torch.sqrt(1.0 + (math.pi / 8.0) * var)
    expected = num_classes / (num_classes + torch.sum(torch.exp(adjusted), dim=-1))
    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_gaussian_dempster_shafer_propagates_gradients() -> None:
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: PLC0415

    mean = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    var = torch.tensor([[0.5, 1.0, 0.25]], dtype=torch.float64, requires_grad=True)
    distribution = TorchGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution)
    measured.sum().backward()

    assert mean.grad is not None
    assert var.grad is not None
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(var.grad).all()


def test_torch_gaussian_dempster_shafer_zero_factor_disables_mean_field() -> None:
    from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: PLC0415

    mean = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    var = torch.full_like(mean, 100.0)
    distribution = TorchGaussianDistribution(mean=mean, var=var)

    measured = dempster_shafer_uncertainty(distribution, mean_field_factor=0.0)

    expected = 3.0 / (3.0 + torch.sum(torch.exp(mean), dim=-1))
    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_sample_min_expected_total_variation_known_value_binary() -> None:
    """EU = 0.2 for the K=2 example where it differs from the zero-one EU (TU - AU = 0)."""
    probabilities = torch.tensor([[0.90, 0.10], [0.50, 0.50]], dtype=torch.float64)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert min_expected_total_variation(sample).item() == pytest.approx(0.2, abs=1e-9)


def test_torch_sample_min_expected_total_variation_known_value_ternary_constrained() -> None:
    """EU = 0.3 for a K=3 case where the simplex constraint binds."""
    probabilities = torch.tensor(
        [[0.70, 0.20, 0.10], [0.50, 0.40, 0.10], [0.10, 0.10, 0.80]],
        dtype=torch.float64,
    )
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert min_expected_total_variation(sample).item() == pytest.approx(0.3, abs=1e-9)


def test_torch_sample_min_expected_total_variation_is_zero_for_no_second_order_spread() -> None:
    """A second-order Dirac (all samples identical) has no epistemic uncertainty."""
    probabilities = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64).repeat(5, 1)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert min_expected_total_variation(sample).item() == pytest.approx(0.0, abs=1e-9)


def test_torch_sample_min_expected_total_variation_is_maximal_for_uniform_diracs() -> None:
    """EU attains the upper bound (K-1)/K for a uniform mixture of first-order Diracs."""
    probabilities = torch.eye(3, dtype=torch.float64)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    assert min_expected_total_variation(sample).item() == pytest.approx(2.0 / 3.0, abs=1e-9)


@pytest.mark.parametrize("sample_dim", [0, 1])
def test_torch_sample_min_expected_total_variation_matches_numpy(sample_dim: int) -> None:
    """The torch implementation matches the numpy implementation on random batched data."""
    import numpy as np  # noqa: PLC0415

    from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
        ArrayCategoricalDistributionSample,
        ArrayProbabilityCategoricalDistribution,
    )

    rng = np.random.default_rng(seed=0)
    logits = rng.normal(size=(4, 6, 3))
    base_probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    probabilities = np.moveaxis(base_probabilities, 1, sample_dim)

    torch_sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(torch.as_tensor(probabilities, dtype=torch.float64)),
        sample_dim=sample_dim,
    )
    array_sample = ArrayCategoricalDistributionSample(
        array=ArrayProbabilityCategoricalDistribution(probabilities),
        sample_axis=sample_dim,
    )

    measured = min_expected_total_variation(torch_sample).numpy()
    expected = min_expected_total_variation(array_sample)

    np.testing.assert_allclose(measured, expected, rtol=1e-9, atol=1e-9)


def test_torch_sample_min_expected_total_variation_differs_from_zero_one_epistemic() -> None:
    """The OT epistemic measure is genuinely distinct from the additive zero-one EU."""
    probabilities = torch.tensor([[0.90, 0.10], [0.50, 0.50]], dtype=torch.float64)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(probabilities),
        sample_dim=0,
    )

    wasserstein_eu = min_expected_total_variation(sample)
    zero_one_eu = max_disagreement(sample)

    assert not torch.allclose(wasserstein_eu, zero_one_eu)
    assert zero_one_eu.item() == pytest.approx(0.0, abs=1e-12)


def test_torch_dirichlet_min_expected_total_variation_delegates_to_sampling() -> None:
    """The Dirichlet EU draws Monte-Carlo samples and reuses the sample estimator."""
    alphas = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas)

    torch.manual_seed(0)
    measured = min_expected_total_variation(distribution, num_samples=500)
    torch.manual_seed(0)
    expected = min_expected_total_variation(distribution.sample(500))

    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_expected_max_probability_complement_delegates_to_sampling() -> None:
    """The Dirichlet aleatoric uncertainty draws Monte-Carlo samples and reuses the sample estimator."""
    alphas = torch.tensor([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    distribution = TorchDirichletDistribution(alphas)

    torch.manual_seed(0)
    measured = expected_max_probability_complement(distribution, num_samples=500)
    torch.manual_seed(0)
    expected = expected_max_probability_complement(distribution.sample(500))

    assert torch.allclose(measured, expected, rtol=1e-12, atol=1e-12)


def test_torch_dirichlet_distance_measures_concentrated_limits() -> None:
    """A near-uniform Dirichlet has EU ~ 0 and AU ~ (K-1)/K. A near-vertex one has both near 0."""
    torch.manual_seed(0)
    near_uniform = TorchDirichletDistribution(torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64))
    eu_uniform = min_expected_total_variation(near_uniform, num_samples=4000)
    au_uniform = expected_max_probability_complement(near_uniform, num_samples=4000)
    assert eu_uniform.item() == pytest.approx(0.0, abs=2e-2)
    assert au_uniform.item() == pytest.approx(2.0 / 3.0, abs=2e-2)

    near_vertex = TorchDirichletDistribution(torch.tensor([1000.0, 1.0, 1.0], dtype=torch.float64))
    eu_vertex = min_expected_total_variation(near_vertex, num_samples=4000)
    au_vertex = expected_max_probability_complement(near_vertex, num_samples=4000)
    assert eu_vertex.item() == pytest.approx(0.0, abs=2e-2)
    assert au_vertex.item() == pytest.approx(0.0, abs=2e-2)


def test_torch_dirichlet_distance_measures_warn_on_generator_but_still_run() -> None:
    """The torch Dirichlet sampler uses the global RNG, so a passed generator warns but is ignored."""
    distribution = TorchDirichletDistribution(torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64))
    generator = torch.Generator().manual_seed(0)

    with pytest.warns(UserWarning, match="generator is not used"):
        eu = min_expected_total_variation(distribution, num_samples=100, generator=generator)
    with pytest.warns(UserWarning, match="generator is not used"):
        au = expected_max_probability_complement(distribution, num_samples=100, generator=generator)

    assert torch.isfinite(eu).all()
    assert torch.isfinite(au).all()
