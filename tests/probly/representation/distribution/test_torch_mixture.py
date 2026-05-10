"""Tests for torch-based mixture distribution representation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.representation.distribution import MixtureDistribution  # noqa: E402
from probly.representation.distribution.torch_categorical import (  # noqa: E402
    TorchCategoricalDistribution,
    TorchProbabilityCategoricalDistribution,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution  # noqa: E402
from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402


def test_torch_mixture_initialization_valid() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    weights = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])

    distribution = TorchMixtureDistribution(components=components, mixture_weights=weights)

    assert isinstance(distribution, MixtureDistribution)
    assert distribution.type == "mixture"
    assert distribution.shape == (2,)
    assert distribution.ndim == 1
    torch.testing.assert_close(distribution.normalized_mixture_weights.sum(dim=-1), torch.ones(2))


def test_torch_mixture_rejects_shape_mismatch() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    weights = torch.ones(2, 4)

    with pytest.raises(ValueError, match="components shape must match mixture_weights shape"):
        TorchMixtureDistribution(components=components, mixture_weights=weights)


def test_torch_mixture_rejects_invalid_weights() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(3), var=torch.ones(3))

    with pytest.raises(ValueError, match="non-negative"):
        TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, -1.0, 1.0]))

    with pytest.raises(ValueError, match="positive sums"):
        TorchMixtureDistribution(components=components, mixture_weights=torch.zeros(3))


def test_torch_mixture_indexing_preserves_component_axis() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.ones(2, 3))

    sliced = distribution[0]

    assert isinstance(sliced, TorchMixtureDistribution)
    assert sliced.shape == ()
    assert sliced.components.shape == (3,)
    assert sliced.mixture_weights.shape == (3,)

    with pytest.raises(IndexError):
        _ = distribution[:, 0]


def test_torch_mixture_reshape_inserts_before_component_axis() -> None:
    components = TorchGaussianDistribution(mean=torch.zeros(2, 3), var=torch.ones(2, 3))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.ones(2, 3))

    reshaped = torch.reshape(distribution, (1, 2))

    assert isinstance(reshaped, TorchMixtureDistribution)
    assert reshaped.shape == (1, 2)
    assert reshaped.components.shape == (1, 2, 3)
    assert reshaped.mixture_weights.shape == (1, 2, 3)


def test_torch_mixture_sampling_categorical_components_matches_mixture_weights() -> None:
    components = TorchProbabilityCategoricalDistribution(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, 3.0]))
    rng = torch.Generator().manual_seed(0)

    sample = distribution.sample(num_samples=30_000, rng=rng)

    assert isinstance(sample, TorchSample)
    assert sample.sample_axis == 0
    assert sample.tensor.shape == (30_000,)
    assert sample.tensor.dtype == torch.int64

    counts = torch.bincount(sample.tensor, minlength=2).to(dtype=torch.float64)
    frequencies = counts / torch.sum(counts)
    torch.testing.assert_close(frequencies, torch.tensor([0.25, 0.75], dtype=torch.float64), atol=0.02, rtol=0.0)


def test_torch_mixture_sampling_dirichlet_components_preserves_class_axis() -> None:
    components = TorchDirichletDistribution(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    distribution = TorchMixtureDistribution(components=components, mixture_weights=torch.tensor([1.0, 1.0]))

    sample = distribution.sample(num_samples=4)

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, TorchCategoricalDistribution)
    assert sample.tensor.shape == (4,)
    assert sample.tensor.probabilities.shape == (4, 3)
    assert sample.sample_axis == 0
    torch.testing.assert_close(sample.tensor.probabilities.sum(dim=-1), torch.ones(4))


def test_torch_mixture_mean_averages_dirichlet_component_means_with_batch_weights() -> None:
    alphas = torch.tensor(
        [
            [[2.0, 1.0], [1.0, 3.0], [3.0, 1.0]],
            [[1.0, 5.0], [4.0, 2.0], [2.0, 2.0]],
        ]
    )
    weights = torch.tensor([[1.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    distribution = TorchMixtureDistribution(components=TorchDirichletDistribution(alphas), mixture_weights=weights)

    mean = distribution.mean

    component_means = alphas / alphas.sum(dim=-1, keepdim=True)
    expected = torch.sum(component_means * weights.unsqueeze(-1), dim=1) / torch.sum(weights, dim=1, keepdim=True)
    assert isinstance(mean, TorchCategoricalDistribution)
    assert mean.shape == (2,)
    torch.testing.assert_close(mean.probabilities, expected)


def _torch_modules():
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestTorchMixtureDistribution:
    """Validation and behaviour of TorchMixtureDistribution."""

    def test_components_must_be_distribution(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        with pytest.raises(TypeError, match="torch distribution"):
            TorchMixtureDistribution(components=torch.zeros(2, 3), mixture_weights=torch.tensor([0.5, 0.5]))  # type: ignore[arg-type]

    def test_mixture_weights_must_be_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(TypeError, match="torch tensor"):
            TorchMixtureDistribution(components=d, mixture_weights=[0.5, 0.5])  # type: ignore[arg-type]

    def test_mixture_weights_at_least_1d(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="at least one dimension"):
            TorchMixtureDistribution(components=d, mixture_weights=torch.tensor(0.5))

    def test_mixture_weights_must_be_finite(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="finite"):
            TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([float("inf"), 0.5]))

    def test_mixture_weights_must_be_non_negative(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="non-negative"):
            TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([-0.1, 1.1]))

    def test_mixture_weights_positive_sums(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="positive sums"):
            TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([0.0, 0.0]))

    def test_mixture_weights_shape_match_components(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        with pytest.raises(ValueError, match="components shape must match"):
            TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([0.5, 0.5, 0.5]))

    def test_normalized_mixture_weights_sum_to_one(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        m = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([1.0, 3.0]))
        normalized = m.normalized_mixture_weights
        torch.testing.assert_close(normalized.sum(dim=-1), torch.tensor(1.0))
        torch.testing.assert_close(normalized, torch.tensor([0.25, 0.75]))

    def test_hash(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
        m = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([1.0, 3.0]))
        assert isinstance(hash(m), int)


class TestTorchMixtureEquality:
    def test_eq_with_unrelated_type_returns_not_implemented(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
        m = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([[1.0, 3.0]]))
        # __eq__ returns NotImplemented for non-mixture types — Python's default
        # equality fallback then yields a regular boolean.
        result = m.__eq__("not-a-mixture")
        assert result is NotImplemented

    def test_eq_two_equal_mixtures(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
        m1 = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([[1.0, 3.0]]))
        m2 = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([[1.0, 3.0]]))
        result = m1 == m2
        # Returns a tensor of bool comparison results.
        assert torch.is_tensor(result)
        assert bool(result.all())


class TestTorchMixtureMean:
    def test_mean_property(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
        m = TorchMixtureDistribution(components=d, mixture_weights=torch.tensor([[0.5, 0.5]]))
        mean = m.mean
        # Mean is a TorchCategoricalDistribution.
        assert isinstance(mean, TorchCategoricalDistribution)


class TestCreateDirichletMixture:
    """The torch dispatch handler for the Dirichlet mixture factory."""

    def test_factory_constructs_torch_mixture(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution import (  # noqa: PLC0415
            create_dirichlet_mixture_distribution_from_alphas_and_weights,
        )
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        alphas = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
        weights = torch.tensor([[0.5, 0.5]])
        m = create_dirichlet_mixture_distribution_from_alphas_and_weights(alphas, weights)
        assert isinstance(m, TorchMixtureDistribution)


class TestTorchMixtureSample:
    """``TorchMixtureDistribution.sample``."""

    def test_sample_shape(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        # 2 components, 1 batch element, 3 classes.
        torch.manual_seed(0)
        d = TorchDirichletDistribution(alphas=torch.tensor([[[1.0, 1.0, 1.0], [10.0, 1.0, 1.0]]]))
        m = TorchMixtureDistribution(
            components=d,
            mixture_weights=torch.tensor([[0.5, 0.5]]),
        )
        sample = m.sample(num_samples=5)
        # Inner sample has shape (5, 1, 3) — categorical-distribution samples.
        assert sample.tensor.tensor.shape[0] == 5
        assert sample.sample_dim == 0

    def test_mean_property(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415
        from probly.representation.distribution.torch_mixture import TorchMixtureDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
        m = TorchMixtureDistribution(
            components=d,
            mixture_weights=torch.tensor([[0.5, 0.5]]),
        )
        mean = m.mean
        # The mean should be a TorchCategoricalDistribution.
        from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: PLC0415

        assert isinstance(mean, TorchCategoricalDistribution)
