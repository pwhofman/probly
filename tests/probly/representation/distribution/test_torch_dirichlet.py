"""Tests for torch-based Dirichlet distribution representation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from probly.decider import categorical_from_mean  # noqa: E402
from probly.representation.distribution import create_dirichlet_distribution_from_alphas  # noqa: E402
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution  # noqa: E402
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402


def test_torch_dirichlet_initialization_valid() -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]])

    distribution = TorchDirichletDistribution(alphas)

    assert distribution.alphas is alphas
    assert distribution.shape == (2,)
    assert distribution.ndim == 1
    assert distribution.size() == torch.Size([2])


def test_torch_dirichlet_factory_from_alphas() -> None:
    alphas = torch.tensor([[1.0, 2.0, 3.0]])

    distribution = create_dirichlet_distribution_from_alphas(alphas)

    assert isinstance(distribution, TorchDirichletDistribution)
    assert distribution.alphas is alphas


@pytest.mark.parametrize("invalid_value", [0.0, -0.1, -5.0])
def test_torch_dirichlet_raises_on_non_positive_alphas(invalid_value: float) -> None:
    alphas = torch.tensor([1.0, invalid_value, 2.0])

    with pytest.raises(ValueError, match="alphas must be strictly positive"):
        TorchDirichletDistribution(alphas)


def test_categorical_from_mean_reduces_torch_dirichlet_to_expected_categorical_distribution() -> None:
    distribution = TorchDirichletDistribution(torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]]))

    single = categorical_from_mean(distribution)

    assert isinstance(single, TorchCategoricalDistribution)
    assert torch.allclose(single.probabilities, torch.tensor([[1 / 6, 2 / 6, 3 / 6], [2 / 8, 2 / 8, 4 / 8]]))


def test_torch_dirichlet_sample_returns_categorical_distribution_sample() -> None:
    distribution = TorchDirichletDistribution(torch.tensor([1.0, 2.0, 3.0]))

    sample = distribution.sample(num_samples=4)

    assert isinstance(sample, TorchSample)
    assert isinstance(sample.tensor, TorchCategoricalDistribution)
    assert sample.tensor.unnormalized_probabilities.shape == (4, 3)
    assert sample.sample_axis == 0
    assert torch.allclose(sample.tensor.probabilities.sum(dim=-1), torch.ones(4))


def test_torch_dirichlet_torch_operations_preserve_protected_class_axis() -> None:
    alphas = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1.0
    distribution = TorchDirichletDistribution(alphas)

    meaned = torch.mean(distribution, dim=0)

    assert isinstance(meaned, TorchDirichletDistribution)
    assert meaned.shape == (3,)
    assert meaned.alphas.shape == (3, 4)
    assert torch.allclose(meaned.alphas, torch.mean(alphas, dim=0))


def _torch_modules():
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


class TestTorchDirichletDistribution:
    """Validation and properties for the torch Dirichlet distribution."""

    def test_alphas_must_be_positive(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="strictly positive"):
            TorchDirichletDistribution(alphas=torch.tensor([0.0, 1.0, 2.0]))

    def test_alphas_must_be_at_least_two_classes(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least 2 classes"):
            TorchDirichletDistribution(alphas=torch.tensor([1.0]))

    def test_alphas_zero_dim_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            TorchDirichletDistribution(alphas=torch.tensor(1.0))

    def test_alphas_must_be_tensor(self) -> None:
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        with pytest.raises(TypeError, match="torch tensor"):
            TorchDirichletDistribution(alphas=[1.0, 1.0])  # type: ignore[arg-type]

    def test_from_tensor_classmethod(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution.from_tensor([1.0, 2.0, 3.0])
        assert d.alphas.shape == (3,)
        # dtype-aware variant.
        d2 = TorchDirichletDistribution.from_tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert d2.alphas.dtype == torch.float64

    def test_eq_with_dirichlet(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        a = TorchDirichletDistribution(alphas=torch.tensor([1.0, 2.0]))
        b = TorchDirichletDistribution(alphas=torch.tensor([1.0, 2.0]))
        assert bool((a == b).all())

    def test_eq_with_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        a = TorchDirichletDistribution(alphas=torch.tensor([1.0, 2.0]))
        eq = a == torch.tensor([1.0, 2.0])
        assert bool(eq.all())

    def test_hash(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        a = TorchDirichletDistribution(alphas=torch.tensor([1.0, 2.0]))
        assert isinstance(hash(a), int)

    def test_sample_shape(self) -> None:
        torch = _torch_modules()
        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([1.0, 1.0, 1.0]))
        s = d.sample(num_samples=5)
        assert s.tensor.tensor.shape == (5, 3)
        assert s.sample_dim == 0

    def test_numpy_conversion(self) -> None:
        torch = _torch_modules()
        import numpy as np  # noqa: PLC0415

        from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution  # noqa: PLC0415

        d = TorchDirichletDistribution(alphas=torch.tensor([1.0, 2.0]))
        arr = d.numpy()
        np.testing.assert_allclose(arr, [1.0, 2.0])
