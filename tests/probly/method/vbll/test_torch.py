"""Tests for the torch Variational Bayesian Last Layer (VBLL) method."""

from __future__ import annotations

import pytest

from probly.method.vbll import vbll
from probly.predictor import predict
from probly.quantification import decompose
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition
from probly.representation.distribution import GaussianDistribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

PARAMETERIZATIONS = ["diagonal", "dense", "lowrank"]


def _regression_model(out_features: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, out_features))


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_vbll_regression_predict_returns_gaussian(parameterization: str) -> None:
    predictor = vbll(_regression_model(), parameterization=parameterization)

    distribution = predict(predictor, torch.ones(4, 10))

    assert isinstance(distribution, GaussianDistribution)
    assert distribution.mean.shape == (4, 1)
    assert distribution.var.shape == (4, 1)
    assert torch.all(distribution.var > 0)


VARIANTS = ["discriminative", "student_t", "heteroscedastic"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("parameterization", ["diagonal", "dense"])
def test_vbll_variants_predict_and_represent(variant: str, parameterization: str) -> None:
    predictor = vbll(_regression_model(out_features=3), variant=variant, parameterization=parameterization)

    distribution = predict(predictor, torch.ones(4, 10))
    assert isinstance(distribution, GaussianDistribution)
    assert distribution.var.shape == (4, 3)
    assert torch.all(distribution.var > 0)

    sample = representer(predictor, num_samples=6).represent(torch.ones(4, 10))
    assert isinstance(sample, TorchCategoricalDistributionSample)


def test_vbll_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        vbll(_regression_model(), variant="banana")


@pytest.mark.parametrize("parameterization", ["diagonal", "dense"])
def test_vbll_t_and_het_layer_train_loss(parameterization: str) -> None:
    from probly.layers.torch import HetVBLLLayer, TVBLLLayer  # noqa: PLC0415

    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))
    for layer in (
        TVBLLLayer(8, 3, parameterization=parameterization),
        HetVBLLLayer(8, 3, parameterization=parameterization),
    ):
        loss = layer.train_loss(features, targets, regularization_weight=1.0 / 16)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()


def test_vbll_replaces_last_linear_and_drops_softmax() -> None:
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))

    predictor = vbll(model)
    modules = list(predictor)

    from probly.layers.torch import VBLLLayer  # noqa: PLC0415

    assert isinstance(modules[2], VBLLLayer)
    assert isinstance(modules[3], nn.Identity)
    # The first linear layer must be left untouched.
    assert isinstance(modules[0], nn.Linear)


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_vbll_representer_returns_categorical_sample(parameterization: str) -> None:
    predictor = vbll(_regression_model(out_features=3), parameterization=parameterization)

    sample = representer(predictor, num_samples=8).represent(torch.ones(4, 10))

    assert isinstance(sample, TorchCategoricalDistributionSample)


def test_vbll_decomposition_is_additive() -> None:
    predictor = vbll(_regression_model(out_features=3))

    sample = representer(predictor, num_samples=16).represent(torch.ones(4, 10))
    decomposition = decompose(sample)

    assert isinstance(decomposition, SecondOrderEntropyDecomposition)
    total = decomposition.total
    aleatoric = decomposition.aleatoric
    epistemic = decomposition.epistemic
    assert torch.all(total >= 0)
    assert torch.all(aleatoric >= 0)
    assert torch.all(epistemic >= -1e-5)
    assert torch.allclose(total, aleatoric + epistemic, atol=1e-4)


def test_vbll_rejects_unknown_parameterization() -> None:
    with pytest.raises(ValueError, match="parameterization"):
        vbll(_regression_model(), parameterization="banana")


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_vbll_layer_kl_divergence_is_finite_scalar(parameterization: str) -> None:
    from probly.layers.torch import VBLLLayer  # noqa: PLC0415

    layer = VBLLLayer(8, 3, parameterization=parameterization)
    kl = layer.kl_divergence

    assert kl.ndim == 0
    assert torch.isfinite(kl)
    assert kl.requires_grad


@pytest.mark.parametrize("parameterization", PARAMETERIZATIONS)
def test_vbll_layer_train_loss(parameterization: str) -> None:
    from probly.layers.torch import VBLLLayer  # noqa: PLC0415

    layer = VBLLLayer(8, 3, parameterization=parameterization)
    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))

    loss = layer.train_loss(features, targets, regularization_weight=1.0 / 16)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
