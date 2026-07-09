"""Tests for the torch Variational Bayesian Last Layer (VBLL) method."""

from __future__ import annotations

import pytest

from probly.method.vbll import find_vbll_layer, vbll
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
    from probly.train.vbll.torch import het_vbll_loss, t_vbll_loss  # noqa: PLC0415

    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))
    for layer, loss_fn in (
        (TVBLLLayer(8, 3, parameterization=parameterization), t_vbll_loss),
        (HetVBLLLayer(8, 3, parameterization=parameterization), het_vbll_loss),
    ):
        loss = loss_fn(layer, features, targets, regularization_weight=1.0 / 16)
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
    from probly.train.vbll.torch import vbll_loss  # noqa: PLC0415

    layer = VBLLLayer(8, 3, parameterization=parameterization)
    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))

    loss = vbll_loss(layer, features, targets, regularization_weight=1.0 / 16)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()


def test_vbll_keeps_non_trailing_softmax() -> None:
    # A softmax that is not the final layer must be left in place; only a trailing one is dropped.
    model = nn.Sequential(nn.Linear(10, 8), nn.Softmax(dim=-1), nn.Linear(8, 3))

    predictor = vbll(model)
    modules = list(predictor)

    from probly.layers.torch import VBLLLayer  # noqa: PLC0415

    assert isinstance(modules[2], VBLLLayer)
    assert isinstance(modules[1], nn.Softmax)
    assert isinstance(modules[0], nn.Linear)


@pytest.mark.parametrize("variant", VARIANTS)
def test_find_vbll_layer_returns_the_swapped_layer(variant: str) -> None:
    from probly.layers.torch import HetVBLLLayer, TVBLLLayer, VBLLLayer  # noqa: PLC0415

    predictor = vbll(_regression_model(out_features=3), variant=variant)

    layer = find_vbll_layer(predictor)

    expected = {"discriminative": VBLLLayer, "student_t": TVBLLLayer, "heteroscedastic": HetVBLLLayer}[variant]
    assert isinstance(layer, expected)
    assert layer is list(predictor)[2]


def test_find_vbll_layer_raises_without_vbll_layer() -> None:
    with pytest.raises(ValueError, match="No layer"):
        find_vbll_layer(_regression_model())


def test_compute_vbll_categorical_sample_rejects_unsupported_type() -> None:
    from probly.method.vbll._common import compute_vbll_categorical_sample  # noqa: PLC0415

    with pytest.raises(NotImplementedError, match="compute_vbll_categorical_sample"):
        compute_vbll_categorical_sample(object())


def test_vbll_layer_rejects_invalid_arguments() -> None:
    from probly.layers.torch import VBLLLayer  # noqa: PLC0415

    with pytest.raises(ValueError, match="parameterization"):
        VBLLLayer(8, 3, parameterization="banana")
    with pytest.raises(ValueError, match="prior_scale"):
        VBLLLayer(8, 3, prior_scale=0.0)
    with pytest.raises(ValueError, match="noise_init"):
        VBLLLayer(8, 3, noise_init=-1.0)


def test_t_vbll_layer_rejects_invalid_arguments() -> None:
    from probly.layers.torch import TVBLLLayer  # noqa: PLC0415

    with pytest.raises(ValueError, match="diagonal"):
        TVBLLLayer(8, 3, parameterization="lowrank")
    with pytest.raises(ValueError, match="dof"):
        TVBLLLayer(8, 3, dof=1.0)


def test_het_vbll_layer_rejects_invalid_parameterization() -> None:
    from probly.layers.torch import HetVBLLLayer  # noqa: PLC0415

    with pytest.raises(ValueError, match="diagonal"):
        HetVBLLLayer(8, 3, parameterization="lowrank")
