"""Tests for the torch Generative Variational Bayesian Last Layer (G-VBLL) method."""

from __future__ import annotations

import pytest

from probly.method.g_vbll import find_g_vbll_layer, g_vbll
from probly.predictor import predict
from probly.quantification import quantify
from probly.representation.distribution import CategoricalDistribution
from probly.representer import representer

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


def _regression_model(out_features: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, out_features))


def test_g_vbll_predict_returns_categorical() -> None:
    predictor = g_vbll(_regression_model(out_features=3))

    distribution = predict(predictor, torch.ones(4, 10))

    assert isinstance(distribution, CategoricalDistribution)
    assert distribution.num_classes == 3
    assert distribution.probabilities.shape == (4, 3)


def test_g_vbll_replaces_last_linear_and_drops_softmax() -> None:
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3), nn.Softmax(dim=-1))

    predictor = g_vbll(model)
    modules = list(predictor)

    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415

    assert isinstance(modules[2], GVBLLLayer)
    assert isinstance(modules[3], nn.Identity)
    assert isinstance(modules[0], nn.Linear)


def test_g_vbll_representer_quantifies_to_entropy() -> None:
    predictor = g_vbll(_regression_model(out_features=3))

    representation = representer(predictor).represent(torch.ones(4, 10))
    uncertainty = quantify(representation)

    assert torch.all(uncertainty.total >= 0)


def test_find_g_vbll_layer_returns_the_swapped_layer() -> None:
    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415

    predictor = g_vbll(_regression_model(out_features=3))

    layer = find_g_vbll_layer(predictor)

    assert isinstance(layer, GVBLLLayer)
    assert layer is list(predictor)[2]


def test_g_vbll_layer_train_loss_and_kl_are_finite() -> None:
    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415
    from probly.train.vbll.torch import g_vbll_loss  # noqa: PLC0415

    layer = GVBLLLayer(8, 3)
    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))

    loss = g_vbll_loss(layer, features, targets, regularization_weight=1.0 / 16)
    kl = layer.kl_divergence

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_g_vbll_keeps_non_trailing_softmax() -> None:
    # A softmax that is not the final layer must be left in place; only a trailing one is dropped.
    model = nn.Sequential(nn.Linear(10, 8), nn.Softmax(dim=-1), nn.Linear(8, 3))

    predictor = g_vbll(model)
    modules = list(predictor)

    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415

    assert isinstance(modules[2], GVBLLLayer)
    assert isinstance(modules[1], nn.Softmax)
    assert isinstance(modules[0], nn.Linear)


def test_g_vbll_layer_rejects_invalid_arguments() -> None:
    from probly.layers.torch import GVBLLLayer  # noqa: PLC0415

    with pytest.raises(ValueError, match="prior_scale"):
        GVBLLLayer(8, 3, prior_scale=0.0)
    with pytest.raises(ValueError, match="noise_init"):
        GVBLLLayer(8, 3, noise_init=-1.0)
