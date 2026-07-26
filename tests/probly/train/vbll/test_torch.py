from __future__ import annotations

import pytest

from probly.train.vbll import vbll_loss

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.layers.torch import GVBLLLayer, HetVBLLLayer, TVBLLLayer, VBLLLayer  # noqa: E402


def test_vbll_loss_dispatches_to_specific_losses() -> None:
    from probly.train.vbll.torch import disc_vbll_loss, g_vbll_loss, het_vbll_loss, t_vbll_loss  # noqa: PLC0415

    torch.manual_seed(0)
    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))
    regularization_weight = 1.0 / 16

    cases = [
        (VBLLLayer(8, 3), disc_vbll_loss),
        (TVBLLLayer(8, 3), t_vbll_loss),
        (HetVBLLLayer(8, 3), het_vbll_loss),
        (GVBLLLayer(8, 3), g_vbll_loss),
    ]
    for layer, specific_loss in cases:
        generic = vbll_loss(layer, features, targets, regularization_weight)
        specific = specific_loss(layer, features, targets, regularization_weight)
        assert torch.allclose(generic, specific)


def test_disc_vbll_loss_regularizes_the_noise() -> None:
    # The Wishart term must enter the loss so the learnable noise cannot collapse.
    from probly.train.vbll.torch import disc_vbll_loss  # noqa: PLC0415

    torch.manual_seed(0)
    layer = VBLLLayer(8, 3, wishart_scale=1.0)
    features = torch.randn(16, 8)
    targets = torch.randint(0, 3, (16,))

    loss = disc_vbll_loss(layer, features, targets, regularization_weight=1.0 / 16)
    layer.wishart_scale = 100.0
    reweighted_loss = disc_vbll_loss(layer, features, targets, regularization_weight=1.0 / 16)

    assert not torch.allclose(loss, reweighted_loss)
    assert torch.isfinite(reweighted_loss)


def test_vbll_loss_raises_for_unsupported_layer() -> None:
    features = torch.randn(4, 8)
    targets = torch.randint(0, 3, (4,))

    with pytest.raises(NotImplementedError, match="vbll_loss is not implemented"):
        vbll_loss(nn.Linear(8, 3), features, targets, 1.0)
