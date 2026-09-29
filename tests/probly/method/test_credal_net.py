"""Tests for ``probly.method.credal_net``."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestCredalNetMethod:
    """``credal_net`` is a wrapper over ``interval_classifier``."""

    @pytest.mark.parametrize("use_base_weights", [False, True])
    def test_credal_net_radii_train_and_intervals_keep_a_width(self, use_base_weights: bool) -> None:
        """Every interval radius of a credal net receives a gradient and changes, and the intervals keep a width."""
        torch, nn = _torch_nn()
        from probly.layers.torch import IntConv2d, IntLinear  # noqa: PLC0415
        from probly.losses.torch import intersection_probability_ce_loss  # noqa: PLC0415
        from probly.method.credal_net import credal_net  # noqa: PLC0415
        from probly.predictor import LogitClassifier, predict, predict_raw  # noqa: PLC0415

        torch.manual_seed(0)
        x = torch.rand(32, 1, 4, 4)
        y = torch.arange(32) % 3
        base = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        model = credal_net(base, predictor_type=LogitClassifier, use_base_weights=use_base_weights)
        layers = [module for module in model.modules() if isinstance(module, (IntConv2d, IntLinear))]
        assert len(layers) == 3
        radii_before = [(layer.radius_weight.detach().clone(), layer.radius_bias.detach().clone()) for layer in layers]

        model.train()
        intersection_probability_ce_loss(predict_raw(model, x), y).backward()
        for layer in layers:
            assert torch.any(layer.radius_weight.grad != 0)
            assert torch.any(layer.radius_bias.grad != 0)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            intersection_probability_ce_loss(predict_raw(model, x), y).backward()
            optimizer.step()
        for layer, (weight_before, bias_before) in zip(layers, radii_before, strict=True):
            assert not torch.equal(layer.radius_weight.detach(), weight_before)
            assert not torch.equal(layer.radius_bias.detach(), bias_before)
        model.eval()
        with torch.no_grad():
            credal_set = predict(model, x)
        # The forward pass sets negative radii to zero before using them.
        for layer in layers:
            assert torch.all(layer.radius_weight >= 0)
            assert torch.all(layer.radius_bias >= 0)
        assert torch.all(credal_set.upper() - credal_set.lower() > 1e-4)

    def test_credal_net_transforms_classifier(self) -> None:
        torch, nn = _torch_nn()
        from probly.method.credal_net import credal_net  # noqa: PLC0415
        from probly.predictor import LogitClassifier  # noqa: PLC0415

        base = nn.Sequential(nn.Linear(4, 3))
        net = credal_net(base, predictor_type=LogitClassifier)
        x = torch.randn(2, 4)
        from probly.predictor import predict  # noqa: PLC0415

        result = predict(net, x)
        assert result is not None
