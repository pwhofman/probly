"""Tests for the torch backend of probly.transformation.interval_classifier."""

from __future__ import annotations

import pytest


def _torch_nn():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestIntervalClassifierEnd2End:
    """``interval_classifier`` builds a working interval-prediction model."""

    def test_transforms_simple_classifier(self) -> None:
        torch, nn = _torch_nn()
        from probly.predictor import LogitClassifier, predict  # noqa: PLC0415
        from probly.representation.credal_set import ProbabilityIntervalsCredalSet  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        base = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3))
        wrapped = interval_classifier(base, predictor_type=LogitClassifier)
        x = torch.randn(2, 8)
        out = predict(wrapped, x)
        assert isinstance(out, ProbabilityIntervalsCredalSet)

    def test_no_linear_raises(self) -> None:
        _, nn = _torch_nn()
        from probly.predictor import LogitClassifier  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        base = nn.Sequential(nn.ReLU(), nn.Tanh())
        with pytest.raises(ValueError, match="no Linear"):
            interval_classifier(base, predictor_type=LogitClassifier)

    def test_use_base_weights_copies_into_center(self) -> None:
        torch, nn = _torch_nn()
        from probly.layers.torch import IntLinear  # noqa: PLC0415
        from probly.predictor import LogitClassifier  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        base = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
        with torch.no_grad():
            base[0].weight.fill_(0.5)
            base[0].bias.fill_(0.25)
        wrapped = interval_classifier(base, use_base_weights=True, predictor_type=LogitClassifier)
        # Find IntLinears in the wrapped module tree.
        int_linears = [m for m in wrapped.modules() if isinstance(m, IntLinear)]
        # At least 2 IntLinears: one for the head, one for the inner Linear.
        assert len(int_linears) >= 2

    def test_interval_classifier_ignores_dilation(self) -> None:
        """Passing a Conv2d with non-trivial dilation should warn."""
        _, nn = _torch_nn()
        from probly.predictor import LogitClassifier  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        base = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, dilation=2),
            nn.Flatten(),
            nn.Linear(4, 3),
        )
        with pytest.warns(UserWarning, match="dilation"):
            interval_classifier(base, predictor_type=LogitClassifier)


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestIntervalClassifierWeightCopy:
    """The weight-copy paths in interval_classifier/torch.py."""

    def test_copy_bn1d_weights_into_center(self) -> None:
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.layers.torch import IntBatchNorm1d  # noqa: PLC0415
        from probly.transformation.interval_classifier import torch as _torch_register  # noqa: F401, PLC0415
        from probly.transformation.interval_classifier._common import (  # noqa: PLC0415
            REPLACED,
            USE_BASE_WEIGHTS,
            interval_classifier_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        bn = nn.BatchNorm1d(4, affine=True, track_running_stats=True)
        with torch.no_grad():
            bn.weight.fill_(2.0)
            bn.bias.fill_(0.5)
            bn.running_mean.fill_(0.1)  # ty: ignore[unresolved-attribute]
            bn.running_var.fill_(0.9)  # ty: ignore[unresolved-attribute]

        result, _ = traverse_with_state(
            bn,
            nn_compose(interval_classifier_traverser),
            init={REPLACED: True, USE_BASE_WEIGHTS: True},
        )
        assert isinstance(result, IntBatchNorm1d)
        torch.testing.assert_close(result.center_weight, bn.weight)
        torch.testing.assert_close(result.center_bias, bn.bias)

    def test_copy_bn2d_weights_into_center(self) -> None:
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.layers.torch import IntBatchNorm2d  # noqa: PLC0415
        from probly.transformation.interval_classifier import torch as _torch_register  # noqa: F401, PLC0415
        from probly.transformation.interval_classifier._common import (  # noqa: PLC0415
            REPLACED,
            USE_BASE_WEIGHTS,
            interval_classifier_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        bn = nn.BatchNorm2d(4, affine=True, track_running_stats=True)
        with torch.no_grad():
            bn.weight.fill_(2.0)
            bn.bias.fill_(0.5)
            bn.running_mean.fill_(0.1)  # ty: ignore[unresolved-attribute]
            bn.running_var.fill_(0.9)  # ty: ignore[unresolved-attribute]

        result, _ = traverse_with_state(
            bn,
            nn_compose(interval_classifier_traverser),
            init={REPLACED: True, USE_BASE_WEIGHTS: True},
        )
        assert isinstance(result, IntBatchNorm2d)
        torch.testing.assert_close(result.center_weight, bn.weight)
        torch.testing.assert_close(result.center_bias, bn.bias)

    def test_copy_conv2d_weights_into_center(self) -> None:
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.layers.torch import IntConv2d  # noqa: PLC0415
        from probly.transformation.interval_classifier import torch as _torch_register  # noqa: F401, PLC0415
        from probly.transformation.interval_classifier._common import (  # noqa: PLC0415
            REPLACED,
            USE_BASE_WEIGHTS,
            interval_classifier_traverser,
        )
        from probly.traverse_nn import nn_compose  # noqa: PLC0415
        from pytraverse import traverse_with_state  # noqa: PLC0415

        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        with torch.no_grad():
            conv.weight.fill_(0.5)
            conv.bias.fill_(0.25)  # ty: ignore[unresolved-attribute]

        result, _ = traverse_with_state(
            conv,
            nn_compose(interval_classifier_traverser),
            init={REPLACED: True, USE_BASE_WEIGHTS: True},
        )
        assert isinstance(result, IntConv2d)
        torch.testing.assert_close(result.center_weight, conv.weight)
        torch.testing.assert_close(result.center_bias, conv.bias)
        # Only the centers are copied; the radii keep their initialization.
        assert torch.all(result.radius_weight > 0.0)
        assert torch.all(result.radius_bias == 0.0)


class TestIntervalClassifierRadii:
    """The interval radii keep their initialization and train with and without ``use_base_weights``."""

    def test_use_base_weights_copies_only_the_centers(self) -> None:
        torch, nn = _torch_nn()
        from probly.layers.torch import IntConv2d, IntLinear  # noqa: PLC0415
        from probly.predictor import LogitClassifier  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        base = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
        )
        torch.manual_seed(0)
        fresh = interval_classifier(base, predictor_type=LogitClassifier, use_base_weights=False)
        torch.manual_seed(0)
        copied = interval_classifier(base, predictor_type=LogitClassifier, use_base_weights=True)

        interval_types = (IntConv2d, IntLinear)
        fresh_layers = [m for m in fresh.modules() if isinstance(m, interval_types)]
        copied_layers = [m for m in copied.modules() if isinstance(m, interval_types)]
        base_layers = [m for m in base.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        assert len(copied_layers) == len(base_layers) == 3
        for new, fresh_layer, old in zip(copied_layers, fresh_layers, base_layers, strict=True):
            assert torch.equal(new.center_weight, old.weight)
            assert torch.equal(new.center_bias, old.bias)
            # The copy does not touch the radii, so they equal the fresh draw under the same seed.
            assert torch.equal(new.radius_weight, fresh_layer.radius_weight)
            assert torch.equal(new.radius_bias, fresh_layer.radius_bias)
            assert torch.all(new.radius_weight > 0)
            assert torch.all(new.radius_bias == 0)

    @pytest.mark.parametrize("use_base_weights", [False, True])
    def test_bias_radii_at_exactly_zero_start_training(self, use_base_weights: bool) -> None:
        """Bias radii that are exactly 0, as in models whose bias radii never trained, change during training."""
        torch, nn = _torch_nn()
        from probly.layers.torch import IntLinear  # noqa: PLC0415
        from probly.losses.torch import intersection_probability_ce_loss  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        torch.manual_seed(0)
        x = torch.rand(64, 2) * 4.0
        y = (x[:, 0] > x[:, 1]).long() + (x.sum(dim=1) > 4.0).long()
        base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=1))
        model = interval_classifier(base, predictor_type="probabilistic_classifier", use_base_weights=use_base_weights)
        layers = [m for m in model.modules() if isinstance(m, IntLinear)]
        with torch.no_grad():
            for layer in layers:
                layer.radius_bias.zero_()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            intersection_probability_ce_loss(predict_raw(model, x), y).backward()
            optimizer.step()

        for layer in layers:
            assert torch.any(layer.radius_bias != 0)

    @pytest.mark.parametrize("use_base_weights", [False, True])
    def test_radii_train_and_intervals_keep_a_width(self, use_base_weights: bool) -> None:
        torch, nn = _torch_nn()
        from probly.layers.torch import IntLinear  # noqa: PLC0415
        from probly.losses.torch import intersection_probability_ce_loss  # noqa: PLC0415
        from probly.predictor import predict, predict_raw  # noqa: PLC0415
        from probly.transformation.interval_classifier import interval_classifier  # noqa: PLC0415

        torch.manual_seed(0)
        x = torch.rand(64, 2) * 4.0
        y = (x[:, 0] > x[:, 1]).long() + (x.sum(dim=1) > 4.0).long()
        base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3), nn.Softmax(dim=1))
        model = interval_classifier(base, predictor_type="probabilistic_classifier", use_base_weights=use_base_weights)
        layers = [m for m in model.modules() if isinstance(m, IntLinear)]
        before = [(m.radius_weight.detach().clone(), m.radius_bias.detach().clone()) for m in layers]

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            intersection_probability_ce_loss(predict_raw(model, x), y).backward()
            optimizer.step()

        for layer, (radius_weight, radius_bias) in zip(layers, before, strict=True):
            assert not torch.equal(layer.radius_weight, radius_weight)
            assert not torch.equal(layer.radius_bias, radius_bias)
        model.eval()
        with torch.no_grad():
            credal_set = predict(model, x)
        assert torch.all(credal_set.upper() - credal_set.lower() > 0)
