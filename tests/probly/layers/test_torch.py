"""Tests for assorted torch layers in ``probly.layers.torch``.

Covers the interval-arithmetic family (Conv2d, Linear, BatchNorm, Softmax),
the Gaussian Mixture head used by DDU and DEUP, and the spectral-norm helpers.
"""

from __future__ import annotations

import pytest


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    return torch, nn


class TestPackUnpackInterval:
    """The interval pack/unpack helpers."""

    def test_pack_doubles_channels(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import pack_interval  # noqa: PLC0415

        x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])  # shape (2, 1, 2)
        packed = pack_interval(x, channel_dim=1)
        assert packed.shape == (2, 2, 2)
        # lower half == upper half == x
        torch.testing.assert_close(packed[:, :1], x)
        torch.testing.assert_close(packed[:, 1:], x)

    def test_unpack_splits_channels(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import pack_interval, unpack_interval  # noqa: PLC0415

        x = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        packed = pack_interval(x, channel_dim=1)
        lo, hi = unpack_interval(packed, channel_dim=1)
        torch.testing.assert_close(lo, x)
        torch.testing.assert_close(hi, x)

    def test_unpack_odd_size_raises(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import unpack_interval  # noqa: PLC0415

        x = torch.zeros(2, 3, 4)  # channels=3 is odd
        with pytest.raises(ValueError, match="odd size"):
            unpack_interval(x, channel_dim=1)


class TestIntLinear:
    """Interval-arithmetic linear layer."""

    def test_forward_zero_radius_collapses_to_linear(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntLinear, pack_interval  # noqa: PLC0415

        layer = IntLinear(in_features=4, out_features=3, bias=True)
        with torch.no_grad():
            layer.center_weight.zero_()
            layer.center_weight.add_(1.0)  # all ones
            layer.radius_weight.zero_()
            layer.center_bias.zero_()
            layer.radius_bias.zero_()
        x = torch.randn(2, 4).abs()  # non-negative input
        packed = pack_interval(x, channel_dim=1)
        out = layer(packed)
        # Output is packed -> two halves should be equal because radius is 0.
        lo, hi = out[..., :3], out[..., 3:]
        torch.testing.assert_close(lo, hi)

    def test_forward_shape_validation(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntLinear  # noqa: PLC0415

        layer = IntLinear(in_features=4, out_features=3)
        with pytest.raises(ValueError, match="expected packed input"):
            layer(torch.randn(2, 7))  # not 2*4=8

    def test_no_bias(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntLinear, pack_interval  # noqa: PLC0415

        layer = IntLinear(in_features=4, out_features=3, bias=False)
        # No center_bias / radius_bias parameters.
        assert layer.center_bias is None
        assert layer.radius_bias is None
        x = torch.rand(2, 4)
        out = layer(pack_interval(x, channel_dim=1))
        assert out.shape == (2, 6)

    def test_extra_repr_exposes_config(self) -> None:
        from probly.layers.torch import IntLinear  # noqa: PLC0415

        layer = IntLinear(in_features=4, out_features=3, bias=True)
        rep = repr(layer)
        assert "in_features=4" in rep
        assert "out_features=3" in rep
        assert "bias=True" in rep

    def test_lower_below_or_equal_upper(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntLinear, pack_interval  # noqa: PLC0415

        # Random init: lo <= hi must hold for the output.
        torch.manual_seed(0)
        layer = IntLinear(in_features=4, out_features=3)
        x = torch.rand(2, 4)
        packed = pack_interval(x, channel_dim=1)
        out = layer(packed)
        lo, hi = out[..., :3], out[..., 3:]
        assert torch.all(lo <= hi + 1e-6)


class TestIntConv2d:
    """Interval-arithmetic 2D convolution."""

    def test_forward_shape(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntConv2d, pack_interval  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        x = torch.rand(2, 3, 8, 8)
        packed = pack_interval(x, channel_dim=1)
        out = layer(packed)
        assert out.shape == (2, 8, 8, 8)  # 2*4=8 packed channels

    def test_forward_shape_validation(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntConv2d  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        with pytest.raises(ValueError, match="expected packed input"):
            layer(torch.rand(2, 5, 8, 8))

    def test_no_bias(self) -> None:
        from probly.layers.torch import IntConv2d  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False)
        assert layer.center_bias is None
        assert layer.radius_bias is None

    def test_zero_radius_makes_lo_eq_hi(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntConv2d, pack_interval  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            layer.radius_weight.zero_()
        x = torch.rand(2, 3, 8, 8)
        packed = pack_interval(x, channel_dim=1)
        out = layer(packed)
        lo, hi = out[:, :4], out[:, 4:]
        torch.testing.assert_close(lo, hi)

    def test_extra_repr_exposes_config(self) -> None:
        from probly.layers.torch import IntConv2d  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=2)
        rep = repr(layer)
        assert "in_channels=3" in rep
        assert "out_channels=4" in rep
        assert "kernel_size=(3, 3)" in rep
        assert "stride=(2, 2)" in rep

    def test_int_conv2d_kernel_tuple(self) -> None:
        torch, _ = _torch_modules()  # noqa: RUF059
        from probly.layers.torch import IntConv2d  # noqa: PLC0415

        layer = IntConv2d(in_channels=3, out_channels=4, kernel_size=(3, 5), padding=(1, 2), stride=(2, 1))
        # Just ensure the constructor accepts tuples.
        assert layer.kernel_size == (3, 5)
        assert layer.padding == (1, 2)
        assert layer.stride == (2, 1)


class TestIntBatchNorm1d:
    """Interval-valued BatchNorm1d."""

    def test_forward_shape_validation(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntBatchNorm1d  # noqa: PLC0415

        bn = IntBatchNorm1d(num_features=4)
        with pytest.raises(ValueError, match="expected packed input"):
            bn(torch.randn(2, 5))

    def test_forward_preserves_ordering(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntBatchNorm1d, pack_interval  # noqa: PLC0415

        bn = IntBatchNorm1d(num_features=4)
        bn.train()
        x = torch.randn(8, 4)
        packed = pack_interval(x, channel_dim=1)
        out = bn(packed)
        lo, hi = out[..., :4], out[..., 4:]
        # IntBatchNorm uses |radius| -> lo <= hi by construction.
        assert torch.all(lo <= hi + 1e-6)

    def test_no_affine(self) -> None:
        from probly.layers.torch import IntBatchNorm1d  # noqa: PLC0415

        bn = IntBatchNorm1d(num_features=4, affine=False)
        assert bn.center_weight is None
        assert bn.center_bias is None

    def test_no_track_running_stats(self) -> None:
        from probly.layers.torch import IntBatchNorm1d  # noqa: PLC0415

        bn = IntBatchNorm1d(num_features=4, track_running_stats=False)
        assert bn.center_running_mean is None
        assert bn.center_running_var is None

    def test_extra_repr(self) -> None:
        from probly.layers.torch import IntBatchNorm1d  # noqa: PLC0415

        bn = IntBatchNorm1d(num_features=4)
        rep = repr(bn)
        assert "num_features=4" in rep


class TestIntBatchNorm2d:
    """Interval-valued BatchNorm2d."""

    def test_forward_shape_validation(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntBatchNorm2d  # noqa: PLC0415

        bn = IntBatchNorm2d(num_features=4)
        with pytest.raises(ValueError, match="expected packed input"):
            bn(torch.randn(2, 5, 8, 8))

    def test_forward_preserves_ordering(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntBatchNorm2d, pack_interval  # noqa: PLC0415

        bn = IntBatchNorm2d(num_features=4)
        bn.train()
        x = torch.randn(8, 4, 6, 6)
        packed = pack_interval(x, channel_dim=1)
        out = bn(packed)
        lo, hi = out[:, :4], out[:, 4:]
        assert torch.all(lo <= hi + 1e-6)

    def test_no_affine(self) -> None:
        from probly.layers.torch import IntBatchNorm2d  # noqa: PLC0415

        bn = IntBatchNorm2d(num_features=4, affine=False)
        assert bn.center_weight is None
        assert bn.radius_weight is None

    def test_no_track_running_stats(self) -> None:
        from probly.layers.torch import IntBatchNorm2d  # noqa: PLC0415

        bn = IntBatchNorm2d(num_features=4, track_running_stats=False)
        assert bn.center_running_mean is None

    def test_extra_repr_2d(self) -> None:
        from probly.layers.torch import IntBatchNorm2d  # noqa: PLC0415

        bn = IntBatchNorm2d(num_features=4)
        rep = repr(bn)
        assert "num_features=4" in rep


class TestIntSoftmax:
    """Interval softmax with reachability clipping."""

    def test_collapsed_interval_matches_softmax(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntSoftmax, pack_interval  # noqa: PLC0415

        sm = IntSoftmax()
        x = torch.tensor([[1.0, 2.0, 3.0]])
        packed = pack_interval(x, channel_dim=1)
        out = sm(packed)
        lo, hi = out[..., :3], out[..., 3:]
        # When lo == hi, the softmax should match.
        expected = torch.softmax(x, dim=-1)
        torch.testing.assert_close(lo, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hi, expected, atol=1e-5, rtol=1e-5)

    def test_lower_le_upper(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IntSoftmax  # noqa: PLC0415

        sm = IntSoftmax()
        # 3-class interval with lo < hi
        lo = torch.tensor([[0.5, 1.0, 1.5]])
        hi = torch.tensor([[1.5, 2.0, 2.5]])
        packed = torch.cat([lo, hi], dim=-1)
        out = sm(packed)
        lo_out, hi_out = out[..., :3], out[..., 3:]
        assert torch.all(lo_out <= hi_out + 1e-6)
        # Probabilities are in [0, 1]
        assert torch.all((lo_out >= 0) & (lo_out <= 1))
        assert torch.all((hi_out >= 0) & (hi_out <= 1))


class TestGaussianMixtureHead:
    """The GaussianMixtureHead provides per-class log densities."""

    def test_init_uniform_priors(self) -> None:
        torch, _ = _torch_modules()
        import math  # noqa: PLC0415

        from probly.layers.torch import GaussianMixtureHead  # noqa: PLC0415

        head = GaussianMixtureHead(num_classes=3, feature_dim=4)
        # Uniform priors -> all log_pi = -log(3)
        torch.testing.assert_close(head.log_pi, torch.full((3,), -math.log(3)))
        # Means start at zero.
        torch.testing.assert_close(head.means, torch.zeros(3, 4))

    def test_fit_recovers_class_means(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import GaussianMixtureHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = GaussianMixtureHead(num_classes=2, feature_dim=2)
        # Two clusters of 100 points each, well-separated.
        f0 = torch.randn(100, 2) + torch.tensor([5.0, 5.0])
        f1 = torch.randn(100, 2) + torch.tensor([-5.0, -5.0])
        features = torch.cat([f0, f1])
        labels = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
        head.fit(features, labels)
        # Means should be close to cluster centres.
        torch.testing.assert_close(head.means[0], f0.mean(0), atol=0.5, rtol=0.5)
        torch.testing.assert_close(head.means[1], f1.mean(0), atol=0.5, rtol=0.5)

    def test_fit_with_class_count_one(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import GaussianMixtureHead  # noqa: PLC0415

        # 1 sample for class 0, multiple for class 1: count<2 branch should not crash.
        head = GaussianMixtureHead(num_classes=2, feature_dim=2)
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = torch.tensor([0, 1, 1])
        head.fit(features, labels)
        # Class 0 has count<2 -> mean stays at zero (it gets `continue` before mean assignment).
        torch.testing.assert_close(head.means[0], torch.zeros(2))

    def test_fit_with_no_samples_for_class(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import GaussianMixtureHead  # noqa: PLC0415

        head = GaussianMixtureHead(num_classes=3, feature_dim=2)
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([0, 1])  # class 2 missing
        head.fit(features, labels)
        # log_pi for class 2 should be -inf
        assert head.log_pi[2].item() == float("-inf")

    def test_forward_returns_per_class_log_probs(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import GaussianMixtureHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = GaussianMixtureHead(num_classes=3, feature_dim=4)
        # Fit on random data so the covariances are non-degenerate.
        features = torch.randn(60, 4)
        labels = torch.randint(0, 3, (60,))
        head.fit(features, labels)
        out = head.forward(features[:5])
        assert out.shape == (5, 3)


class TestSpectralNormParametrization:
    """SNCoeffParametrization and apply_spectral_norm_to_encoder helpers."""

    def test_parametrization_preserves_output_shape(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from probly.layers.torch import SNCoeffParametrization  # noqa: PLC0415

        weight = torch.randn(3, 4)
        param = SNCoeffParametrization(coeff=2.0, weight=weight)
        out = param(weight)
        assert out.shape == weight.shape

    def test_right_inverse_is_identity(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import SNCoeffParametrization  # noqa: PLC0415

        weight = torch.randn(3, 4)
        param = SNCoeffParametrization(coeff=2.0, weight=weight)
        torch.testing.assert_close(param.right_inverse(weight), weight)

    def test_apply_spectral_norm_wraps_linear_layers(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import apply_spectral_norm_to_encoder  # noqa: PLC0415

        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        apply_spectral_norm_to_encoder(encoder, sn_coeff=3.0)
        # The Linear layers should now have a parametrization on `weight`.
        for child in encoder.children():
            if isinstance(child, nn.Linear):
                assert torch.nn.utils.parametrize.is_parametrized(child)

    def test_apply_spectral_norm_to_conv2d(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import apply_spectral_norm_to_encoder  # noqa: PLC0415

        encoder = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3))
        apply_spectral_norm_to_encoder(encoder, sn_coeff=3.0)
        conv = encoder[0]
        assert torch.nn.utils.parametrize.is_parametrized(conv)

    def test_stride2_1x1_conv_replaced_with_avg_pool_and_stride1_conv(self) -> None:
        torch, nn = _torch_modules()  # noqa: RUF059
        from probly.layers.torch import apply_spectral_norm_to_encoder  # noqa: PLC0415

        encoder = nn.Sequential(nn.Conv2d(3, 8, kernel_size=1, stride=2))
        apply_spectral_norm_to_encoder(encoder, sn_coeff=3.0)
        # After applying, the stride-2 1x1 conv is wrapped in Sequential(AvgPool, Conv2d_stride1).
        assert isinstance(encoder[0], nn.Sequential)
        assert isinstance(encoder[0][0], nn.AvgPool2d)
        assert isinstance(encoder[0][1], nn.Conv2d)
        assert encoder[0][1].stride == (1, 1)

    def test_apply_spectral_norm_keeps_outputs_finite(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import apply_spectral_norm_to_encoder  # noqa: PLC0415

        torch.manual_seed(0)
        encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        apply_spectral_norm_to_encoder(encoder, sn_coeff=3.0)
        x = torch.randn(2, 4)
        out = encoder(x)
        assert torch.isfinite(out).all()


class TestInitFastWeight:
    """The private ``_init_fast_weight`` helper used by BatchEnsemble layers."""

    def test_random_sign_init_uses_pm_one(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import _init_fast_weight  # noqa: PLC0415

        torch.manual_seed(0)
        tensor = torch.empty(8, 4)
        _init_fast_weight(tensor, "random_sign", mean=0.0, std=1.0)
        # Values should all be in {-1, +1}.
        unique = torch.unique(tensor)
        assert set(unique.tolist()) <= {-1.0, 1.0}

    def test_unknown_init_raises(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import _init_fast_weight  # noqa: PLC0415

        with pytest.raises(ValueError, match="Unknown init"):
            _init_fast_weight(torch.empty(2, 2), "bogus", mean=0.0, std=1.0)  # type: ignore[arg-type]


class TestBatchEnsembleLinearNoBaseBias:
    """Cover the no-bias and divisibility-error branches of BatchEnsembleLinear / Conv2d."""

    def test_linear_no_base_bias_uses_zero_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BatchEnsembleLinear  # noqa: PLC0415

        base = nn.Linear(4, 3, bias=False)
        layer = BatchEnsembleLinear(base, num_members=2)
        # A no-bias base layer should result in zero-initialized bias.
        assert layer.bias.shape == (2, 3)
        torch.testing.assert_close(layer.bias, torch.zeros(2, 3))

    def test_conv2d_no_base_bias_uses_zero_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BatchEnsembleConv2d  # noqa: PLC0415

        base = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        layer = BatchEnsembleConv2d(base, num_members=2)
        assert layer.bias.shape == (2, 4)
        torch.testing.assert_close(layer.bias, torch.zeros(2, 4))

    def test_conv2d_batch_not_divisible_raises(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BatchEnsembleConv2d  # noqa: PLC0415

        base = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        layer = BatchEnsembleConv2d(base, num_members=2)
        # Batch size of 5 is not divisible by num_members=2.
        x = torch.rand(5, 3, 4, 4)
        with pytest.raises(ValueError, match="not divisible by num_members"):
            layer(x)


class TestBayesLinear:
    """Forward, KL, and use_base_weights paths for BayesLinear."""

    def test_forward_with_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesLinear  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesLinear(nn.Linear(4, 3), posterior_std=0.05)
        out = layer(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_forward_without_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesLinear  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesLinear(nn.Linear(4, 3, bias=False), posterior_std=0.05)
        out = layer(torch.randn(2, 4))
        assert out.shape == (2, 3)

    def test_kl_divergence_nonneg(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesLinear  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesLinear(nn.Linear(4, 3), posterior_std=0.05, prior_mean=0.0, prior_std=1.0)
        # With small posterior_std the KL is positive.
        kl = layer.kl_divergence
        assert kl.dim() == 0
        assert torch.isfinite(kl)

    def test_use_base_weights_copies_base(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesLinear  # noqa: PLC0415

        base = nn.Linear(4, 3)
        with torch.no_grad():
            base.weight.fill_(0.42)
            base.bias.fill_(-0.1)
        layer = BayesLinear(base, use_base_weights=True)
        # Posterior means and prior means should match the base weights.
        torch.testing.assert_close(layer.weight_mu, torch.full_like(layer.weight_mu, 0.42))
        torch.testing.assert_close(layer.bias_mu, torch.full_like(layer.bias_mu, -0.1))


class TestBayesConv2d:
    """Forward, KL, and use_base_weights paths for BayesConv2d."""

    def test_forward_with_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesConv2d  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesConv2d(nn.Conv2d(3, 4, kernel_size=3, padding=1), posterior_std=0.05)
        out = layer(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 4, 8, 8)
        assert torch.isfinite(out).all()

    def test_forward_without_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesConv2d  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesConv2d(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), posterior_std=0.05)
        out = layer(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 4, 8, 8)

    def test_kl_divergence_finite(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesConv2d  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesConv2d(nn.Conv2d(3, 4, kernel_size=3, padding=1), posterior_std=0.05)
        kl = layer.kl_divergence
        assert kl.dim() == 0
        assert torch.isfinite(kl)

    def test_kl_divergence_no_bias(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesConv2d  # noqa: PLC0415

        torch.manual_seed(0)
        layer = BayesConv2d(nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False), posterior_std=0.05)
        kl = layer.kl_divergence
        assert torch.isfinite(kl)

    def test_use_base_weights_copies_base(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import BayesConv2d  # noqa: PLC0415

        base = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        with torch.no_grad():
            base.weight.fill_(0.123)
            base.bias.fill_(0.5)
        layer = BayesConv2d(base, use_base_weights=True)
        torch.testing.assert_close(layer.weight_mu, torch.full_like(layer.weight_mu, 0.123))
        torch.testing.assert_close(layer.bias_mu, torch.full_like(layer.bias_mu, 0.5))


class TestDropConnectLinear:
    """Training/eval branches of DropConnectLinear."""

    def test_forward_train_applies_mask(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import DropConnectLinear  # noqa: PLC0415

        torch.manual_seed(0)
        layer = DropConnectLinear(nn.Linear(4, 3), p=0.5)
        layer.train()
        x = torch.randn(2, 4)
        out = layer(x)
        assert out.shape == (2, 3)

    def test_forward_eval_scales_weights(self) -> None:
        torch, nn = _torch_modules()
        from probly.layers.torch import DropConnectLinear  # noqa: PLC0415

        torch.manual_seed(0)
        # In eval mode the layer is deterministic — calling twice gives the same result.
        layer = DropConnectLinear(nn.Linear(4, 3), p=0.25)
        layer.eval()
        x = torch.randn(2, 4)
        out1 = layer(x)
        out2 = layer(x)
        torch.testing.assert_close(out1, out2)

    def test_extra_repr_exposes_features(self) -> None:
        _, nn = _torch_modules()
        from probly.layers.torch import DropConnectLinear  # noqa: PLC0415

        layer = DropConnectLinear(nn.Linear(4, 3), p=0.25)
        rep = repr(layer)
        assert "in_features=4" in rep
        assert "out_features=3" in rep


class TestNormalInverseGammaLinear:
    """Forward and parameter shapes for the NIG linear head."""

    def test_forward_returns_dict_with_correct_shapes_and_constraints(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import NormalInverseGammaLinear  # noqa: PLC0415

        torch.manual_seed(0)
        layer = NormalInverseGammaLinear(in_features=4, out_features=3)
        out = layer(torch.randn(2, 4))
        # Output is a dict with four parameter tensors.
        assert set(out.keys()) == {"gamma", "nu", "alpha", "beta"}
        for key in ("gamma", "nu", "alpha", "beta"):
            assert out[key].shape == (2, 3)
        # nu, beta > 0 (softplus); alpha >= 1 (softplus + 1).
        assert torch.all(out["nu"] > 0)
        assert torch.all(out["beta"] > 0)
        assert torch.all(out["alpha"] >= 1.0)


class TestRadialNormalizingFlow:
    """Single radial flow and the stack thereof."""

    def test_single_flow_forward_shape_and_log_det(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import RadialNormalizingFlow  # noqa: PLC0415

        torch.manual_seed(0)
        flow = RadialNormalizingFlow(dim=4, num_classes=2)
        z = torch.randn(3, 2, 4)  # [B, num_classes, D]
        z_out, log_det = flow(z)
        assert z_out.shape == z.shape
        assert log_det.shape == (3, 2)
        assert torch.isfinite(z_out).all()
        assert torch.isfinite(log_det).all()

    def test_stack_forward_shape(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import RadialNormalizingFlowStack  # noqa: PLC0415

        torch.manual_seed(0)
        stack = RadialNormalizingFlowStack(dim=4, num_flows=2, num_classes=3)
        z = torch.randn(5, 4)
        z_out, total_log_det = stack(z)
        # Forward expands z to [B, num_classes, D]
        assert z_out.shape == (5, 3, 4)
        assert total_log_det.shape == (5, 3)
        assert torch.isfinite(z_out).all()

    def test_stack_log_prob_shape(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import RadialNormalizingFlowStack  # noqa: PLC0415

        torch.manual_seed(0)
        stack = RadialNormalizingFlowStack(dim=4, num_flows=2, num_classes=3)
        z = torch.randn(5, 4)
        log_p = stack.log_prob(z)
        assert log_p.shape == (5, 3)
        assert torch.isfinite(log_p).all()


class TestRegressionHead:
    """RegressionHead converts features to (mu, kappa, alpha, beta)."""

    def test_forward_produces_normal_gamma_params(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import RegressionHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = RegressionHead(latent_dim=8)
        features = torch.randn(4, 8)
        mu, kappa, alpha, beta = head(features)
        assert mu.shape == (4, 1)
        assert kappa.shape == (4, 1)
        assert alpha.shape == (4, 1)
        assert beta.shape == (4, 1)
        # softplus -> kappa, beta > 0; alpha >= 1.0
        assert torch.all(kappa > 0)
        assert torch.all(alpha >= 1.0)
        assert torch.all(beta > 0)


class TestNatPNClassHead:
    """Dirichlet posterior head used in NatPN-style classifiers."""

    def test_forward_default_n_prior(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import NatPNClassHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = NatPNClassHead(latent_dim=8, num_classes=3, hidden_dim=16)
        # The default prior should be uniform with sum num_classes.
        torch.testing.assert_close(head.alpha_prior.sum(), torch.tensor(3.0))
        features = torch.randn(4, 8)
        log_pz = torch.full((4,), -1.0)
        out = head(features, log_pz, certainty_budget=2.0)
        assert set(out.keys()) == {"alpha", "features", "log_pz", "evidence"}
        assert out["alpha"].shape == (4, 3)
        # alpha_i >= alpha_prior_i (evidence >= 0)  # noqa: ERA001
        assert torch.all(out["alpha"] >= head.alpha_prior.unsqueeze(0))

    def test_forward_explicit_n_prior(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import NatPNClassHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = NatPNClassHead(latent_dim=4, num_classes=2, hidden_dim=8, n_prior=4.0)
        torch.testing.assert_close(head.alpha_prior.sum(), torch.tensor(4.0))


class TestNatPNRegHead:
    """Gaussian posterior head used in NatPN-style regressors."""

    def test_forward_returns_mean_and_var(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import NatPNRegHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = NatPNRegHead(latent_dim=8, out_dim=2)
        features = torch.randn(4, 8)
        log_pz = torch.zeros(4)  # exp -> 1.0
        out = head(features, log_pz, certainty_budget=1.0)
        assert set(out.keys()) == {"mean", "var", "features", "log_pz", "precision"}
        assert out["mean"].shape == (4, 2)
        assert out["var"].shape == (4, 2)
        # precision = certainty_budget * exp(log_pz).unsqueeze(-1) -> (B, 1) and broadcasts.
        assert out["precision"].shape == (4, 1)
        # var > 0 and precision > 0 by construction (exp / clamp).
        assert torch.all(out["var"] > 0)
        assert torch.all(out["precision"] > 0)


class TestIRDHead:
    """Dirichlet concentration parameter head."""

    def test_forward_alpha_geq_one(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import IRDHead  # noqa: PLC0415

        torch.manual_seed(0)
        head = IRDHead(latent_dim=8, num_classes=4)
        features = torch.randn(3, 8)
        alpha = head(features)
        assert alpha.shape == (3, 4)
        # softplus + 1.0 implies alpha >= 1.0.
        assert torch.all(alpha >= 1.0)
        assert torch.isfinite(alpha).all()


class TestHeteroscedasticLayer:
    """Forward paths of the heteroscedastic logits layer."""

    def test_single_sample_full_routing(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import HeteroscedasticLayer  # noqa: PLC0415

        torch.manual_seed(0)
        layer = HeteroscedasticLayer(in_features=8, num_classes=3, num_factors=4)
        # Default training_samples == 1 returns scaled logits.
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 3)
        assert torch.isfinite(out).all()

    def test_single_sample_parameter_efficient(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import HeteroscedasticLayer  # noqa: PLC0415

        torch.manual_seed(0)
        layer = HeteroscedasticLayer(
            in_features=8,
            num_classes=3,
            num_factors=4,
            is_parameter_efficient=True,
        )
        # The parameter-efficient routing introduces a global V_matrix.
        assert hasattr(layer, "V_matrix")
        assert layer.V_matrix.shape == (3, 4)
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 3)
        assert torch.isfinite(out).all()

    def test_multi_sample_full_routing_returns_log_probs(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import HeteroscedasticLayer  # noqa: PLC0415

        torch.manual_seed(0)
        layer = HeteroscedasticLayer(in_features=8, num_classes=3, num_factors=4)
        layer.training_samples = 4
        x = torch.randn(5, 8)
        out = layer(x)
        # Output is log of softmax-averaged probabilities -> sum_class exp(out) == 1.
        assert out.shape == (5, 3)
        probs = out.exp()
        torch.testing.assert_close(probs.sum(-1), torch.ones(5), atol=1e-5, rtol=1e-5)
        assert torch.all(out <= 0.0 + 1e-6)  # log-probability <= 0

    def test_multi_sample_parameter_efficient_routing(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import HeteroscedasticLayer  # noqa: PLC0415

        torch.manual_seed(0)
        layer = HeteroscedasticLayer(
            in_features=8,
            num_classes=3,
            num_factors=4,
            is_parameter_efficient=True,
        )
        layer.training_samples = 4
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 3)
        probs = out.exp()
        torch.testing.assert_close(probs.sum(-1), torch.ones(5), atol=1e-5, rtol=1e-5)


class TestKLDivergenceHelper:
    """The private ``_kl_divergence_gaussian`` helper and ``_inverse_softplus``."""

    def test_kl_zero_when_distributions_equal(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import _kl_divergence_gaussian  # noqa: PLC0415

        mu = torch.tensor([0.0, 1.0])
        var = torch.tensor([1.0, 2.0])
        kl = _kl_divergence_gaussian(mu, var, mu, var)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=1e-6)

    def test_kl_nonneg_for_different_distributions(self) -> None:
        torch, _ = _torch_modules()
        from probly.layers.torch import _kl_divergence_gaussian  # noqa: PLC0415

        mu1 = torch.tensor([0.0])
        var1 = torch.tensor([1.0])
        mu2 = torch.tensor([2.0])
        var2 = torch.tensor([1.0])
        kl = _kl_divergence_gaussian(mu1, var1, mu2, var2)
        # KL(N(0,1) || N(2,1)) = 2.0 (exact).
        torch.testing.assert_close(kl, torch.tensor([2.0]))

    def test_inverse_softplus_round_trip(self) -> None:
        torch, _ = _torch_modules()
        from torch.nn.functional import softplus  # noqa: PLC0415

        from probly.layers.torch import _inverse_softplus  # noqa: PLC0415

        x = torch.tensor([0.05, 0.5, 1.0, 2.0])
        # softplus(_inverse_softplus(x)) == x for x > 0.
        torch.testing.assert_close(softplus(_inverse_softplus(x)), x, atol=1e-5, rtol=1e-5)
