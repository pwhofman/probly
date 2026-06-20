"""Tests for shared dropout masks in ``probly.representer.sampler.torch``.

The ``shared_dropout_mask`` feature replaces the per-batch-element mask of the
torch dropout layers with a single mask shared across the batch.  These tests
check that the shared mask honors the differing semantics of each variant:
element-wise vs. channel-wise granularity and zeroing vs. alpha-dropout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.representer.sampler._common import get_sampling_predictor  # noqa: E402
from probly.representer.sampler.torch import _ALPHA_PRIME  # noqa: E402


def _shared_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``model`` once with a shared dropout mask and return the output."""
    sampling_model, cleanup = get_sampling_predictor(model, shared_dropout_mask=True)
    try:
        return sampling_model(x)
    finally:
        cleanup()


def _alpha_constants(p: float) -> tuple[float, float]:
    """Return the affine ``(a, b)`` used by alpha dropout for probability ``p``."""
    keep = 1.0 - p
    a = (keep + _ALPHA_PRIME**2 * keep * p) ** -0.5
    b = -a * p * _ALPHA_PRIME
    return a, b


class TestSharedDropoutMask:
    """The shared mask reproduces each dropout variant's semantics."""

    def test_dropout_is_element_wise_and_shared(self) -> None:
        """``nn.Dropout`` zeroes individual elements with one mask for the batch."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Dropout(p=0.5))
        x = torch.ones(4, 8)

        out = _shared_output(model, x)

        assert torch.allclose(out, out[:1].expand_as(out))
        assert set(out.flatten().tolist()) <= {0.0, 2.0}

    @pytest.mark.parametrize(
        ("layer", "shape"),
        [
            (nn.Dropout1d, (4, 16, 5)),  # (N, C, L)
            (nn.Dropout2d, (4, 16, 3, 3)),  # (N, C, H, W)
            (nn.Dropout3d, (4, 16, 2, 2, 2)),  # (N, C, D, H, W)
        ],
    )
    def test_dropout_nd_drops_whole_channels(self, layer: type[nn.Module], shape: tuple[int, ...]) -> None:
        """``nn.Dropout{1,2,3}d`` drop entire channels with one mask for the batch."""
        torch.manual_seed(0)
        model = nn.Sequential(layer(p=0.5))
        x = torch.ones(*shape)

        out = _shared_output(model, x)

        assert torch.allclose(out, out[:1].expand_as(out))
        for channel in out[0]:
            assert set(channel.flatten().tolist()) in ({0.0}, {2.0})

    def test_alpha_dropout_uses_saturation_value(self) -> None:
        """``nn.AlphaDropout`` sets dropped units to the affine SELU value, not 0."""
        torch.manual_seed(0)
        p = 0.3
        a, b = _alpha_constants(p)
        expected_keep = a * 1.0 + b
        expected_drop = a * _ALPHA_PRIME + b

        model = nn.Sequential(nn.AlphaDropout(p=p))
        out = _shared_output(model, torch.ones(3, 64))

        assert torch.allclose(out, out[:1].expand_as(out))
        diff_keep = (out - expected_keep).abs()
        diff_drop = (out - expected_drop).abs()
        assert torch.minimum(diff_keep, diff_drop).max().item() < 1e-5
        assert (diff_keep < 1e-5).any()
        assert (diff_drop < 1e-5).any()

    def test_alpha_dropout_preserves_moments(self) -> None:
        """Alpha dropout keeps mean ~0 and variance ~1 (unlike plain zeroing)."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.AlphaDropout(p=0.2))
        x = torch.randn(2, 4096)

        out = _shared_output(model, x)

        assert abs(out.mean().item()) < 0.1
        assert abs(out.var().item() - 1.0) < 0.3

    def test_feature_alpha_dropout_is_channel_wise_alpha(self) -> None:
        """``nn.FeatureAlphaDropout`` applies the alpha value to whole channels."""
        torch.manual_seed(0)
        p = 0.4
        a, b = _alpha_constants(p)
        expected_keep = a * 1.0 + b
        expected_drop = a * _ALPHA_PRIME + b

        model = nn.Sequential(nn.FeatureAlphaDropout(p=p))
        out = _shared_output(model, torch.ones(2, 32, 3, 3))

        assert torch.allclose(out, out[:1].expand_as(out))
        for channel in out[0]:
            keep_close = torch.allclose(channel, torch.full_like(channel, expected_keep))
            drop_close = torch.allclose(channel, torch.full_like(channel, expected_drop))
            assert keep_close or drop_close

    def test_zero_probability(self) -> None:
        """With ``p == 0`` no hook is installed and the input passes through."""
        model = nn.Sequential(nn.Dropout(p=0.0))
        x = torch.randn(2, 8)

        assert torch.allclose(_shared_output(model, x), x)
