"""Torch masksembles implementation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from probly.layers.torch import Masksembles2DLayer, MasksemblesLinearLayer
from probly.predictor import predict
from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample.torch import TorchSample

from ._common import MasksemblesPredictor, _attach_n_masks, _wrap_masksembles_logits, register


def tile_inputs(x: torch.Tensor, n_masks: int) -> torch.Tensor:
    """Tile the leading batch dim by ``n_masks``, preserving ``channels_last`` for 4D inputs."""
    out = torch.tile(x, (n_masks,) + (1,) * (x.dim() - 1))
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        out = out.contiguous(memory_format=torch.channels_last)
    return out


@_attach_n_masks.register(nn.Module)
def _(model: nn.Module, n_masks: int) -> None:
    """Register ``n_masks`` as a persistent buffer so it survives ``state_dict``."""
    model.register_buffer("n_masks", torch.tensor(n_masks, dtype=torch.long))


def generate_masks_(m: int, n: int, scale: float) -> torch.Tensor:
    """Sample one candidate mask set; shape may vary — retry in :func:`generate_masks` until fixed."""
    total_positions = int(m * scale)
    masks = torch.zeros(n, total_positions)
    for _ in range(n):
        idx = torch.randperm(total_positions)[:m]
        masks[_, idx] = 1
    masks = masks[:, ~(masks == 0).all(dim=0)]
    return masks


def generate_masks(m: int, n: int, scale: float) -> torch.Tensor:
    """Generate masks with exactly ``expected_size`` columns per :cite:`durasovMasksembles2021`."""
    masks = generate_masks_(m, n, scale)
    expected_size = int(m * scale * (1 - (1 - 1 / scale) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, scale)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> torch.Tensor:
    """Bisect over scale to produce masks with exactly ``c`` active columns.

    Args:
        c: Target number of active features/channels per mask. Must be >= 10.
        n: Number of masks.
        scale: Initial scale hint; the bisection adjusts it to hit exactly ``c`` columns.

    Returns:
        Binary mask tensor of shape ``[n, c]``.

    Raises:
        ValueError: If ``c < 10`` (too few features for meaningful mask generation) or
            if bisection fails to converge within 1000 iterations.
    """
    if c < 10:
        msg = (
            "Masksembles cannot be used where the number of channels/features is less "
            f"than 10. Current value is (channels={c}). Increase the number of features "
            "in this layer or remove Masksembles from this part of the architecture."
        )
        raise ValueError(msg)

    m = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))
    max_iter = 1000

    lo = max([scale * 0.8, 1.0])
    hi = scale * 1.2

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        masks = generate_masks(m, n, mid)
        if masks.shape[-1] == c:
            break
        if masks.shape[-1] > c:
            hi = mid
        else:
            lo = mid

    if masks.shape[-1] != c:
        msg = "generation_wrapper was unable to generate fitting masks"
        raise ValueError(msg)
    return masks


@predict.register(MasksemblesPredictor)
def predict_masksembles(
    predictor: MasksemblesPredictor,
    x: torch.Tensor,
) -> TorchSample[torch.Tensor]:
    """Run a Masksembles predictor and return a :class:`TorchSample` over masks.

    Tiles the user's ``[B, ...]`` input by ``n_masks``, runs the model once on the
    tiled ``[N*B, ...]`` tensor in eval mode (each contiguous block of ``B`` rows gets
    one of the ``N`` masks applied, per :class:`Masksembles2DLayer`/:class:`MasksemblesLinearLayer`),
    and reshapes the output to ``[N, B, ...]`` with ``sample_dim=0``.
    """
    n_masks = int(predictor.n_masks)
    b = x.shape[0]
    predictor.eval()
    raw = predictor(tile_inputs(x, n_masks))
    out = raw.view(n_masks, b, *raw.shape[1:])
    return TorchSample(tensor=out, sample_dim=0)


@_wrap_masksembles_logits.register(TorchSample)
def _torch_wrap_batchensemble_logits(sample: TorchSample[Any]) -> TorchCategoricalDistributionSample:
    """Wrap per-member logits as a categorical sample (assumes a logit-classifier base)."""
    tensor = sample.tensor
    sample_dim = sample.sample_dim
    if tensor.ndim >= 3 and sample_dim == 0:
        tensor = tensor.transpose(0, 1)
        sample_dim = 1

    distribution = create_categorical_distribution_from_logits(tensor)
    return TorchCategoricalDistributionSample(tensor=distribution, sample_dim=sample_dim)  # ty: ignore[invalid-argument-type]


def append_torch_masksembles_conv(
    obj: nn.Module,
    n_masks: int,
    scale: float,
    rng_collection: Any = None,  # noqa: ANN401, ARG001
    rngs: Any = None,  # noqa: ANN401, ARG001
) -> nn.Module:
    """Append a Masksembles2DLayer after a Conv2d layer."""
    if isinstance(obj, nn.Conv2d):
        channels = obj.out_channels
        mask_layer = Masksembles2DLayer(
            masks=generation_wrapper(channels, n_masks, scale),
            channels=channels,
            n=n_masks,
            scale=scale,
        )
        return nn.Sequential(obj, mask_layer)

    return obj


def append_torch_masksembles_linear(
    obj: nn.Module,
    n_masks: int,
    scale: float,
    rng_collection: Any = None,  # noqa: ANN401, ARG001
    rngs: Any = None,  # noqa: ANN401, ARG001
) -> nn.Module:
    """Append a MasksemblesLinearLayer after a Linear layer."""
    if isinstance(obj, nn.Linear):
        features = obj.out_features
        mask_layer = MasksemblesLinearLayer(
            masks=generation_wrapper(features, n_masks, scale),
            features=features,
            n=n_masks,
            scale=scale,
        )
        return nn.Sequential(obj, mask_layer)

    return obj


register(nn.Conv2d, append_torch_masksembles_conv)
register(nn.Linear, append_torch_masksembles_linear)
