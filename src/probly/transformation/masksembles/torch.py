"""Torch masksembles implementation."""

from __future__ import annotations

from typing import Any

from torch import nn

from probly.layers.torch import Masksembles2DLayer, MasksemblesLinearLayer

from ._common import generation_wrapper, register


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
