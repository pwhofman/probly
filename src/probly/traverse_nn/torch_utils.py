"""PyTorch-specific utils for traverse_nn."""

from __future__ import annotations

from torch import nn

from .utils import LAST_OUTPUT_DIM, output_dim_traverser


@output_dim_traverser.register(
    nn.Linear,
    vars={"_current": LAST_OUTPUT_DIM},
    update_vars=True,
)
def _(obj: nn.Linear, _current: int) -> tuple[nn.Linear, dict[str, int]]:
    """Record the output feature count of a torch ``nn.Linear`` layer."""
    return obj, {"_current": obj.out_features}


@output_dim_traverser.register(
    (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
    vars={"_current": LAST_OUTPUT_DIM},
    update_vars=True,
)
def _(obj: nn.Module, _current: int) -> tuple[nn.Module, dict[str, int]]:
    """Record the output channel count of a torch convolutional layer."""
    return obj, {"_current": obj.out_channels}  # ty: ignore
