"""Torch dropout implementation."""

from __future__ import annotations

from typing import Any

from torch import nn

from probly.layers.torch import DropConnectLinear

from .common import register


def replace_torch_dropconnect(
    obj: nn.Linear,
    p: float,
    rng_collection: Any = None,  # noqa: ANN401, ARG001
    rngs: Any = None,  # noqa: ANN401, ARG001
) -> DropConnectLinear:
    """Replace a given layer by a DropConnectLinear layer based on :cite:`mobinyDropConnectEffective2019`."""
    return DropConnectLinear(obj, p=p)


register(nn.Linear, replace_torch_dropconnect)
