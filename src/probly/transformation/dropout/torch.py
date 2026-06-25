"""Torch dropout implementation."""

from __future__ import annotations

from typing import Any

from torch import nn

from probly.layers.torch import SharedMaskDropout

from ._common import register


def prepend_torch_dropout(
    obj: nn.Module,
    p: float,
    rng_collection: Any = None,  # noqa: ANN401, ARG001
    rngs: Any = None,  # noqa: ANN401, ARG001
    shared_mask: bool = False,
) -> nn.Sequential:
    """Prepend a Dropout layer before the given layer based on :cite:`galDropoutBayesian2016`.

    This construction allows for Monte Carlo Dropout inference by keeping the dropout layer active during prediction.
    When ``shared_mask`` is True, a :class:`~probly.layers.torch.SharedMaskDropout` is used so that a single mask is
    drawn per forward pass and shared across the batch.
    """
    layer = SharedMaskDropout(p=p) if shared_mask else nn.Dropout(p=p)
    return nn.Sequential(layer, obj)


register(nn.Linear, prepend_torch_dropout)
