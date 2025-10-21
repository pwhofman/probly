"""Sampling preparation for torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn

from probly.layers.torch import DropConnectLinear

from . import sampler

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from pytraverse import State


def _enforce_train_mode(obj: torch.nn.Module, state: State) -> tuple[torch.nn.Module, State]:
    if not obj.training:
        obj.train()
        state[sampler.CLEANUP_FUNCS].add(lambda: obj.train(False))
        return obj, state
    return obj, state


def register_forced_train_mode(cls: LazyType) -> None:
    """Register a class to be forced into train mode during sampling."""
    sampler.sampling_preparation_traverser.register(
        cls,
        _enforce_train_mode,
    )


register_forced_train_mode(
    torch.nn.Dropout
    | torch.nn.Dropout1d
    | torch.nn.Dropout2d
    | torch.nn.Dropout3d
    | torch.nn.AlphaDropout
    | torch.nn.FeatureAlphaDropout
    | DropConnectLinear,
)
