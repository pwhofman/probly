"""Sampling preparation for torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn

from probly.layers.torch import DropConnectLinear

from ._common import CLEANUP_FUNCS, sampling_preparation_traverser

if TYPE_CHECKING:
    from flextype.isinstance import LazyType
    from pytraverse import State


def _enforce_train_mode(obj: torch.nn.Module, state: State) -> tuple[torch.nn.Module, State]:
    if not obj.training:
        obj.train()
        state[CLEANUP_FUNCS].add(lambda: obj.train(False))
        return obj, state
    return obj, state


def register_forced_train_mode(cls: LazyType) -> None:
    """Register a class to be forced into train mode during sampling.

    This enables Monte Carlo sampling techniques like MC Dropout :cite:`galDropoutBayesian2016`
    or DropConnect :cite:`mobinyDropConnectEffective2021`.
    """
    sampling_preparation_traverser.register(cls, _enforce_train_mode)


def _install_shared_dropout_hook(obj: torch.nn.Dropout, state: State) -> tuple[torch.nn.Dropout, State]:
    """Install a forward hook that draws one shared mask per forward pass.

    ``nn.Dropout`` normally draws an independent mask for every sample in the
    batch.  When the representer evaluates a large grid of inputs in a single
    batch (e.g. a 200x200 uncertainty map), this means every grid point gets a
    *different* sub-network, so the averaged uncertainty map looks dotty.

    This hook generates a single mask for the feature dimensions and broadcasts
    it over the batch, matching the implicit behaviour of
    :class:`~probly.layers.torch.DropConnectLinear` and producing smooth maps.
    The hook is registered via ``CLEANUP_FUNCS`` and removed automatically after
    sampling.
    """
    p = obj.p
    if p == 0.0:
        return obj, state

    def _hook(
        module: torch.nn.Module,  # noqa: ARG001
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,  # noqa: ARG001
    ) -> torch.Tensor:
        x = inp[0]
        mask = torch.bernoulli(torch.full(x.shape[1:], 1.0 - p, device=x.device, dtype=x.dtype)).unsqueeze(0) / (
            1.0 - p
        )
        return x * mask

    handle = obj.register_forward_hook(_hook)
    state[CLEANUP_FUNCS].add(handle.remove)
    return obj, state


for _dropout_cls in (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
):
    sampling_preparation_traverser.register(_dropout_cls, _install_shared_dropout_hook)

register_forced_train_mode(DropConnectLinear)
