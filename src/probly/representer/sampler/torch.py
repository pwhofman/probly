"""Sampling preparation for torch."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn

from probly.layers.torch import DropConnectLinear

from ._common import CLEANUP_FUNCS, SHARED_DROPOUT_MASK, sampling_preparation_traverser

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype.isinstance import LazyType
    from pytraverse import State

# SELU saturation constants, matching ``torch.nn.functional.alpha_dropout``.
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946
_ALPHA_PRIME = -_SELU_SCALE * _SELU_ALPHA


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


def _make_shared_dropout_hook(
    p: float,
    *,
    channel_wise: bool,
    alpha: bool,
) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]:
    """Build a forward hook that applies one shared mask per forward pass.

    Args:
        p: The dropout probability of the wrapped layer.
        channel_wise: If True, draw one value per channel (dim 1) and broadcast it
            over the spatial dimensions, matching ``Dropout2d``-style layers that
            drop whole feature maps.  If False, draw one value per feature element.
        alpha: If True, reproduce alpha-dropout semantics: dropped units are set to
            the SELU saturation value and an affine transform restores the mean and
            variance, matching ``AlphaDropout``.  If False, dropped units are zeroed
            and survivors are scaled by ``1 / (1 - p)``.

    Returns:
        A forward hook recomputing the layer output from its clean input so that the
        same mask is shared across the whole batch.
    """
    keep = 1.0 - p
    # Alpha-dropout affine, depends only on ``p`` so compute it once here.  Guard
    # the negative power behind ``alpha``: only the alpha path uses ``a``/``b``,
    # and only there is ``keep`` assumed non-zero (the plain path tolerates p=1).
    a = (keep + _ALPHA_PRIME**2 * keep * p) ** -0.5 if alpha else 0.0
    b = -a * p * _ALPHA_PRIME

    def _hook(
        module: torch.nn.Module,  # noqa: ARG001
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,  # noqa: ARG001
    ) -> torch.Tensor:
        x = inp[0]
        shape = (x.shape[1], *([1] * (x.ndim - 2))) if channel_wise else tuple(x.shape[1:])
        bern = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype)).unsqueeze(0)
        if not alpha:
            return x * (bern / keep)
        return a * (x * bern + _ALPHA_PRIME * (1.0 - bern)) + b

    return _hook


def _install_shared_dropout_hook(
    obj: torch.nn.Dropout,
    state: State,
    *,
    channel_wise: bool,
    alpha: bool,
) -> tuple[torch.nn.Dropout, State]:
    """Prepare a torch dropout layer for sampling.

    The torch dropout layers normally draw an independent mask for every sample in
    the batch.  When the representer evaluates a large grid of inputs in a single
    batch (e.g. a 200x200 uncertainty map), every grid point then gets a
    *different* sub-network, so the averaged uncertainty map looks dotty.

    When ``SHARED_DROPOUT_MASK`` is set, this installs a forward hook that draws a
    single mask and broadcasts it over the batch -- matching the implicit behavior
    of :class:`~probly.layers.torch.DropConnectLinear` and producing smooth maps.
    ``channel_wise`` and ``alpha`` select the masking rule so that channel-dropping
    (``Dropout1d/2d/3d``, ``FeatureAlphaDropout``) and alpha-dropout
    (``AlphaDropout``, ``FeatureAlphaDropout``) semantics are honored, not just plain
    element-wise zeroing.  The hook is registered via ``CLEANUP_FUNCS`` and removed
    automatically after sampling.

    Otherwise (shared mask off) the layer is simply forced into train mode, the
    classic MC-dropout path.  A layer with ``p == 0`` is left untouched either way.
    """
    if not state[SHARED_DROPOUT_MASK]:
        _, state = _enforce_train_mode(obj, state)
        return obj, state

    p = obj.p
    if p == 0.0:
        return obj, state

    handle = obj.register_forward_hook(_make_shared_dropout_hook(p, channel_wise=channel_wise, alpha=alpha))
    state[CLEANUP_FUNCS].add(handle.remove)
    return obj, state


# Map each torch dropout layer to its (channel_wise, alpha) masking rule.
for _dropout_cls, (_channel_wise, _alpha) in {
    torch.nn.Dropout: (False, False),
    torch.nn.Dropout1d: (True, False),
    torch.nn.Dropout2d: (True, False),
    torch.nn.Dropout3d: (True, False),
    torch.nn.AlphaDropout: (False, True),
    torch.nn.FeatureAlphaDropout: (True, True),
}.items():
    sampling_preparation_traverser.register(
        _dropout_cls,
        partial(_install_shared_dropout_hook, channel_wise=_channel_wise, alpha=_alpha),
    )

register_forced_train_mode(DropConnectLinear)
