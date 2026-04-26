"""Torch credal net implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import torch
from torch import nn

from probly.layers.torch import (
    IntBatchNorm1d,
    IntBatchNorm2d,
    IntConv2d,
    IntLinear,
    IntSoftmax,
    pack_interval,
)
from probly.predictor import predict_raw

from ._common import REPLACED, CredalNetPredictor, credal_net_traverser, intersection_probability

if TYPE_CHECKING:
    from pytraverse import State


@intersection_probability.register(torch.Tensor)
def _intersection_probability(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    width = upper - lower
    alpha = (1.0 - lower.sum(dim=-1, keepdim=True)) / width.sum(dim=-1, keepdim=True)
    return lower + alpha * width


@predict_raw.register(CredalNetPredictor)
def _(predictor: CredalNetPredictor, x: torch.Tensor, /, *args: object, **kwargs: object) -> torch.Tensor:
    """Pack the input on the channel dim before invoking the credal-net model."""
    return predictor(pack_interval(x, channel_dim=1), *args, **kwargs)  # ty:ignore[call-non-callable]


@credal_net_traverser.register(nn.Module)
def _(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default: pass through after the head is placed; blank before.

    With ``TRAVERSE_REVERSED``, modules visited before ``REPLACED`` becomes
    True are tail modules sitting after the last ``Linear`` (e.g. a final
    softmax in a ``ProbabilisticClassifier`` base). The credal head replaces
    them, so we drop them here.
    """
    if state[REPLACED]:
        return obj, state
    return nn.Sequential(), state


def _warn_conv2d_unsupported(obj: nn.Conv2d) -> None:
    """Warn about ``nn.Conv2d`` options that ``IntConv2d`` does not preserve."""
    if obj.dilation != (1, 1):
        warnings.warn(
            f"IntConv2d does not support dilation={obj.dilation}; ignoring (using default).",
            stacklevel=3,
        )
    if obj.groups != 1:
        warnings.warn(
            f"IntConv2d does not support groups={obj.groups}; ignoring (using default).",
            stacklevel=3,
        )
    if obj.padding_mode != "zeros":
        warnings.warn(
            f"IntConv2d does not support padding_mode={obj.padding_mode!r}; using 'zeros'.",
            stacklevel=3,
        )


def _bn_momentum(obj: nn.BatchNorm1d | nn.BatchNorm2d) -> float:
    """Return the BatchNorm's momentum, warning and falling back if it is ``None``."""
    if obj.momentum is None:
        warnings.warn(
            "IntBatchNorm does not support momentum=None (cumulative moving average); falling back to momentum=0.1.",
            stacklevel=3,
        )
        return 0.1
    return obj.momentum


@credal_net_traverser.register(nn.Conv2d)
def _(obj: nn.Conv2d, state: State) -> tuple[nn.Module, State]:
    """Replace ``Conv2d`` with ``IntConv2d`` after the head is placed; blank before."""
    if not state[REPLACED]:
        return nn.Sequential(), state
    _warn_conv2d_unsupported(obj)
    new = IntConv2d(
        in_channels=obj.in_channels,
        out_channels=obj.out_channels,
        kernel_size=obj.kernel_size,  # ty:ignore[invalid-argument-type]
        stride=obj.stride,  # ty:ignore[invalid-argument-type]
        padding=obj.padding,  # ty:ignore[invalid-argument-type]
        bias=obj.bias is not None,
    )
    return new, state


@credal_net_traverser.register(nn.BatchNorm2d)
def _(obj: nn.BatchNorm2d, state: State) -> tuple[nn.Module, State]:
    """Replace ``BatchNorm2d`` with ``IntBatchNorm2d`` after the head is placed; blank before."""
    if not state[REPLACED]:
        return nn.Sequential(), state
    new = IntBatchNorm2d(
        num_features=obj.num_features,
        eps=obj.eps,
        momentum=_bn_momentum(obj),
        affine=obj.affine,
        track_running_stats=obj.track_running_stats,
    )
    return new, state


@credal_net_traverser.register(nn.BatchNorm1d)
def _(obj: nn.BatchNorm1d, state: State) -> tuple[nn.Module, State]:
    """Replace ``BatchNorm1d`` with ``IntBatchNorm1d`` after the head is placed; blank before."""
    if not state[REPLACED]:
        return nn.Sequential(), state
    new = IntBatchNorm1d(
        num_features=obj.num_features,
        eps=obj.eps,
        momentum=_bn_momentum(obj),
        affine=obj.affine,
        track_running_stats=obj.track_running_stats,
    )
    return new, state


@credal_net_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace ``Linear``: insert the credal head for the last one, ``IntLinear`` for earlier ones."""
    if state[REPLACED]:
        return IntLinear(obj.in_features, obj.out_features, bias=obj.bias is not None), state
    state[REPLACED] = True
    new_head = nn.Sequential(
        IntLinear(obj.in_features, obj.out_features, bias=obj.bias is not None),
        IntBatchNorm1d(obj.out_features),
        IntSoftmax(),
    )
    return new_head, state
