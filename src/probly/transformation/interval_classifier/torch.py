"""Torch interval classifier implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
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

from ._common import REPLACED, USE_BASE_WEIGHTS, IntervalClassifierPredictor, interval_classifier_traverser

if TYPE_CHECKING:
    from pytraverse import State


@predict_raw.register(IntervalClassifierPredictor)
def _(predictor: IntervalClassifierPredictor, x: torch.Tensor, /, *args: object, **kwargs: object) -> torch.Tensor:
    """Pack the input on the channel dim before invoking the interval classifier."""
    return cast("Any", predictor)(pack_interval(x, channel_dim=1), *args, **kwargs)


@interval_classifier_traverser.register(nn.Module)
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


@torch.no_grad()
def _copy_weight_into_center(new: IntConv2d | IntLinear, obj: nn.Conv2d | nn.Linear) -> None:
    """Copy a base layer's weight (and optional bias) into the interval layer's center slots.

    Works for both ``nn.Conv2d`` and ``nn.Linear`` because they share the same
    ``weight``/``bias`` attribute names and shapes match the corresponding
    ``IntConv2d``/``IntLinear`` ``center_weight``/``center_bias`` slots.

    Also zeros the radius weight and bias so the interval starts degenerate
    (lo == hi), reproducing the pretrained model's predictions exactly before
    any training has modified the radius parameters.
    """
    new.center_weight.copy_(obj.weight)
    new.radius_weight.zero_()
    if obj.bias is not None and new.center_bias is not None:
        new.center_bias.copy_(obj.bias)
    if new.radius_bias is not None:
        new.radius_bias.zero_()


@torch.no_grad()
def _copy_bn_into_center(new: IntBatchNorm1d | IntBatchNorm2d, obj: nn.BatchNorm1d | nn.BatchNorm2d) -> None:
    """Copy a base BatchNorm's affine and running stats into the interval BN's center slots."""
    if obj.affine and new.center_weight is not None and new.center_bias is not None:
        new.center_weight.copy_(obj.weight)
        new.center_bias.copy_(obj.bias)
    if obj.track_running_stats and new.center_running_mean is not None and new.center_running_var is not None:
        new.center_running_mean.copy_(obj.running_mean)
        new.center_running_var.copy_(obj.running_var)


@interval_classifier_traverser.register(nn.Conv2d)
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
    if state[USE_BASE_WEIGHTS]:
        _copy_weight_into_center(new, obj)
    return new, state


@interval_classifier_traverser.register(nn.BatchNorm2d)
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
    if state[USE_BASE_WEIGHTS]:
        _copy_bn_into_center(new, obj)
    return new, state


@interval_classifier_traverser.register(nn.BatchNorm1d)
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
    if state[USE_BASE_WEIGHTS]:
        _copy_bn_into_center(new, obj)
    return new, state


@interval_classifier_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace ``Linear``: insert interval head for the last one, ``IntLinear`` for earlier ones."""
    if state[REPLACED]:
        new_linear = IntLinear(obj.in_features, obj.out_features, bias=obj.bias is not None)
        if state[USE_BASE_WEIGHTS]:
            _copy_weight_into_center(new_linear, obj)
        return new_linear, state
    state[REPLACED] = True
    head_linear = IntLinear(obj.in_features, obj.out_features, bias=obj.bias is not None)
    if state[USE_BASE_WEIGHTS]:
        _copy_weight_into_center(head_linear, obj)
    new_head = nn.Sequential(
        head_linear,
        IntBatchNorm1d(obj.out_features),
        IntSoftmax(),
    )
    return new_head, state
