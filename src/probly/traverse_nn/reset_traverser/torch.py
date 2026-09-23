"""Torch implementation of reset traverser."""

from __future__ import annotations

import math

import torch
from torch import nn

from ._common import reset_traverser


def _redraw_own_parameters(module: nn.Module) -> None:
    """Redraw the trainable parameters that ``module`` holds itself.

    This is the fallback for modules without a reset method. The scheme their parameters (for
    example a class token or a positional embedding) were initialized with is unknown, so each one
    is redrawn from a normal distribution with the mean and standard deviation of its current
    values. The redrawn parameter keeps its scale, and constant parameters, such as a zero-initialized
    token or a unit gain, keep their value. Single-element parameters have no spread to redraw from
    and are left unchanged, as are frozen parameters (``requires_grad=False``, e.g. fixed random
    features) and buffers.

    Args:
        module: The module whose own parameters, excluding those of its submodules, are redrawn.
    """
    with torch.no_grad():
        for param in module.parameters(recurse=False):
            if not param.requires_grad or not param.is_floating_point() or param.numel() < 2:
                continue
            std, mean = (value.item() for value in torch.std_mean(param))
            if 0.0 < std < math.inf:
                param.normal_(mean, std)


@reset_traverser.register(cls=nn.Module)
def _(obj: nn.Module) -> nn.Module:
    """Re-initialize the parameters that ``obj`` holds itself.

    The traversal visits every submodule before its parent, so each module resets only its own
    parameters, with the scheme it uses at construction:

    1. its public ``reset_parameters()``, the torch convention;
    2. otherwise its private ``_reset_parameters()``, through which ``nn.MultiheadAttention`` and
       ``nn.Transformer`` initialize themselves;
    3. otherwise the fallback of :func:`_redraw_own_parameters`.
    """
    reset = getattr(obj, "reset_parameters", None)
    if not callable(reset):
        reset = getattr(obj, "_reset_parameters", None)
    if callable(reset):
        reset()
    else:
        _redraw_own_parameters(obj)
    return obj
