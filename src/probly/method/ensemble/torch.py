"""Torch ensemble implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.predictor._common import predict_raw
from probly.traverse_nn import nn_compose, nn_traverser, reset_traverser
from pytraverse import CLONE, traverse

from ._common import ensemble_generator


def _reset_copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_compose(reset_traverser), init={CLONE: True})


def _copy(module: nn.Module) -> nn.Module:
    return traverse(module, nn_traverser, init={CLONE: True})


@ensemble_generator.register(nn.Module)
def generate_torch_ensemble(
    obj: nn.Module,
    num_members: int,
    reset_params: bool = True,
) -> nn.ModuleList:
    """Build a torch ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        return nn.ModuleList([_reset_copy(obj) for _ in range(num_members)])
    return nn.ModuleList([_copy(obj) for _ in range(num_members)])


@predict_raw.register(nn.ModuleList)
def predict_module_list[**In](predictor: nn.ModuleList, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
    """Predict for a torch module list ensemble."""
    tensors = [predict_raw(p, *args, **kwargs) for p in predictor]
    return torch.stack(tensors, dim=0)
