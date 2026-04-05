"""Torch ensemble implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.predictor import predict
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, singledispatch_traverser, traverse

from ._common import ensemble_generator

reset_traverser = singledispatch_traverser[nn.Module](name="reset_traverser")


@reset_traverser.register
def _(obj: nn.Module) -> nn.Module:
    if hasattr(obj, "reset_parameters"):
        obj.reset_parameters()  # ty: ignore[call-non-callable]
    return obj


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


@predict.register(nn.ModuleList)
def predict_module_list[**In](predictor: nn.ModuleList, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
    """Predict for a torch module list ensemble."""
    return torch.stack([p(*args, **kwargs) for p in predictor], dim=1)
