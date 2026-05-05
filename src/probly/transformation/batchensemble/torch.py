"""Torch BatchEnsemble implementation."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear
from probly.predictor import predict
from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample.torch import TorchSample

from ._common import BatchEnsemblePredictor, _attach_num_members, _wrap_batchensemble_logits, register


def tile_inputs(x: torch.Tensor, num_members: int) -> torch.Tensor:
    """Tile the leading batch dim by ``num_members``, preserving ``channels_last`` for 4D inputs."""
    out = torch.tile(x, (num_members,) + (1,) * (x.dim() - 1))
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        out = out.contiguous(memory_format=torch.channels_last)
    return out


@_attach_num_members.register(nn.Module)
def _(model: nn.Module, num_members: int) -> None:
    """Register ``num_members`` as a persistent buffer so it survives ``state_dict``."""
    # BatchEnsembleLinear / BatchEnsembleConv2d set ``num_members`` as a plain int in __init__;
    # if the model is just one such layer, drop that attribute before registering the buffer.
    if hasattr(model, "num_members") and "num_members" not in model._buffers:  # noqa: SLF001
        delattr(model, "num_members")
    model.register_buffer("num_members", torch.tensor(num_members, dtype=torch.long))


@predict.register(BatchEnsemblePredictor)
def predict_batchensemble(
    predictor: BatchEnsemblePredictor,
    x: torch.Tensor,
) -> TorchSample[torch.Tensor]:
    """Run a BatchEnsemble predictor and return a :class:`TorchSample` over members.

    Tiles the user's ``[B, ...]`` input by ``num_members``, runs the model on the
    ``[E*B, ...]`` tensor, and reshapes the output to ``[E, B, ...]`` with
    ``sample_dim=0``.
    """
    num_members = int(predictor.num_members)
    b = x.shape[0]
    raw = predictor(tile_inputs(x, num_members))
    out = raw.view(num_members, b, *raw.shape[1:])
    return TorchSample(tensor=out, sample_dim=0)


@_wrap_batchensemble_logits.register(TorchSample)
def _torch_wrap_batchensemble_logits(sample: TorchSample[Any]) -> TorchCategoricalDistributionSample:
    """Wrap per-member logits as a categorical sample (assumes a logit-classifier base)."""
    tensor = sample.tensor
    sample_dim = sample.sample_dim
    if tensor.ndim >= 3 and sample_dim == 0:
        tensor = tensor.transpose(0, 1)
        sample_dim = 1

    distribution = create_categorical_distribution_from_logits(tensor)
    return TorchCategoricalDistributionSample(tensor=distribution, sample_dim=sample_dim)  # ty: ignore[invalid-argument-type]


def replace_torch_batchensemble_linear(
    obj: nn.Linear,
    num_members: int,
    use_base_weights: bool,
    init: Literal["random_sign", "normal"],
    r_mean: float,
    r_std: float,
    s_mean: float,
    s_std: float,
    rngs: Any = None,  # noqa: ARG001, ANN401
) -> BatchEnsembleLinear:
    """Replace a given layer by a BatchEnsembleLinear layer."""
    return BatchEnsembleLinear(
        base_layer=obj,
        num_members=num_members,
        use_base_weights=use_base_weights,
        init=init,
        r_mean=r_mean,
        r_std=r_std,
        s_mean=s_mean,
        s_std=s_std,
    )


def replace_torch_batchensemble_conv2d(
    obj: nn.Conv2d,
    num_members: int,
    use_base_weights: bool,
    init: Literal["random_sign", "normal"],
    r_mean: float,
    r_std: float,
    s_mean: float,
    s_std: float,
    rngs: Any = None,  # noqa: ARG001, ANN401
) -> BatchEnsembleConv2d:
    """Replace a given layer by a BatchEnsembleConv2d layer."""
    return BatchEnsembleConv2d(
        base_layer=obj,
        num_members=num_members,
        use_base_weights=use_base_weights,
        init=init,
        r_mean=r_mean,
        r_std=r_std,
        s_mean=s_mean,
        s_std=s_std,
    )


register(nn.Linear, replace_torch_batchensemble_linear)
register(nn.Conv2d, replace_torch_batchensemble_conv2d)
