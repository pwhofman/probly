"""Torch VBLL implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from probly.layers.torch import HetVBLLLayer, TVBLLLayer, VBLLLayer
from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample.torch import TorchSample

from ._common import (
    COV_RANK,
    DOF,
    LAST_LAYER,
    NOISE_INIT,
    NOISE_PRIOR_SCALE,
    PARAMETERIZATION,
    PRIOR_SCALE,
    VARIANT,
    WISHART_SCALE,
    compute_vbll_categorical_sample,
    vbll_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


def _build_vbll_layer(obj: nn.Linear, state: State) -> nn.Module:
    """Build the discriminative VBLL last layer for the requested variant."""
    variant = state[VARIANT]
    parameterization = state[PARAMETERIZATION]
    if variant == "student_t":
        return TVBLLLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            parameterization=parameterization,  # ty:ignore[invalid-argument-type]
            prior_scale=state[PRIOR_SCALE],
            wishart_scale=state[WISHART_SCALE],
            dof=state[DOF],
        )
    if variant == "heteroscedastic":
        return HetVBLLLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            parameterization=parameterization,  # ty:ignore[invalid-argument-type]
            prior_scale=state[PRIOR_SCALE],
            noise_prior_scale=state[NOISE_PRIOR_SCALE],
        )
    return VBLLLayer(
        in_features=obj.in_features,
        num_outputs=obj.out_features,
        parameterization=parameterization,  # ty:ignore[invalid-argument-type]
        prior_scale=state[PRIOR_SCALE],
        noise_init=state[NOISE_INIT],
        cov_rank=state[COV_RANK],
        wishart_scale=state[WISHART_SCALE],
        dof=state[DOF],
    )


@vbll_traverser.register(nn.Linear)
def replace_linear_with_vbll(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with a VBLL last layer; leave the rest untouched."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return _build_vbll_layer(obj, state), state
    return obj, state


@vbll_traverser.register(nn.Softmax)
def remove_softmax(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove a trailing softmax layer; the VBLL layer outputs logits."""
    if state[LAST_LAYER]:
        return nn.Identity(), state
    return obj, state


@vbll_traverser.register(nn.Module)
def skip_other_modules(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Leave other modules unchanged."""
    return obj, state


@compute_vbll_categorical_sample.register(TorchSample)
def torch_compute_vbll_categorical_sample(sample: TorchSample[Any]) -> TorchCategoricalDistributionSample:
    """Convert a TorchSample of VBLL logits to a categorical distribution sample."""
    tensor = sample.tensor
    sample_dim = sample.sample_dim
    if tensor.ndim >= 3 and sample_dim == 0:
        tensor = tensor.transpose(0, 1)
        sample_dim = 1

    categorical_dist = create_categorical_distribution_from_logits(tensor)
    return TorchCategoricalDistributionSample(tensor=categorical_dist, sample_dim=sample_dim)  # ty:ignore[invalid-argument-type]
