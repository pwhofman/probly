"""Torch SNGP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from probly.layers.torch import SNGPLayer, SpectralNormWithMultiplier
from probly.representation.distribution._common import create_categorical_distribution_from_logits
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.sample.torch import TorchSample

from ._common import (
    EPS,
    LAST_LAYER,
    MOMENTUM,
    N_POWER_ITERATIONS,
    NAME,
    NORM_MULTIPLIER,
    NUM_INDUCING,
    RIDGE_PENALTY,
    compute_categorical_sample_from_logits,
    sngp_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


@sngp_traverser.register(nn.Linear)
def replace_linear_with_sngp(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with a HeteroscedasticLayer."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return SNGPLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            num_inducing=state[NUM_INDUCING],
            ridge_penalty=state[RIDGE_PENALTY],
            momentum=state[MOMENTUM],
        ), state
    return SpectralNormWithMultiplier(
        module=obj,
        name=state[NAME],
        n_power_iterations=state[N_POWER_ITERATIONS],
        norm_multiplier=state[NORM_MULTIPLIER],
        eps=state[EPS],
    ), state


@sngp_traverser.register(nn.Conv2d)
def replace_conv_with_spectral_norm(obj: nn.Conv2d, state: State) -> tuple[nn.Module, State]:
    """Replace convolutional layers with spectral normalized versions."""
    return SpectralNormWithMultiplier(
        module=obj,
        name=state[NAME],
        n_power_iterations=state[N_POWER_ITERATIONS],
        norm_multiplier=state[NORM_MULTIPLIER],
        eps=state[EPS],
    ), state


@sngp_traverser.register(nn.Module)
def skip_other_modules(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Skip other modules."""
    return obj, state


@sngp_traverser.register(nn.Softmax)
def remove_layer(obj: nn.Softmax, state: State) -> tuple[nn.Module, State]:
    """Remove the softmax layer."""
    if state[LAST_LAYER]:
        return nn.Identity(), state
    return obj, state


@compute_categorical_sample_from_logits.register(TorchSample)
def torch_compute_categorical_sample_from_logits(
    sample: TorchSample[Any],
) -> TorchCategoricalDistributionSample:
    """Convert a TorchSample of SNGP logits to a categorical distribution sample."""
    tensor = sample.tensor
    sample_dim = sample.sample_dim
    if tensor.ndim >= 3 and sample_dim == 0:
        tensor = tensor.transpose(0, 1)
        sample_dim = 1

    categorical_dist = create_categorical_distribution_from_logits(tensor)
    return TorchCategoricalDistributionSample(tensor=categorical_dist, sample_dim=sample_dim)  # ty: ignore[invalid-argument-type]
