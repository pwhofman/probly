"""Torch SNGP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from probly.layers.torch import SNGPLayer, _SpectralNormParametrization
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
    _collect_skipped_param_bearing_layer_classes,
    compute_categorical_sample_from_logits,
    reset_precision_matrix,
    sngp_traverser,
)

if TYPE_CHECKING:
    from pytraverse import State


def _register_spectral_norm(obj: nn.Module, state: State) -> None:
    """Register `_SpectralNormParametrization` on `obj`'s weight (in place)."""
    name = state[NAME]
    nn.utils.parametrize.register_parametrization(
        obj,
        name,
        _SpectralNormParametrization(
            getattr(obj, name),
            n_power_iterations=state[N_POWER_ITERATIONS],
            norm_multiplier=state[NORM_MULTIPLIER],
            eps=state[EPS],
        ),
    )


@sngp_traverser.register(nn.Linear)
def replace_linear_with_sngp(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last linear layer with an `SNGPLayer`; spec-norm the rest."""
    if state[LAST_LAYER]:
        state[LAST_LAYER] = False
        return SNGPLayer(
            in_features=obj.in_features,
            num_classes=obj.out_features,
            num_inducing=state[NUM_INDUCING],
            ridge_penalty=state[RIDGE_PENALTY],
            momentum=state[MOMENTUM],
        ), state
    _register_spectral_norm(obj, state)
    return obj, state


@sngp_traverser.register(nn.Conv2d)
def replace_conv_with_spectral_norm(obj: nn.Conv2d, state: State) -> tuple[nn.Module, State]:
    """Register a spectral-norm parametrization on the conv's weight (in place)."""
    _register_spectral_norm(obj, state)
    return obj, state


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


@reset_precision_matrix.register(nn.Module)
def _torch_reset_precision_matrix(predictor: nn.Module) -> None:
    """Zero the precision matrix of every ``SNGPLayer`` in a torch predictor."""
    found = 0
    for module in predictor.modules():
        if isinstance(module, SNGPLayer):
            module.reset_precision_matrix()
            found += 1
    if found == 0:
        print(  # noqa: T201
            f"reset_precision_matrix: no SNGPLayer instances found in predictor "
            f"(passed object of type {type(predictor).__name__}).",
        )


@_collect_skipped_param_bearing_layer_classes.register(nn.Module)
def _torch_collect_skipped_param_bearing_layer_classes(predictor: nn.Module) -> list[str]:
    """Return sorted unique class names of param-bearing layers the SNGP traverser skipped.

    A torch layer is "handled" if it is one of:
    - :class:`probly.layers.torch.SNGPLayer` (the GP output layer);
    - :class:`probly.layers.torch._SpectralNormParametrization` (the SN
      parametrization machinery);
    - any ``nn.Linear`` or ``nn.Conv2d`` (the two layer types the traverser
      wraps with spectral normalization);
    - :class:`torch.nn.utils.parametrize.ParametrizationList` (the container
      that the parametrize API uses to hold the original pre-parametrization
      weight as a direct ``Parameter``);
    - any norm layer (``BatchNorm{1,2,3}d``, ``LayerNorm``, ``GroupNorm``)
      which we intentionally do not spec-norm.

    Anything else with direct (non-recursive) parameters - ``Conv1d``,
    ``Conv3d``, ``ConvTranspose*d``, ``LSTM``, ``MultiheadAttention``, custom
    user layers, etc. - is "skipped" by SNGP.
    """
    handled_classes: tuple[type, ...] = (
        nn.Linear,
        nn.Conv2d,
        SNGPLayer,
        _SpectralNormParametrization,
        nn.utils.parametrize.ParametrizationList,
        nn.modules.batchnorm._NormBase,  # noqa: SLF001
        nn.LayerNorm,
        nn.GroupNorm,
    )

    skipped: set[str] = set()
    for module in predictor.modules():
        if isinstance(module, handled_classes):
            continue
        # Direct (non-recursive) parameters only.
        has_direct_params = any(True for _ in module.parameters(recurse=False))
        if has_direct_params:
            skipped.add(type(module).__name__)
    return sorted(skipped)
