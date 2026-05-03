"""SNGP: Spectral-normalized Neural Gaussian Process implementation."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, JAX_ARRAY, TORCH_MODULE, TORCH_SAMPLE
from probly.representation.distribution import create_gaussian_distribution

from ._common import (
    SNGPPredictor,
    SNGPRepresenter,
    compute_categorical_sample_from_logits,
    register,
    sngp,
    sngp_traverser,
)

JAX_ARRAY_SAMPLE = "probly.representation.sample.jax.JaxArraySample"


## Torch
@sngp_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@compute_categorical_sample_from_logits.delayed_register(TORCH_SAMPLE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@sngp_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


@compute_categorical_sample_from_logits.delayed_register(JAX_ARRAY_SAMPLE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


@create_gaussian_distribution.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "SNGPPredictor",
    "SNGPRepresenter",
    "compute_categorical_sample_from_logits",
    "register",
    "sngp",
    "sngp_traverser",
]
