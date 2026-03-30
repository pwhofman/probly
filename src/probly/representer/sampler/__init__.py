"""Samplers for creating representations from predictor outputs."""

from probly.lazy_types import FLAX_MODULE, SKLEARN_MODULE, TORCH_MODULE

from ._common import IterableSampler, Sampler, sampling_preparation_traverser


@sampling_preparation_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@sampling_preparation_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


@sampling_preparation_traverser.delayed_register(SKLEARN_MODULE)
def _(_: type) -> None:
    from . import sklearn as sklearn  # noqa: PLC0415


__all__ = [
    "IterableSampler",
    "Sampler",
]
