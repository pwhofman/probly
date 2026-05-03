"""Reset traverser module."""

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import RNGS, reset_traverser


## Torch
@reset_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
@reset_traverser.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "RNGS",
    "reset_traverser",
]
