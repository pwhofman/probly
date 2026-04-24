"""Reset traverser module."""

from probly.lazy_types import TORCH_MODULE

from ._common import reset_traverser


## Torch
@reset_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "reset_traverser",
]
