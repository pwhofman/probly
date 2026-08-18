"""Reset traverser module."""

from probly.lazy_types import FLAX_MODULE, FLAX_VARIABLE, TORCH_MODULE

from ._common import reset_traverser


## Torch
@reset_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


## Flax
# Registered for variables as well as modules: the flax nn_traverser recurses into ``nnx.Param``
# children, so the traverser is dispatched on a variable before it ever sees a module.
@reset_traverser.delayed_register(FLAX_MODULE)
@reset_traverser.delayed_register(FLAX_VARIABLE)
def _(_: type) -> None:
    from . import flax as flax  # noqa: PLC0415


__all__ = [
    "reset_traverser",
]
