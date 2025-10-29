"""from flax to ensemble."""

from __future__ import annotations

from flax.nnx import Module, Sequential

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

# traverser for flax copy
copy_traverser = lazydispatch_traverser[object](name="copy_traverser")


@copy_traverser.register
def _(obj: Module) -> Module:
    # reset_parameter: gewichte zurücksetzt
    return obj


def _copy(module: Module) -> Module:
    return traverse(module, nn_compose(copy_traverser), init={CLONE: True})


def generate_flax_ensemble(obj: Module, members: int) -> Sequential:
    """Flax ensemble with copied model (members times)."""
    return Sequential([_copy(obj) for _ in range(members)])


register(Module, generate_flax_ensemble)
