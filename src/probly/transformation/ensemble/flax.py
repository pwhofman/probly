"""Flax nnx ensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, GlobalVariable, singledispatch_traverser, traverse
from pytraverse.core import State  # noqa: TC001

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")

RNGS = GlobalVariable[nnx.Rngs]("RNGS")


@reset_traverser.register
def _(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    if hasattr(obj, "rngs") and hasattr(obj, "rng_collection") and not any(obj.iter_children()):
        obj.rngs = RNGS(state)[obj.rng_collection].fork()
        return obj, state
    if not any(obj.iter_children()) and "rngs" in obj.__init__.__code__.co_varnames:
        params = {}

        params.update(
            {
                name: getattr(obj, name)
                for name in obj.__init__.__code__.co_varnames
                if name in obj.__dict__ and name != "rngs"
            }
        )
        params["rngs"] = RNGS(state)

        params.pop("kernel_shape", None)

        new_obj = obj.__class__(**params)
        return new_obj, state
    return obj, state


def _clone(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module, rngs: nnx.Rngs) -> nnx.Module:
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True, RNGS: rngs})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
    seed: int,
    rngs: nnx.Rngs,
) -> nnx.List:
    """Build a flax ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        if rngs is None:
            rngs = nnx.Rngs(seed)
        return nnx.List([_clone_reset(obj, rngs) for _ in range(num_members)])
    return nnx.List([_clone(obj) for _ in range(num_members)])


register(nnx.Module, generate_flax_ensemble)
