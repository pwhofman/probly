"""Flax nnx ensemble implementation."""

from __future__ import annotations

from flax import nnx

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import CLONE, GlobalVariable, singledispatch_traverser, traverse
from pytraverse.core import State  # noqa: TC001

from .common import register

reset_traverser = singledispatch_traverser[nnx.Module](name="reset_traverser")

KEY = GlobalVariable[int]("KEY", "Key to be used for reinitialization.")


@reset_traverser.register
def _(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    rng = nnx.Rngs(KEY(state))
    if not any(obj.iter_children()) and "rngs" in obj.__init__.__code__.co_varnames:
        params = {}

        params.update(
            {
                name: getattr(obj, name)
                for name in obj.__init__.__code__.co_varnames
                if name in obj.__dict__ and name != "rngs"
            }
        )
        params["rngs"] = rng

        params.pop("kernel_shape", None)

        new_obj = obj.__class__(**params)
        return new_obj, state
    return obj, state


def _clone(obj: nnx.Module) -> nnx.Module:
    return traverse(obj, nn_traverser, init={CLONE: True})


def _clone_reset(obj: nnx.Module, key: int) -> nnx.Module:
    return traverse(obj, nn_compose(reset_traverser), init={CLONE: True, KEY: key})


def generate_flax_ensemble(
    obj: nnx.Module,
    num_members: int,
    reset_params: bool,
    key: int = 1,
) -> nnx.List:
    """Build a flax ensemble based on :cite:`lakshminarayananSimpleScalable2017`."""
    if reset_params:
        return nnx.List([_clone_reset(obj, key + i) for i in range(num_members)])
    return nnx.List([_clone(obj) for _ in range(num_members)])


register(nnx.Module, generate_flax_ensemble)
