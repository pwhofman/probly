"""Flax implementation of the reset traverser."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from flax import nnx

from pytraverse import GlobalVariable

from ._common import reset_traverser

type RNG = nnx.Rngs | int

# The value is arbitrary, but it must not be 0: a stream that starts where the model's own stream
# started makes whichever reset happens first in the process hand back the very weights it is meant
# to replace.
_RESET_SEED = 2

# Unlike torch, flax has no process-global rng: a module is initialized from the keys it is
# handed. Defaulting to a single, shared ``Rngs`` instance reproduces torch's behaviour, where
# consecutive ``reset_parameters`` calls draw fresh values, so cloning-and-resetting a model N
# times (as the ensemble transformations do) yields N *different* members. Pass an explicit
# ``Rngs`` or seed via ``init={RNGS: ...}`` to control the stream.
RNGS = GlobalVariable[RNG](
    "RNGS",
    "rngs used to re-initialize flax parameters.",
    default=nnx.Rngs(_RESET_SEED),
)


def _owns_variables(obj: nnx.Module) -> bool:
    """Whether ``obj`` holds variables itself, rather than only submodules.

    Containers such as ``nnx.Sequential`` own no parameters of their own; their layers are
    reset individually as the traversal recurses into them.
    """
    return any(isinstance(child, nnx.Variable) for _, child in nnx.iter_children(obj))


def _init_kwargs(obj: nnx.Module, rngs: nnx.Rngs) -> dict[str, Any]:
    """Recover the ``__init__`` arguments of ``obj`` from its attributes.

    Args:
        obj: The module to introspect.
        rngs: The rngs to pass to the ``rngs`` argument, if the module takes one.

    Returns:
        The keyword arguments to re-instantiate ``obj`` with.

    Raises:
        NotImplementedError: If a required argument cannot be recovered from an attribute.
    """
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, param in inspect.signature(type(obj).__init__).parameters.items():
        if name == "self" or param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        if name == "rngs":
            kwargs[name] = rngs
        elif name in obj.__dict__:
            kwargs[name] = getattr(obj, name)
        elif param.default is param.empty:
            missing.append(name)
    if missing:
        msg = (
            f"cannot reset the parameters of {type(obj).__name__}: its __init__ requires "
            f"{missing}, which it does not keep as attributes. Give the class a "
            f"`reset_parameters()` method, or register a reset_traverser handler for it."
        )
        raise NotImplementedError(msg)
    return kwargs


def _reset_variables(obj: nnx.Module, rngs: nnx.Rngs) -> nnx.Module:
    """Re-initialize the variables ``obj`` owns.

    Prefers an explicit ``reset_parameters`` method (mirroring the torch backend); otherwise
    re-instantiates the module, since flax layers initialize their parameters in ``__init__``.

    Note:
        Flax layers do not keep their initializers as attributes, so a re-instantiated layer
        falls back to the flax default initializer for its class. Layers with a custom
        ``kernel_init``/``bias_init`` should define ``reset_parameters`` instead.
    """
    reset = getattr(obj, "reset_parameters", None)
    if callable(reset):
        reset()
        return obj
    return type(obj)(**_init_kwargs(obj, rngs))


def _refork_rng_stream(obj: nnx.Module, rngs: nnx.Rngs) -> None:
    """Give ``obj`` a fresh rng stream if it carries one.

    Layers such as :class:`probly.layers.flax.DropConnectLinear` keep their own stream to draw
    masks from. Cloned members would otherwise share it and draw identical masks.
    """
    if getattr(obj, "rngs", None) is None:
        return
    collection = getattr(obj, "rng_collection", "default")
    obj.rngs = rngs[collection].fork()  # ty: ignore[unresolved-attribute]


@reset_traverser.register(cls=nnx.Variable)
def _reset_variable(obj: nnx.Variable) -> nnx.Variable:
    """Leave variables alone: they are reset by the module that owns them."""
    return obj


# ``nnx.Sequential`` holds plain callables (``nnx.relu``, lambdas, ...) alongside its modules, and
# the flax nn_traverser descends into them. They carry no parameters, so pass them through instead
# of hitting the "not supported" fallback. Modules win this dispatch: they are registered below.
@reset_traverser.register(cls=Callable)
def _reset_callable(obj: Callable) -> Callable:
    """Leave bare callables (activation functions) untouched."""
    return obj


@reset_traverser.register(cls=nnx.Module, vars={"rngs": RNGS}, update_vars=True)
def _reset_module(obj: nnx.Module, rngs: RNG) -> tuple[nnx.Module, dict[str, RNG]]:
    """Re-initialize the parameters a flax module owns."""
    # Normalize a seed to an ``Rngs`` once and write it back, so that every layer of the model
    # draws from the same advancing stream rather than from an identical fresh one.
    if isinstance(rngs, int):
        rngs = nnx.Rngs(rngs)
    if not _owns_variables(obj):
        return obj, {"rngs": rngs}
    new_obj = _reset_variables(obj, rngs)
    _refork_rng_stream(new_obj, rngs)
    return new_obj, {"rngs": rngs}
