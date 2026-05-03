"""Common functions for the reset traverser."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pytraverse import GlobalVariable, flexdispatch_traverser

if TYPE_CHECKING:
    from flax.nnx.rnglib import Rngs, RngStream

type RNG = int | Rngs | RngStream

RNGS = GlobalVariable[RNG](
    "RESET_RNGS",
    "rngs used to draw fresh keys when re-initializing flax parameters during reset.",
    default=1,
)

reset_traverser = flexdispatch_traverser[object](name="reset_traverser")

logger = logging.getLogger(__name__)


@reset_traverser.register
def _(obj: object) -> object:
    """Default fallback: leave the object unchanged.

    Torch implementations register an explicit ``nn.Module`` handler that opportunistically
    calls ``reset_parameters`` if it exists; flax implementations register handlers for the
    layer types that have parameters worth resetting. Anything else (e.g. a jax callable
    used as a Sequential layer, or a raw ``nnx.Variable``) is passed through untouched.
    """
    logger.debug("reset_traverser: no handler for %s; skipping", type(obj).__name__)
    return obj
