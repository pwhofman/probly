"""Shared HetNets representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.sample import Sample


@runtime_checkable
class HetNetsRepresentation(Representation, Protocol):
    """Representation of a HetNets model output."""

    distribution: Sample


@flexdispatch
def create_het_nets_representation(het_nets_output: Sample) -> HetNetsRepresentation:
    """Create a HetNets representation from a HetNets output."""
    msg = f"No HetNets representation factory registered for output type {type(het_nets_output)}"
    raise NotImplementedError(msg)
