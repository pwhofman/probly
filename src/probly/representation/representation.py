"""Base class for representations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flextype import ProtocolRegistry


@runtime_checkable
class Representation(ProtocolRegistry, Protocol, structural_checking=False):
    """Base class for representations."""


@runtime_checkable
class CanonicalRepresentation[T](Protocol):
    """Runtime-structural protocol for representations with a canonical concrete element.

    Args:
        T: The type of the canonical element.
    """

    @property
    def canonical_element(self) -> T:
        """Return the canonical element represented by this uncertainty object."""
        ...
