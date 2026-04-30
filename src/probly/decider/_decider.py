"""Protocols for deciders."""

from __future__ import annotations

from typing import Protocol

from probly.representation.representation import Representation


class Decider[R: Representation, D](Protocol):
    """A decider reduces a representation to a desired decision space."""

    def __call__(self, representation: R) -> D:
        """Reduce the representation to the desired decision space."""
