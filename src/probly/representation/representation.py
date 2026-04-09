"""Base class for representations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lazy_dispatch import ProtocolRegistry


@runtime_checkable
class Representation(ProtocolRegistry, Protocol, structural_checking=False):
    """Base class for representations."""
