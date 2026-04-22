"""Definition of different notions of uncertainty."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from flextype import ProtocolRegistry
from probly.utils.switchdispatch import switch

type NotionName = Literal[
    # aleatoric
    "aleatoric",
    "au",
    "AU",
    # epistemic
    "epistemic",
    "eu",
    "EU",
    # total
    "total",
    "tu",
    "TU",
]


@runtime_checkable
class Notion(ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for a notion of uncertainty."""


type NotionKey = NotionName | type[Notion]
notion_registry = switch[NotionName, type[Notion]]()


@notion_registry.multi_register(["aleatoric", "au", "AU"])
@runtime_checkable
class AleatoricUncertainty(Notion, Protocol):
    """Protocol for aleatoric uncertainty."""


@notion_registry.multi_register(["epistemic", "eu", "EU"])
@runtime_checkable
class EpistemicUncertainty(Notion, Protocol):
    """Protocol for epistemic uncertainty."""


@notion_registry.multi_register(["total", "tu", "TU"])
@runtime_checkable
class TotalUncertainty(Notion, Protocol):
    """Protocol for total uncertainty."""
