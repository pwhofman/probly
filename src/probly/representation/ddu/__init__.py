"""Deep Deterministic Uncertainty representations."""

from __future__ import annotations

from ._common import DDURepresentation, create_ddu_representation
from .torch import TorchDDURepresentation

__all__ = [
    "DDURepresentation",
    "TorchDDURepresentation",
    "create_ddu_representation",
]
