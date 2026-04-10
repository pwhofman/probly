"""Utils module for probly library."""

from .model_inspection import get_output_dim
from .switchdispatch import switchdispatch

__all__ = [
    "get_output_dim",
    "switchdispatch",
]
