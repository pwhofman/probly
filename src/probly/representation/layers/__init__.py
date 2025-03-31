"""Init module for layer implementations."""

from .dropconnect import DropConnectLinear
from .normalinversegamma import NormalInverseGammaLinear

__all__ = ["DropConnectLinear", "NormalInverseGammaLinear"]
