"""Deciders reduce representations to a desired decision space (e.g., second-order to first-order distribution)."""

from ._decider import Decider
from .categorical_distribution import categorical_from_maximin, categorical_from_mean

__all__ = [
    "Decider",
    "categorical_from_maximin",
    "categorical_from_mean",
]
