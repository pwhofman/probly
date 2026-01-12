"""Generic representation builder."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor


class Representer[In, KwIn, Out](ABC):
    """Abstract base class for representation builders."""

    predictor: Predictor[In, KwIn, Out]

    def __init__(self, predictor: Predictor[In, KwIn, Out]) -> None:
        """Initialize the representer with a predictor.

        Args:
            predictor: Predictor[In, KwIn, Out], the predictor to be used for building representations.
        """
        self.predictor = predictor
