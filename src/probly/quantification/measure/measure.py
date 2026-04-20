"""Base class for uncertainty measures."""

from __future__ import annotations

from typing import Protocol

from probly.quantification._quantification import QuantificationResult, Quantifier
from probly.representation.representation import Representation


class MeasureResult(QuantificationResult, Protocol):
    """Protocol for uncertainty measure results."""


class Measure[R: Representation, M: MeasureResult](Quantifier[R, M], Protocol):
    """Protocol for uncertainty measures."""
