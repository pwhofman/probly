"""Shared efficient credal prediction implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probly.predictor import Predictor


def efficient_credal_prediction[**In, Out](base: Predictor[In, Out]) -> Predictor[In, Out]:
    """Create a efficient credal predictor from a base predictor based on :cite:`hofmanefficient`.

    Args:
        base: Predictor, The base model to be used for the efficient credal predictor.

    Returns:
        Predictor, The efficient credal predictor.
    """
    return base
