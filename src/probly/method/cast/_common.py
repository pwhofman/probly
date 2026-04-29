"""Shared no-op predictor casting implementation."""

from __future__ import annotations

from probly.method.method import predictor_transformation
from probly.predictor import Predictor


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
def cast[T: Predictor](base: T) -> T:
    """Return a predictor unchanged while optionally registering its predictor type.

    Args:
        base: Predictor to return unchanged.

    Returns:
        The unchanged predictor. If ``predictor_type`` is passed to this method,
        the returned predictor is registered as that type.
    """
    return base
