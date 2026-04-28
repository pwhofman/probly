"""Uncertainty measures for conformal sets."""

from __future__ import annotations

from typing import Any

from probly.quantification._quantification import measure_atomic
from probly.representation.conformal_set import ConformalSet


@measure_atomic.register(ConformalSet)
def measure_conformal_set_size(conformal_set: ConformalSet) -> Any:  # noqa: ANN401
    """Measure conformal-set uncertainty by set size.

    Args:
        conformal_set: The conformal set to measure.

    Returns:
        The conformal set size for each represented prediction.
    """
    return conformal_set.set_size
