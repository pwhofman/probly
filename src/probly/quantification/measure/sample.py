"""Uncertainty measures for samples."""

from __future__ import annotations

from typing import Any

from probly.quantification._quantification import measure
from probly.representation.sample._common import Sample


@measure.register(Sample)
def measure_sample_variance(sample: Sample) -> Any:  # noqa: ANN401
    """Measure uncertainty for samples via their sample variance."""
    return sample.sample_var()
