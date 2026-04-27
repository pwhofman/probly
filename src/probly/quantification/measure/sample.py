"""Uncertainty measures for samples."""

from __future__ import annotations

from typing import Any

from probly.quantification._quantification import measure_atomic
from probly.representation.sample._common import Sample


@measure_atomic.register(Sample)
def measure_sample_variance(sample: Sample) -> Any:  # noqa: ANN401
    """Measure uncertainty for samples via their sample variance."""
    return sample.sample_var()
