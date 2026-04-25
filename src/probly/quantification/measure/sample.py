"""Uncertainty measures for samples."""

from __future__ import annotations

from typing import Any

from probly.quantification._quantification import quantify
from probly.representation.sample._common import Sample


@quantify.register(Sample)
def quantify_sample_variance(sample: Sample) -> Any:  # noqa: ANN401
    """Quantify uncertainty for samples via their sample variance."""
    return sample.sample_var()
