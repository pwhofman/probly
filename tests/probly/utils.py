"""Utils for testing."""

from __future__ import annotations

import numpy as np

# Machine epsilon for np.float64. We assume (for now) that we don't use more precise floats.
# For more information, see https://numpy.org/doc/stable/reference/generated/numpy.finfo.html.
VALIDATE_EPS = np.finfo(np.float64).eps


def validate_uncertainty(uncertainty: np.ndarray) -> None:
    assert isinstance(uncertainty, np.ndarray)
    assert not np.isnan(uncertainty).any()
    assert not np.isinf(uncertainty).any()
    assert (uncertainty >= -VALIDATE_EPS).all()
