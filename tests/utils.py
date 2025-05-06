"""Utils for testing."""

from __future__ import annotations

import numpy as np


def validate_uncertainty(uncertainty: np.ndarray) -> None:
    assert isinstance(uncertainty, np.ndarray)
    assert not np.isnan(uncertainty).any()
    assert not np.isinf(uncertainty).any()
    assert (uncertainty >= 0).all()
