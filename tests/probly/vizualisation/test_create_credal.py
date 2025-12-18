"""Test for create_credal_plot."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest

# Use a non-interactive backend so tests do not open GUI windows.
mpl.use("Agg")


from probly.visualization.create_credal import create_credal_plot


def test_check_shape_raises_on_empty_dataset() -> None:
    with pytest.raises(ValueError, match=r"must not be empty"):
        create_credal_plot(np.array([]))
