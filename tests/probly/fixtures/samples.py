"""Fixtures for Sample representations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.sample import NumpySample


@pytest.fixture
def array_sample_2d() -> NumpySample[int]:
    sample_array = np.arange(12).reshape((3, 4))
    sample = NumpySample[int](sample_array, sample_axis=1)

    return sample
