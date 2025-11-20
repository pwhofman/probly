"""Tests for the Sample Representation."""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.sample import ArraySample


class TestArraySample:
    def test_sample_slicing(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, :3]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_dim == 1
        assert indexed_sample.shape == (3, 3)

    def test_sample_selection(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, 3]

        assert isinstance(indexed_sample, np.ndarray)
        assert indexed_sample.shape == (3,)
