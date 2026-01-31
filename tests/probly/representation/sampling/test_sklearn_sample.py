"""Tests for the SklearnSample Representation."""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.sample import create_sample
from probly.representation.sampling.sklearn_sample import SklearnSample


class TestSklearnSample:
    """Test suite for SklearnSample class."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        samples = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        assert isinstance(sample.array, np.ndarray)
        assert sample.sample_axis == 0
        assert sample.array.shape == (2, 3)

    def test_init_default_axis(self) -> None:
        """Test default sample_axis is 1."""
        samples = [np.array([1, 2]), np.array([3, 4])]
        sample = SklearnSample(samples=samples)
        assert sample.sample_axis == 1

    def test_sample_mean(self) -> None:
        """Test sample mean calculation."""
        samples = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        mean = sample.sample_mean()
        expected = np.array([2.5, 3.5, 4.5])
        assert np.allclose(mean, expected)

    def test_sample_std(self) -> None:
        """Test sample standard deviation."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        std = sample.sample_std(ddof=1)
        expected = np.array([np.sqrt(2), np.sqrt(2)])
        assert np.allclose(std, expected)

    def test_sample_std_ddof_0(self) -> None:
        """Test sample std with ddof=0."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        std = sample.sample_std(ddof=0)
        assert isinstance(std, np.ndarray)

    def test_sample_var(self) -> None:
        """Test sample variance."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        var = sample.sample_var(ddof=1)
        expected = np.array([2.0, 2.0])
        assert np.allclose(var, expected)

    def test_sample_var_ddof_0(self) -> None:
        """Test sample var with ddof=0."""
        samples = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        var = sample.sample_var(ddof=0)
        assert isinstance(var, np.ndarray)

    def test_samples_method_axis_0(self) -> None:
        """Test samples iterator when sample_axis=0."""
        samples = [np.array([1, 2]), np.array([3, 4])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        result = sample.samples()
        assert isinstance(result, np.ndarray)

    def test_samples_method_axis_1(self) -> None:
        """Test samples iterator when sample_axis=1."""
        samples = [np.array([1, 2]), np.array([3, 4])]
        sample = SklearnSample(samples=samples, sample_axis=1)
        result = sample.samples()
        assert isinstance(result, np.ndarray)

    def test_create_sample_with_list(self) -> None:
        """Test create_sample with list of arrays."""
        samples_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        sample = create_sample(samples_list, sample_axis=0)
        assert isinstance(sample, SklearnSample)
        assert sample.array.shape == (2, 3)

    def test_create_sample_2d_array(self) -> None:
        """Test create_sample with 2D array."""
        arr = np.arange(12).reshape((3, 4))
        sample = create_sample(arr, sample_axis=1)
        assert hasattr(sample, "array")
        assert hasattr(sample, "sample_axis")

    def test_init_with_2d_arrays(self) -> None:
        """Test init with 2D arrays in list."""
        samples = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        sample = SklearnSample(samples=samples, sample_axis=0)
        assert sample.array.shape == (2, 2, 2)

    def test_sample_mean_axis_1(self) -> None:
        """Test sample mean with axis=1."""
        samples = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        sample = SklearnSample(samples=samples, sample_axis=1)
        mean = sample.sample_mean()
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (2,)
