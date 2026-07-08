"""Test sample dispatching logic."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")
from jax import numpy as jnp
import numpy as np
import torch

from probly.representation.sample import ArraySample, ListSample, create_sample
from probly.representation.sample.jax import JaxArraySample
from probly.representation.sample.torch import TorchSample


class TestSampleDispatching:
    def test_create_array_sample_numpy(self) -> None:
        x = np.arange(12).reshape((3, 4))
        sample = create_sample(x)
        assert isinstance(sample, ArraySample)
        assert sample.shape == (4, 3)
        assert sample.sample_axis == 1

    def test_create_array_sample_jax(self) -> None:
        x = jnp.arange(12).reshape((3, 4))
        sample = create_sample(x, sample_axis=1)
        assert isinstance(sample, JaxArraySample)
        assert sample.shape == (4, 3)
        assert sample.sample_axis == 1

    def test_create_array_sample_torch(self) -> None:
        x = torch.arange(12).reshape((3, 4))
        sample = create_sample(x, sample_axis=0)
        assert isinstance(sample, TorchSample)
        assert sample.shape == (3, 4)
        assert sample.sample_dim == 0
        assert x is sample.tensor

    def test_create_array_sample_numpy_preserves_weights(self) -> None:
        x = np.arange(12).reshape((3, 4))
        weights = np.array([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, ArraySample)
        assert np.array_equal(sample.weights, weights)

    def test_create_array_sample_jax_preserves_weights(self) -> None:
        x = jnp.arange(12).reshape((3, 4))
        weights = jnp.array([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, JaxArraySample)
        assert np.array_equal(np.asarray(sample.weights), np.asarray(weights))

    def test_create_array_sample_torch_preserves_weights(self) -> None:
        x = torch.arange(12).reshape((3, 4))
        weights = torch.tensor([0.1, 0.2, 0.3])

        sample = create_sample(x, sample_axis=0, weights=weights)

        assert isinstance(sample, TorchSample)
        assert torch.equal(sample.weights, weights)


class TestListSampleWeights:
    def test_from_iterable_preserves_weights(self) -> None:
        sample = ListSample.from_iterable([1, 2, 3], weights=[0.1, 0.2, 0.3])

        assert sample.weights == [0.1, 0.2, 0.3]

    def test_constructor_rejects_wrong_weight_length(self) -> None:
        with pytest.raises(ValueError, match="Length of weights"):
            ListSample([1, 2, 3], weights=[0.1, 0.2])

    def test_concat_combines_weights(self) -> None:
        left = ListSample([1, 2], weights=[0.1, 0.2])
        right = ListSample([3, 4], weights=[0.3, 0.4])

        result = left.concat(right)

        assert result == [1, 2, 3, 4]
        assert result.weights == [0.1, 0.2, 0.3, 0.4]

    def test_concat_fills_missing_weights_with_ones(self) -> None:
        left = ListSample([1, 2])
        right = ListSample([3, 4], weights=[0.3, 0.4])

        result = left.concat(right)

        assert result == [1, 2, 3, 4]
        assert result.weights == [1.0, 1.0, 0.3, 0.4]


class TestListSampleConstruction:
    def test_basic(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s = ListSample([1, 2, 3])
        assert s.sample_size == 3
        assert list(s.samples) == [1, 2, 3]
        assert not s.is_weighted

    def test_with_weights(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s = ListSample([1, 2, 3], weights=[0.5, 0.3, 0.2])
        assert s.is_weighted
        assert s.weights == [0.5, 0.3, 0.2]

    def test_weight_length_mismatch_raises(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="Length of weights"):
            ListSample([1, 2, 3], weights=[0.5, 0.5])

    def test_from_iterable_default(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s = ListSample.from_iterable([1, 2, 3])
        assert s.sample_size == 3

    def test_from_iterable_with_explicit_axis_raises(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        with pytest.raises(ValueError, match="user-defined sample_dim"):
            ListSample.from_iterable([1, 2, 3], sample_axis=0)


class TestListSampleConcat:
    def test_unweighted_concat(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s1 = ListSample([1, 2])
        s2 = ListSample([3, 4])
        out = s1.concat(s2)
        assert list(out.samples) == [1, 2, 3, 4]
        assert not out.is_weighted

    def test_concat_one_weighted(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s1 = ListSample([1, 2], weights=[0.5, 0.5])
        s2 = ListSample([3, 4])
        out = s1.concat(s2)
        # other was unweighted -> filled with 1.0.
        assert out.weights == [0.5, 0.5, 1.0, 1.0]

    def test_concat_other_weighted(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s1 = ListSample([1, 2])
        s2 = ListSample([3, 4], weights=[0.7, 0.3])
        out = s1.concat(s2)
        assert out.weights == [1.0, 1.0, 0.7, 0.3]


class TestCreateSampleDispatch:
    def test_existing_sample_with_sample_first_element(self) -> None:
        """create_sample dispatches on ``first_element``; if the first element is a Sample,
        the passthrough handler fires.
        """  # noqa: D205
        from probly.representation.sample._common import ListSample, create_sample  # noqa: PLC0415

        # A ListSample whose first element is itself a ListSample.
        inner = ListSample([1, 2, 3])
        outer = ListSample([inner])
        # The dispatch is on first_element(outer) = inner, which matches the Sample handler.
        result = create_sample(outer)
        # The passthrough returns the wrapping Sample.
        assert result is outer


class TestSampleAbstractFallbacks:
    """The default Sample.sample_mean / sample_std / sample_var raise NotImplementedError."""

    def test_default_fallbacks_raise(self) -> None:
        from probly.representation.sample._common import Sample  # noqa: PLC0415

        # ListSample inherits these defaults — call them directly via Sample base class.
        # Subclass that doesn't override these methods.
        class Tiny:  # not a real Sample, just enough to call the methods.
            pass

        class _RealTiny(Sample):
            def __iter__(self):  # noqa: ANN204
                return iter([])

            @property
            def samples(self):
                return []

            @property
            def weights(self):
                return None

            @classmethod
            def from_iterable(cls, samples, weights=None, sample_axis="auto", **kwargs):  # noqa: ANN003, ANN206, ARG003
                return cls()

            def from_sample(self, sample, sample_axis="auto"):  # noqa: ARG002
                return self

        t = _RealTiny()
        with pytest.raises(NotImplementedError):
            t.sample_mean()
        with pytest.raises(NotImplementedError):
            t.sample_std()
        with pytest.raises(NotImplementedError):
            t.sample_var()


class TestListSampleSampleSize:
    def test_with_iterator(self) -> None:
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        s = ListSample([1, 2, 3])
        # ListSample overrides sample_size to use len(); just verify it.
        assert s.sample_size == 3
