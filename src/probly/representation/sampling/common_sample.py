"""Classes representing prediction samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypedDict, Unpack

from lazy_dispatch.singledispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


type SampleAxis = int | Literal["auto"]


class SampleParams(TypedDict, total=False):
    """Default parameters for sample creation."""

    sample_axis: SampleAxis


class SampleFactory[T, S: Sample](Protocol):
    """Protocol for the creation of samples."""

    def __call__(self, samples: Iterable[T], **kwargs: Unpack[SampleParams]) -> S:
        """Create a sample from the given predictions.

        Args:
            samples: The predictions to create the sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created sample.
        """


class Sample[T](ABC):
    """Abstract base class for samples."""

    @classmethod
    @abstractmethod
    def from_iterable(cls, samples: Iterable[T], **kwargs: Unpack[SampleParams]) -> Self:
        """Create an Sample from an iterable of samples.

        Args:
            samples: The predictions to create the sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created ArraySample.
        """
        ...

    @classmethod
    def from_sample(cls, sample: Sample[T], **kwargs: Unpack[SampleParams]) -> Self:
        """Create a new Sample from an existing Sample.

        Args:
            sample: The sample to create the new sample from.
            kwargs: Parameters for sample creation.

        Returns:
            The created Sample.
        """
        return cls.from_iterable(list(sample.samples), **kwargs)

    @property
    @abstractmethod
    def samples(self) -> Iterable[T]:
        """Return an iterator over the samples."""
        ...

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return sum(1 for _ in self.samples)

    def concat(self, other: Sample[T]) -> Self:
        """Append another sample to this sample."""
        return type(self).from_iterable(samples=(sample for s in (self, other) for sample in s.samples))

    def sample_mean(self) -> T:
        """Compute the mean of the sample."""
        msg = "mean method not implemented."
        raise NotImplementedError(msg)

    def sample_std(self, ddof: int = 1) -> T:
        """Compute the standard deviation of the sample."""
        msg = "std method not implemented."
        raise NotImplementedError(msg)

    def sample_var(self, ddof: int = 1) -> T:
        """Compute the variance of the sample."""
        msg = "var method not implemented."
        raise NotImplementedError(msg)


class ListSample[T](list[T], Sample[T]):
    """A sample of predictions stored in a list."""

    @classmethod
    def from_iterable(cls, samples: Iterable[T], sample_axis: SampleAxis = "auto") -> Self:
        """Create a ListSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The axis along which samples are organized.

        Returns:
            The created ListSample.
        """
        if sample_axis != "auto":
            msg = "List-based samples do not support a user-defined sample_dim."
            raise ValueError(msg)

        return cls(samples)

    @property
    def samples(self) -> Sequence[T]:
        """Return an iterator over the samples."""
        return self

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return len(self)

    def concat(self, other: Sample[T]) -> Self:
        """Creates a new sample by concatenating another sample to this sample."""
        return type(self)(self + list(other.samples))


def first_dispatchable_sample(samples: Iterable, **_kwargs: Unpack[SampleParams]) -> Any:  # noqa: ANN401
    """Get the first dispatchable sample from an iterable of samples.

    Args:
        samples: The predictions to create the sample from.
        kwargs: Parameters for sample creation.

    Returns:
        The first dispatchable sample.
    """
    try:
        return samples[0]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None


create_sample = lazydispatch[SampleFactory, Sample](
    ListSample.from_iterable,
    dispatch_on=first_dispatchable_sample,
)
