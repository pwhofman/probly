"""Classes representing prediction samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypedDict, Unpack

from flextype import Flexdispatch, flexdispatch
from probly.representation.representation import Representation
from probly.utils.iterable import first_element

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


class Sample[T](Representation, ABC):
    """Abstract base class for samples."""

    @classmethod
    @abstractmethod
    def from_iterable(
        cls, samples: Iterable[T], weights: Iterable[float] | None = None, **kwargs: Unpack[SampleParams]
    ) -> Self:
        """Create an Sample from an iterable of samples.

        Args:
            samples: The predictions to create the sample from.
            weights: The (optional) weights for each sample.
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

    @property
    @abstractmethod
    def weights(self) -> Iterable[float] | None:
        """Return an iterator over the sample weights."""

    @property
    def is_weighted(self) -> bool:
        """Return whether the samples are weighted."""
        return self.weights is not None

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return sum(1 for _ in self.samples)

    def concat(self, other: Sample[T]) -> Self:
        """Append another sample to this sample."""
        return type(self).from_iterable(samples=(sample for s in (self, other) for sample in s.samples))

    def sample_mean(self) -> T:
        """Compute the (weighted) mean of the sample."""
        msg = "mean method not implemented."
        raise NotImplementedError(msg)

    def sample_std(self, ddof: int = 0) -> T:
        """Compute the standard deviation of the sample."""
        msg = "std method not implemented."
        raise NotImplementedError(msg)

    def sample_var(self, ddof: int = 0) -> T:
        """Compute the variance of the sample."""
        msg = "var method not implemented."
        raise NotImplementedError(msg)


class ListSample[T](list[T], Sample[T]):
    """A sample of predictions stored in a list."""

    weights: list[float] | None = None

    def __init__(self, samples: Iterable[T], weights: Iterable[float] | None = None) -> None:
        super().__init__(samples)
        if weights is not None:
            self.weights = list(weights)
            if len(self.weights) != len(self):
                msg = "Length of weights must match length of samples."
                raise ValueError(msg)

    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[T],
        weights: Iterable[float] | None = None,
        sample_axis: SampleAxis = "auto",
        **__kwargs: Unpack[SampleParams],
    ) -> Self:
        """Create a ListSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            weights: The (optional) weights for each sample.
            sample_axis: The axis along which samples are organized.
            kwargs: Parameters for sample creation.

        Returns:
            The created ListSample.
        """
        if sample_axis != "auto":
            msg = "List-based samples do not support a user-defined sample_dim."
            raise ValueError(msg)

        return cls(samples, weights=weights)

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
        other_samples = list(other.samples)

        weights = self.weights
        other_weights = other.weights

        if weights is not None or other_weights is not None:
            if weights is None:
                weights = [1.0] * len(self)
            other_weights = [1.0] * len(other_samples) if other_weights is None else list(other_weights)
            weights = weights + other_weights

        return type(self)(self + other_samples, weights=weights)


create_sample: Flexdispatch[Any, Sample] = flexdispatch(
    ListSample.from_iterable,
    dispatch_on=first_element,
)


@create_sample.register(Sample)
def _(sample: Sample) -> Sample:
    return sample
