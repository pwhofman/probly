"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Protocol, TypedDict, Unpack


class Predictor[In, KwIn, Out](Protocol):
    """Protocol for generic predictors."""

    def __call__(self, *args: In, **kwargs: Unpack[KwIn]) -> Out:
        """Call the wrapper with input data."""
        ...


class RepresentationPredictor[In, KwIn, PointwiseOut, RepresentationOut](Protocol):
    """Protocol for representation predictors."""

    def predict_pointwise(self, *args: In, **kwargs: Unpack[KwIn]) -> PointwiseOut:
        """Predict pointwise output from input data."""
        ...

    def predict_representation(self, *args: In, **kwargs: Unpack[KwIn]) -> RepresentationOut:
        """Predict representation output from input data."""
        ...


class SamplingRepresentationCreator[Out, PointwiseOut, RepresentationOut](Protocol):
    """Protocol for representation predictors based on finite samples."""

    def create_representation(self, y: list[Out]) -> RepresentationOut:
        """Create a representation from a collection of outputs."""
        ...

    def create_pointwise(self, y: list[Out]) -> PointwiseOut:
        """Create a pointwise output from a collection of outputs."""
        ...


class ClassifierKwargs(TypedDict):
    """TypedDict for classifier keyword arguments."""

    logits: bool


class Classifier[In, KwIn, Out](Predictor[In, KwIn | ClassifierKwargs, Out], Protocol):
    """Generic protocol for classifiers."""

    def __call__(
        self,
        *args: In,
        logits: bool = False,
        **kwargs: Unpack[KwIn],
    ) -> Out:
        """Call the classifier with input data and return logits or probabilities."""
        ...


class SamplingRepresentationKwargs(TypedDict, total=True):
    """TypedDict for sampling representation keyword arguments."""

    n_samples: int


class SamplingRepresentationPredictor[
    In,
    KwIn,
    Out,
    PointwiseOut,
    RepresentationOut,
](
    Predictor[In, KwIn, Out],
    SamplingRepresentationCreator[Out, PointwiseOut, RepresentationOut],
    RepresentationPredictor[
        In,
        KwIn | SamplingRepresentationKwargs,
        PointwiseOut,
        RepresentationOut,
    ],
    metaclass=ABCMeta,
):
    """Abstract class for predictors that can create representations from finite samples."""

    @abstractmethod
    def _create_representation(self, y: list[Out]) -> RepresentationOut: ...

    @abstractmethod
    def _create_pointwise(self, y: list[Out]) -> PointwiseOut: ...

    def predict_pointwise(self, *args: In, n_samples: int = 0, **kwargs: Unpack[KwIn]) -> PointwiseOut:
        """Produces a point-wise prediction.

        Args:
            args: In, input data
            n_samples: int, number of samples > 0
            kwargs: KwIn, additional keyword arguments
        Returns:
            PointwiseOut, point-wise prediction
        """
        if n_samples <= 0:
            msg = "n_samples must be greater than 0"
            raise ValueError(msg)
        return self.create_pointwise(
            [self.__call__(*args, **kwargs) for _ in range(n_samples)],
        )

    def predict_representation(
        self,
        *args: In,
        n_samples: int = 0,
        **kwargs: Unpack[KwIn],
    ) -> RepresentationOut:
        """Produces an uncertainty representation.

        Args:
            args: In, input data
            n_samples: int, number of samples > 0
            kwargs: KwIn, additional keyword arguments
        Returns:
            RepresentationOut, uncertainty representation
        """
        if n_samples <= 0:
            msg = "n_samples must be greater than 0"
            raise ValueError(msg)
        return self.create_representation(
            [self.__call__(*args, **kwargs) for _ in range(n_samples)],
        )
