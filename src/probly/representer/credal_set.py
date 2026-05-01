"""Generic credal-set representers."""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from typing import Any, override

from flextype import flexdispatch
from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE
from probly.predictor import IterablePredictor, predict
from probly.representation.credal_set import create_convex_credal_set, create_probability_intervals
from probly.representation.credal_set._common import ConvexCredalSet, ProbabilityIntervalsCredalSet
from probly.representation.distribution import CategoricalDistribution
from probly.representation.sample import Sample, create_sample
from probly.representer._representer import Representer
from probly.utils.iterable import first_element


@flexdispatch(dispatch_on=first_element)
def compute_representative_sample[T: CategoricalDistribution](
    sample: Sample[T], alpha: float, distance: str
) -> Sample[T]:
    """Filter a categorical sample to a representative subset."""
    msg = f"No representative-sample computation registered for type {type(sample)}."
    raise NotImplementedError(msg)


class ConvexCredalSetRepresenter[**In, Out: CategoricalDistribution, C: ConvexCredalSet](
    Representer[Any, In, Iterable[Out], C]
):
    """Build a convex credal set from an iterable categorical predictor."""

    predictor: IterablePredictor[In, Out]

    def __init__(self, predictor: IterablePredictor[In, Out]) -> None:
        """Initialize the representer.

        Args:
            predictor: The iterable predictor whose predictions form the credal-set vertices.
        """
        super().__init__(predictor)

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the iterable predictor."""
        predictions = predict(self.predictor, *args, **kwargs)
        return create_sample(predictions)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Build a convex credal set for the given input."""
        cset = create_convex_credal_set(self._predict(*args, **kwargs))
        return cset  # ty:ignore[invalid-return-type]


class ProbabilityIntervalsRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Iterable[Out], C]
):
    """Build probability intervals from an iterable categorical predictor."""

    predictor: IterablePredictor[In, Out]

    def __init__(self, predictor: IterablePredictor[In, Out]) -> None:
        """Initialize the representer.

        Args:
            predictor: The iterable predictor whose predictions define the intervals.
        """
        super().__init__(predictor)

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the iterable predictor."""
        predictions = predict(self.predictor, *args, **kwargs)
        return create_sample(predictions)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Build probability intervals for the given input."""
        cset = create_probability_intervals(self._predict(*args, **kwargs))
        return cset  # ty:ignore[invalid-return-type]


class RepresentativeConvexCredalSetRepresenter[**In, Out: CategoricalDistribution, C: ConvexCredalSet](
    ConvexCredalSetRepresenter[In, Out, C]
):
    """Build a convex credal set from a representative subset of iterable predictions."""

    alpha: float
    distance: str

    def __init__(self, predictor: IterablePredictor[In, Out], alpha: float = 0.0, distance: str = "euclidean") -> None:
        """Initialize the representer.

        Args:
            predictor: The iterable predictor whose predictions form the candidate vertices.
            alpha: Fraction of the most distant predictions to discard.
            distance: Distance metric used to identify representative predictions.
        """
        super().__init__(predictor)
        self.alpha = alpha
        self.distance = distance

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict and filter the sample to a representative subset."""
        sample = super()._predict(*args, **kwargs)
        return compute_representative_sample(sample, alpha=self.alpha, distance=self.distance)


@compute_representative_sample.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    importlib.import_module("probly.representer.credal_set_torch")


__all__ = [
    "ConvexCredalSetRepresenter",
    "ProbabilityIntervalsRepresenter",
    "RepresentativeConvexCredalSetRepresenter",
    "compute_representative_sample",
]
