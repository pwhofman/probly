"""Generic credal-set representers."""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from typing import TYPE_CHECKING, Any, override

from flextype import flexdispatch
from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE
from probly.predictor import IterablePredictor, predict
from probly.representation.credal_set import create_convex_credal_set, create_probability_intervals
from probly.representation.credal_set._common import ConvexCredalSet, ProbabilityIntervalsCredalSet
from probly.representation.distribution import CategoricalDistribution
from probly.representation.sample import Sample, create_sample
from probly.representer._representer import Representer
from probly.representer.sampler import Sampler
from probly.utils.iterable import first_element

if TYPE_CHECKING:
    from probly.transformation.ensemble import EnsemblePredictor


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


class SampleMeanConvexCredalSetRepresenter[**In, Out: CategoricalDistribution, C: ConvexCredalSet](
    ConvexCredalSetRepresenter[In, Out, C]
):
    """Build a convex credal set whose vertices are sample means of stochastic ensemble members.

    Wraps each member of an iterable predictor in a :class:`Sampler` that draws ``num_samples``
    Monte Carlo predictions and averages them into a single representation per member. The
    resulting M averaged representations form the vertices of a convex credal set.

    This representer is appropriate when each ensemble member is a stochastic predictor
    (such as a Bayesian neural network trained with mean-field variational inference) and the
    desired credal-set vertex is the predictive mean over each member's stochastic posterior.
    """

    num_samples: int
    sub_samplers: list[Sampler]

    def __init__(self, predictor: EnsemblePredictor[In, Out], num_samples: int = 20) -> None:
        """Initialize the representer.

        Args:
            predictor: The ensemble predictor whose members are stochastic predictors.
            num_samples: The number of Monte Carlo samples drawn per ensemble member before averaging.
        """
        super().__init__(predictor)
        self.num_samples = num_samples
        self.sub_samplers = [Sampler(member, num_samples=num_samples) for member in predictor]

    @override
    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Sample each ensemble member and return per-member sample means as a sample."""
        per_member_means = [sampler.represent(*args, **kwargs).sample_mean() for sampler in self.sub_samplers]
        return create_sample(per_member_means)


@compute_representative_sample.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    importlib.import_module("probly.representer.credal_set_torch")


__all__ = [
    "ConvexCredalSetRepresenter",
    "ProbabilityIntervalsRepresenter",
    "RepresentativeConvexCredalSetRepresenter",
    "SampleMeanConvexCredalSetRepresenter",
    "compute_representative_sample",
]
