"""Implementation of representers for credal sets based on ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from probly.method.credal_bnn import CredalBNNPredictor
from probly.method.credal_relative_likelihood import CredalRelativeLikelihoodPredictor
from probly.method.credal_wrapper import CredalWrapperPredictor
from probly.representation.credal_set import create_convex_credal_set, create_probability_intervals
from probly.representation.sample import create_sample
from probly.utils.iterable import first_element

if TYPE_CHECKING:
    from probly.representation.sample import Sample

from flextype import flexdispatch
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.predictor import predict
from probly.representation.credal_set._common import ConvexCredalSet, ProbabilityIntervalsCredalSet
from probly.representation.distribution import CategoricalDistribution
from probly.representer._representer import Representer, representer


@flexdispatch(dispatch_on=first_element)
def compute_credal_ensembling_set[T: CategoricalDistribution](
    sample: Sample[T], alpha: float, distance: str
) -> Sample[T]:
    """Compute the credal set from the ensemble predictions."""
    msg = f"compute_representative_set method not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@flexdispatch(dispatch_on=first_element)
def compute_credal_net_set[T: CategoricalDistribution](sample: Sample[T]) -> Sample[T]:
    """Compute the credal set from the ensemble predictions."""
    msg = f"compute_credal_net_set method not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@representer.register(CredalEnsemblingPredictor)
class CredalEnsemblingRepresenter[**In, Out: CategoricalDistribution, C: ConvexCredalSet](Representer[Any, In, Out, C]):
    def __init__(
        self, predictor: CredalEnsemblingPredictor[In, Out], alpha: float = 0.0, distance: str = "euclidean"
    ) -> None:
        super().__init__(predictor)
        self.alpha = alpha
        self.distance = distance

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the ensemble predictor."""
        ensemble_prediction = predict(self.predictor, *args, **kwargs)
        return create_sample(ensemble_prediction)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        sample = compute_credal_ensembling_set(
            self._predict(*args, **kwargs),
            alpha=self.alpha,
            distance=self.distance,
        )
        cset = create_convex_credal_set(sample)
        return cset  # ty:ignore[invalid-return-type]


@representer.register(CredalWrapperPredictor)
class CredalWrapperRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Out, C]
):
    def __init__(self, predictor: CredalWrapperPredictor) -> None:
        super().__init__(predictor)

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the ensemble predictor."""
        ensemble_prediction = predict(self.predictor, *args, **kwargs)
        return create_sample(ensemble_prediction)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        sample = self._predict(*args, **kwargs)
        cset = create_probability_intervals(sample)
        return cset  # ty:ignore[invalid-return-type]


@representer.register(CredalRelativeLikelihoodPredictor)
class CredalRelativeLikelihoodRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Out, C]
):
    def __init__(self, predictor: CredalRelativeLikelihoodPredictor) -> None:
        super().__init__(predictor)

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the ensemble predictor."""
        ensemble_prediction = predict(self.predictor, *args, **kwargs)
        return create_sample(ensemble_prediction)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        sample = self._predict(*args, **kwargs)
        cset = create_probability_intervals(sample)
        return cset  # ty:ignore[invalid-return-type]


@representer.register(CredalBNNPredictor)
class CredalBNNRepresenter[**In, Out: CategoricalDistribution, C: ConvexCredalSet](Representer[Any, In, Out, C]):
    def __init__(self, predictor: CredalBNNPredictor) -> None:
        super().__init__(predictor)

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Sample[Out]:
        """Predict the outputs from the ensemble predictor."""
        ensemble_prediction = predict(self.predictor, *args, **kwargs)
        return create_sample(ensemble_prediction)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        sample = self._predict(*args, **kwargs)
        cset = create_convex_credal_set(sample)
        return cset  # ty:ignore[invalid-return-type]
