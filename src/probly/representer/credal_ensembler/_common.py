"""Implementation of representers for credal sets based on ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from probly.representation.credal_set import create_convex_credal_set
from probly.representation.distribution import CategoricalDistribution
from probly.utils.iterable import first_element

if TYPE_CHECKING:
    from probly.representation.sample import Sample

from lazy_dispatch import lazydispatch
from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.predictor import predict
from probly.representation.credal_set._common import ConvexCredalSet
from probly.representation.sample import create_sample
from probly.representer._representer import Representer, representer


@lazydispatch(dispatch_on=first_element)
def compute_representative_set[T: CategoricalDistribution](probs: Sample[T], alpha: float, distance: str) -> Sample[T]:
    """Compute the credal set from the ensemble predictions."""
    msg = f"compute_representative_set method not implemented for type {type(probs)}."
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
        pred = self._predict(*args, **kwargs)
        distributions = compute_representative_set(pred, alpha=self.alpha, distance=self.distance)
        cset = create_convex_credal_set(distributions)
        return cset  # ty:ignore[invalid-return-type]
