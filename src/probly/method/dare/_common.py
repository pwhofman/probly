"""Shared definitions for the DARE method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from probly.predictor import LogitClassifier
from probly.quantification._quantification import decompose
from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.quantification.measure.sample import (
    mean_squared_distance_to_scaled_one_hot,
    sample_variance,
)
from probly.representation.distribution import CategoricalDistributionSample
from probly.representation.representation import Representation
from probly.representation.sample import Sample, SampleFactory, create_sample
from probly.representer import IterableSampler, representer
from probly.transformation.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


@runtime_checkable
class DarePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the DARE method API."""


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=False)
@DarePredictor.register_factory(autocast_builtins=True)
def dare[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True) -> DarePredictor[In, Out]:
    """Create a DARE predictor from a base predictor."""
    return ensemble(base, num_members=num_members, reset_params=reset_params)


@runtime_checkable
class DARERepresentation(Representation, Protocol):
    """Pseudo-representation type marking outputs of the DARE method.

    Marker protocol used to route :func:`decompose` to :class:`DAREDecomposition`.
    """


# Register as a virtual subclass of CategoricalDistributionSample so the dispatch considers the marker
# more specific than the generic CategoricalDistributionSample handlers.
CategoricalDistributionSample.register(DARERepresentation)


class DARERepresenter[**In, Out](IterableSampler[In, Out, DARERepresentation]):  # ty:ignore[invalid-type-arguments]
    """Representer for DARE predictors that marks the output sample for method-specific dispatch.

    Defaults ``sample_axis=0`` (member axis first, class axis trailing) to match
    :class:`DAREDecomposition`'s ``class_axis=-1`` convention.
    """

    def __init__(
        self,
        predictor: EnsemblePredictor[In, Out],
        sample_factory: SampleFactory[Out, DARERepresentation] = create_sample,  # ty:ignore[invalid-type-arguments]
        sample_axis: int = 0,
    ) -> None:
        """Initialize the DARE representer."""
        super().__init__(predictor, sample_factory=sample_factory, sample_axis=sample_axis)

    @override
    @DARERepresentation.register_factory
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> DARERepresentation:
        """Return a marked DARE sample representation for a given input."""
        return self._create_sample(self._predict(*args, **kwargs))


representer.register(DarePredictor, DARERepresenter)


@decompose.register(DARERepresentation)
@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class DAREDecomposition[T](CachingDecomposition, EpistemicDecomposition[T]):
    """DARE OOD score (Appendix F, Eq. 35 :cite:`mathelinDeepAntiregularizedEnsembles2023`): fit + dispersion.

    Args:
        sample: Logits sample across ensemble members; class axis is last.
        target_scale: One-hot target scale; ``None`` uses K (num classes).
        class_axis: Axis summed for the dispersion term.
    """

    sample: Sample
    target_scale: float | None = None
    class_axis: int = -1

    @override
    @property
    def _epistemic(self) -> T:
        """The DARE OOD score."""
        fit = mean_squared_distance_to_scaled_one_hot(self.sample, scale=self.target_scale)
        dispersion = sample_variance(self.sample).sum(self.class_axis)
        return fit + dispersion


__all__ = ["DAREDecomposition", "DARERepresentation", "DARERepresenter", "DarePredictor", "dare"]
