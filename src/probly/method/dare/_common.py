"""Shared definitions for the DARE method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from probly.quantification.decomposition.decomposition import CachingDecomposition, EpistemicDecomposition
from probly.quantification.measure.sample import (
    mean_squared_distance_to_scaled_one_hot,
    measure_sample_variance,
)
from probly.transformation.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.sample import Sample


@runtime_checkable
class DarePredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the DARE method API."""


@predictor_transformation(permitted_predictor_types=None)
@DarePredictor.register_factory(autocast_builtins=True)
def dare[**In, Out](base: Predictor[In, Out], num_members: int, reset_params: bool = True) -> DarePredictor[In, Out]:
    """Create a DARE predictor from a base predictor."""
    return ensemble(base, num_members=num_members, reset_params=reset_params)


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
        dispersion = measure_sample_variance(self.sample).sum(self.class_axis)
        return fit + dispersion  # ty:ignore[invalid-return-type, unsupported-operator]


__all__ = ["DAREDecomposition", "DarePredictor", "dare"]
