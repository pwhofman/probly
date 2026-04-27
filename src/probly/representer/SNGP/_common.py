"""Shared SNGP representer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from flextype import flexdispatch
from probly.method.SNGP import SNGPPredictor
from probly.predictor import predict
from probly.representation.distribution import CategoricalDistributionSample
from probly.representer._representer import Representer, representer

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representation.sample import Sample


@flexdispatch
def compute_categorical_sample_from_logits(sample: Sample[Any]) -> CategoricalDistributionSample[Any]:
    """Convert a sample of SNGP logits to a categorical distribution sample."""
    msg = f"compute_categorical_sample_from_logits not implemented for type {type(sample)}."
    raise NotImplementedError(msg)


@representer.register(SNGPPredictor)
class SNGPRepresenter[**In, Out](Representer[Any, In, Out, CategoricalDistributionSample[Any]]):
    """A representer that draws samples from a SNGP predictor.

    This representer takes the logits output by the SNGP predictor,
    draws samples from the Gaussian distribution over logits, and converts
    them to a sample of categorical distributions for uncertainty quantification.
    """

    num_samples: int

    def __init__(
        self,
        predictor: Predictor[In, Out],
        num_samples: int = 10,
        *args: In.args,
        **kwargs: In.kwargs,
    ) -> None:
        """Initialize the SNGP representer."""
        super().__init__(predictor, *args, **kwargs)
        self.num_samples = num_samples

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Predict the outputs from the SNGP predictor."""
        return predict(self.predictor, *args, **kwargs)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> CategoricalDistributionSample[Any]:
        distribution = self._predict(*args, **kwargs)
        sampled_logits = distribution.sample(self.num_samples)  # ty:ignore[unresolved-attribute]

        return compute_categorical_sample_from_logits(sampled_logits)
