"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

from probly.representation.predictor import Predictor, PredictorConverter, SamplingRepresentationPredictor
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, lazy_singledispatch_traverser, traverse

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from pytraverse.composition import RegisteredLooseTraverser

P = GlobalVariable[float]("P", "The probability of dropout.")
dropout_traverser = lazy_singledispatch_traverser[object](name="dropout_traverser")


def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
    """Register a class to be prepended by Dropout layers."""
    dropout_traverser.register(cls=cls, traverser=traverser, skip_if=is_first_layer, vars={"p": P})


class Dropout[In, KwIn, Out](
    SamplingRepresentationPredictor[In, KwIn, Out],
    PredictorConverter[In, KwIn, Out],
    metaclass=ABCMeta,
):
    """Implementation of a Dropout ensemble class to be used for uncertainty quantification.

    Attributes:
        p: float, The probability of dropout.
        model: torch.nn.Module, The model with Dropout layers.

    """

    def __init__(
        self,
        base: Predictor[In, KwIn, Out],
        p: float = 0.25,
    ) -> None:
        """Initialize an instance of the Drop class.

        Args:
            base: torch.Predictor, The base model to be used for dropout.
            p: float, The probability of dropping out a neuron.  Default is 0.25.
        """
        self.p = p
        super().__init__(base)

    def _convert(self, base: Predictor[In, KwIn, Out]) -> Predictor[In, KwIn, Out]:
        """Convert the base model to a dropout model.

        Convert the base model by looping through all the layers
        and adding a dropout layer before each linear layer.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.
        """
        return traverse(
            base,
            nn_compose(dropout_traverser),
            init={P: self.p, CLONE: True},
        )
