"""Dropout/DropConnect representation for uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.representation.predictor_torch import TorchSamplingRepresentationPredictor
from probly.traverse_nn import nn_compose
from pytraverse import CLONE, GlobalVariable, Traverser, traverse

if TYPE_CHECKING:
    from torch import nn

P = GlobalVariable[float]("P", "The probability of dropout.", default=0.25)


class Drop[In, KwIn](TorchSamplingRepresentationPredictor[In, KwIn]):
    """This class implements a generic drop layer to be used for uncertainty quantification."""

    _convert_traverser: Traverser
    _eval_traverser: Traverser

    def __init__(
        self,
        base: nn.Module,
        p: float = P.default,
    ) -> None:
        """Initialize an instance of the Drop class.

        Args:
            base: torch.nn.Module, The base model to be used for drop.
            p: float, The probability of dropping out a neuron.
        """
        self.p = p
        super().__init__(base)

    def _convert(self, base: nn.Module) -> nn.Module:
        """Convert base model to a dropout model.

        Convert base model by looping through all the layers
        and adding a dropout layer before each linear layer.

        Args:
            base: torch.nn.Module, The base model to be used for drop.
        """
        return traverse(
            base,
            nn_compose(self._convert_traverser),
            init={P: self.p, CLONE: True},
        )

    def eval(self) -> Drop:
        """Sets the model to evaluation mode, but keeps the drop layers active."""
        super().eval()

        traverse(self.model, nn_compose(self._eval_traverser), init={CLONE: False})

        return self
