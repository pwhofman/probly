"""DropConnect traverser for converting DropConnect layers in a model."""

from __future__ import annotations

from torch import nn  # noqa: TC002

from probly.representation.layers import DropConnectLinear
from probly.representation.predictor_torch import TorchSamplingRepresentationPredictor
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, singledispatch_traverser, traverse

P = GlobalVariable[float]("P", "The probability of dropconnect.")
dropconnect_traverser = singledispatch_traverser[object](name="dropconnect_traverser")


@dropconnect_traverser.register(skip_if=is_first_layer, vars={"p": P})
def _(obj: nn.Linear, p: float) -> DropConnectLinear:
    return DropConnectLinear(obj, p)


@singledispatch_traverser[object]
def eval_traverser(obj: DropConnectLinear) -> DropConnectLinear:
    """Ensure that DropConnect layers are active during evaluation."""
    obj.train()
    return obj


class DropConnect[In, KwIn](TorchSamplingRepresentationPredictor[In, KwIn]):
    """Implementation of a DropConnect model to be used for uncertainty quantification.

    Implementation is based on https://proceedings.mlr.press/v28/wan13.pdf.

    Attributes:
        p: float, the probability of dropping out individual weights.
        model: torch.nn.Module, the model with DropConnect layers.

    """

    def __init__(
        self,
        base: nn.Module,
        p: float = 0.25,
    ) -> None:
        """Initialize an instance of the DropConnect class.

        Args:
            base: torch.nn.Module, The base model to be used for dropconnect.
            p: float, The probability of dropping out a neuron. Default is 0.25.
        """
        self.p = p
        super().__init__(base)

    def _convert(self, base: nn.Module) -> nn.Module:
        """Convert the base model to a dropconnect model.

        Convert the base model by looping through all the layers
        and adding a dropconnect layer before each linear layer.

        Args:
            base: torch.nn.Module, The base model to be used for dropconnect.
        """
        return traverse(
            base,
            nn_compose(dropconnect_traverser),
            init={P: self.p, CLONE: True},
        )

    def eval(self) -> DropConnect:
        """Sets the model to evaluation mode but keeps the dropconnect layers active."""
        super().eval()

        traverse(self.model, nn_compose(eval_traverser), init={CLONE: False})

        return self
