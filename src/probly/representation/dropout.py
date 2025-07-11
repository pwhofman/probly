"""Dropout ensemble implementation for uncertainty quantification."""

from __future__ import annotations

from torch import nn

from probly.representation.predictor_torch import TorchSamplingRepresentationPredictor
from probly.traverse_nn import is_first_layer, nn_compose
from pytraverse import CLONE, GlobalVariable, singledispatch_traverser, traverse

P = GlobalVariable[float]("P", "The probability of dropout.")
dropout_traverser = singledispatch_traverser[object](name="dropout_traverser")


def _prepend_torch_dropout(obj: nn.Module, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(p=p), obj)


def register(cls: type) -> None:
    """Register a class to be prepended by Dropout layers."""
    if issubclass(cls, nn.Module):
        dropout_traverser.register(cls=cls, traverser=_prepend_torch_dropout, skip_if=is_first_layer, vars={"p": P})
    else:
        msg = f"Expected a subclass of nn.Module, got {cls.__name__}"
        raise TypeError(msg)


@singledispatch_traverser[object]
def _eval_dropout_traverser(obj: nn.Dropout) -> nn.Dropout:
    """Ensure that Dropout layers are active during evaluation."""
    return obj.train()


class Dropout[In, KwIn](TorchSamplingRepresentationPredictor[In, KwIn]):
    """Implementation of a Dropout ensemble class to be used for uncertainty quantification.

    Attributes:
        p: float, The probability of dropout.
        model: torch.nn.Module, The model with Dropout layers.

    """

    def __init__(
        self,
        base: nn.Module,
        p: float = 0.25,
    ) -> None:
        """Initialize an instance of the Drop class.

        Args:
            base: torch.nn.Module, The base model to be used for dropout.
            p: float, The probability of dropping out a neuron.  Default is 0.25.
        """
        self.p = p
        super().__init__(base)

    def _convert(self, base: nn.Module) -> nn.Module:
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

    def eval(self) -> Dropout:
        """Sets the model to evaluation mode but keeps the dropout layers active."""
        super().eval()

        traverse(self.model, nn_compose(_eval_dropout_traverser), init={CLONE: False})

        return self


register(nn.Linear)
