import torch
import torch.nn as nn

from probly.traverse import CLONE, GlobalVariable, Traverser
from probly.traverse_nn import nn_traverse

P = GlobalVariable[float]("P", "The probability of dropout.", default=0.25)


class Drop(nn.Module):
    """
    This class implements a generic drop layer to be used for uncertainty quantification.
    """

    _convert_traverser: Traverser[nn.Module]
    _eval_traverser: Traverser[nn.Module]

    def __init__(
        self,
        base: nn.Module,
        p: float = P.default,
    ) -> None:
        super().__init__()
        self.p = p
        self.model = self._convert(base)

    def forward(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Forward pass of the drop ensemble.
        Args:
            x: torch.Tensor, input data
            n_samples: int, number of samples
        Returns:
            torch.Tensor, ensemble output
        """
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self, base: nn.Module) -> nn.Module:
        """
        Converts the base model to a drop model, stored in model, by looping through all the layers
        and adding a drop layer before each linear layer.
        Args:
            base: torch.nn.Module, The base model to be used for drop.
        """
        return nn_traverse(base, self._convert_traverser, init={P: self.p, CLONE: True})

    def eval(self):
        """
        Sets the model to evaluation mode, but keeps the drop layers active.
        """
        super().eval()

        return nn_traverse(self.model, self._eval_traverser, init={CLONE: False})
