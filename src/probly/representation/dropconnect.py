import copy

import torch
import torch.nn as nn
from layers import DropConnectLinear


class DropConnect(nn.Module):
    """
    This class implements DropConnect to be used for uncertainty quantification
    based on http://proceedings.mlr.press/v28/wan13.pdf.
    Args:
        base: torch.nn.Module, The base model to be used for DropConnect.
        p: float, The probability of dropping out individual weights.
    """

    def __init__(self, base: nn.Module, p: float = 0.25) -> None:
        super().__init__()
        self.p = p
        self.model = None
        self._convert(base)

    def forward(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Forward pass of the DropConnect ensemble.
        Args:
            x: torch.Tensor, input data
            n_samples: int, number of stochastic forward passes
        Returns:
            torch.Tensor, ensemble output
        """
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self, base: nn.Module) -> None:
        """
        Converts the base model to a DropConnect model, stored in `self.model`, by modifying all `nn.Linear` layers.
        Args:
            base: torch.nn.Module, The base model to be used for DropConnect.
        """
        self.model = copy.deepcopy(base)

        def apply_drop_connect(module, first_layer=True):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and not first_layer:
                    setattr(module, name, DropConnectLinear(child, p=self.p))  # add DropConnect
                else:
                    if first_layer and not isinstance(child,
                                                      nn.Sequential):  # ignore Sequential layers as first layers
                        first_layer = False  # skip first layer
                    apply_drop_connect(child,
                                       first_layer=first_layer)  # apply recursively to all layers

        apply_drop_connect(self.model)

    def eval(self) -> None:
        """
        Sets the model to evaluation mode but keeps DropConnect layers active.
        """
        super().eval()
        for module in self.model.modules():
            if isinstance(module, DropConnectLinear):
                module.train()  # keep DropConnect active
