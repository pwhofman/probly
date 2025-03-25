import copy

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    This class implements a dropout layer to be used for uncertainty quantification.

    Args:
    base (torch.nn.Module): The base model to be used for dropout.
    p (float): The probability of dropping out a neuron.
    """
    def __init__(self, base, p=0.25):
        super().__init__()
        self.p = p
        self.model = None
        self._convert(base)

    def forward(self, x, n_samples):
        return torch.stack([self.model(x) for _ in range(n_samples)], dim=1)

    def _convert(self, base):
        """
        Converts the base model to a dropout model, stored in model, by looping through all the layers
        and adding a dropout layer before each linear layer.
        """
        self.model = copy.deepcopy(base)

        def apply_dropout(module, first_layer=True):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear) and not first_layer:
                    setattr(module, name, nn.Sequential(nn.Dropout(p=self.p), child))  # add dropout
                else:
                    if first_layer and not isinstance(child,
                                                      nn.Sequential):  # ignore Sequential layers as first layers
                        first_layer = False  # skip first layer
                    apply_dropout(child, first_layer=first_layer)  # apply recursively to all layers

        apply_dropout(self.model)

    def eval(self):
        """
        Sets the model to evaluation mode, but keeps the dropout layers active.
        """
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
