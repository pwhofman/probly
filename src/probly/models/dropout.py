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
    def __init__(self, base, p):
        super(Dropout, self).__init__()
        self.base = base
        self.p = p
        self.model = None
        self._convert(base)

    def forward(self, x, samples):
        return torch.stack([self.model(x) for _ in range(samples)], dim=1)

    def _convert(self, base):
        """
        Converts the base model to a dropout model, stored in model, by looping through all the layers
        and adding a dropout layer after each linear layer as suggested in ...
        """
        self.model = copy.deepcopy(base)
        first = True
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if not first:
                    setattr(self.model, name, nn.Sequential(nn.Dropout(p=self.p), module))
            first = False

    def eval(self):
        """
        Sets the model to evaluation mode, but keeps the dropout layers active.
        """
        super().eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
