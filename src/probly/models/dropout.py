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
        self.convert()

    def forward(self, x, samples):
        return torch.stack([self.model(x) for _ in range(samples)], dim=1)

    def convert(self):
        """
        Converts the base model to a dropout model, stored in model, by looping through all the layers
        and adding a dropout layer after each linear layer as suggested in ...
        """
        model = nn.Sequential()
        #TODO: Check if this is correct. May add dropout after last layer
        for name, module in self.base.named_children():
            model.add_module(name, module)
            if isinstance(module, nn.Linear):
                model.add_module(f"dropout_{name}", nn.Dropout(p=self.p))
        self.model = model

    def eval(self):
        """
        Sets the model to evaluation mode, but keeps the dropout layers active.
        """
        super().eval()
        for module in self.model.children():
            if isinstance(module, nn.Dropout):
                module.train()
