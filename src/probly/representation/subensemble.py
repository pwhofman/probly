import torch
import torch.nn as nn

class SubEnsemble(nn.Module):
    """
    This class implements an ensemble of representation which share a backbone and use
    different classification heads that can be made up of multiple layers.
    The backbone is frozen and only the head can be trained.

    Args:
    base (torch.nn.Module): The base model to be used.
    num_heads (int): The number of heads in the ensemble.
    head (torch.nn.Module): The classification head to be used. Can be a complete
    network or a single layer.
    """
    def __init__(self, base, num_heads, head):
        super(SubEnsemble, self).__init__()
        self.base = base
        self.models = None
        self._convert(base, num_heads, head)

    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=1)

    def _convert(self, base, num_heads, head):
        for param in base.parameters():
            param.requires_grad = False
        self.models = nn.ModuleList([nn.Sequential(base, head) for _ in range(num_heads)])
