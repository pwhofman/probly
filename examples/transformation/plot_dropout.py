"""
========================================
Applying the Dropout Transformation
========================================

Transform a standard PyTorch model with :func:`~probly.transformation.dropout`
so that dropout stays active during inference (MC-Dropout).
"""

import torch
import torch.nn.functional as F
from torch import nn

from probly.transformation import dropout

# %%
# Define a base model
# -------------------

torch.manual_seed(42)


class TinyNet(nn.Module):
    """Small classifier with two hidden layers."""

    def __init__(self, p: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.do1 = nn.Dropout(p)
        self.fc2 = nn.Linear(32, 8)
        self.do2 = nn.Dropout(p)
        self.out = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        return self.out(x)


base_model = TinyNet(p=0.3).eval()

# %%
# Apply the transformation
# ------------------------
# A single call to ``dropout`` wraps the model so that dropout remains active
# even in eval mode — enabling Monte-Carlo inference.

mc_model = dropout(base_model)
mc_model.eval()

print(mc_model)
