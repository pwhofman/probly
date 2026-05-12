"""=========================================
SNGP Distance Awareness on Two Moons
=========================================

This example replicates the toy experiment from Figure 1 of the SNGP paper
(::cite::`liu2022SNGP`). It trains a PyTorch ResNet wrapped with SNGP on the
classic 2-D "Two Moons" classification dataset.

Unlike standard neural networks which confidently extrapolate far away
from the training data, SNGP's Gaussian Process replaces the final layer.
This allows it to inherently measure the distance between a new input and
the training manifold, smoothly increasing epistemic uncertainty in
out-of-distribution regions.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.representer import representer

from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Define a deep residual network and wrap it with SNGP
#
# The SNGP paper explicitly uses a 12-layer, 128-unit deep residual network (ResFFN-12-128)
# to demonstrate that SNGP preserves distance awareness even in deep architectures.

class ResFFNLayer(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.relu(self.norm(self.linear(x)))


class ResFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 128)
        self.layers = nn.ModuleList([ResFFNLayer(128) for _ in range(12)])
        self.last = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        for layer in self.layers:
            x = layer(x)
        return self.last(x)

# 1. Use the Residual Network (CRITICAL)
base_model = ResFFN()

# 2. Configure SNGP parameters
sngp_model = sngp(
    base_model,
    num_random_features=128,
    ridge_penalty=0.01,  # A balanced penalty for ResFFN
    norm_multiplier=0.9,
    n_power_iterations=1,
)
opt = torch.optim.Adam(sngp_model.parameters(), lr=1e-3)

# %%
# 3. Train the SNGP model
sngp_model.train()

# 4. Train for an appropriate number of epochs
for epoch in range(300):
    reset_precision_matrix(sngp_model)
    out = sngp_model(X_tensor)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate Epistemic Uncertainty over a 2D Grid

sngp_model.eval()
rep = representer(sngp_model, num_samples=800)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="SNGP Predictive Uncertainty")
plot.show()
