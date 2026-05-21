"""==================================
SNGP Distance Awareness on 2D Toys
==================================

This example replicates the toy experiment from Figure 1 of the SNGP paper
(:cite:`liu2020SNGP`). It trains a PyTorch ResNet wrapped with SNGP on the
classic 2-D Two Moons and Blobs classification datasets.

Unlike standard neural networks which confidently extrapolate far away
from the training data, SNGP's Gaussian Process replaces the final layer.
This allows it to inherently measure the distance between a new input and
the training manifold, smoothly increasing epistemic uncertainty in
out-of-distribution regions.
"""

from __future__ import annotations

from sklearn.datasets import make_blobs, make_moons
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.representer import representer

from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset and the Blobs dataset

X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=0)
X_moons_tensor = torch.from_numpy(X_moons).float()
y_moons_tensor = torch.from_numpy(y_moons).long()


X_blobs, y_blobs = make_blobs(n_samples=500, centers=[[-1.0, -1.0], [1.0, 1.0]], cluster_std=0.5, random_state=0)
X_blobs_tensor = torch.from_numpy(X_blobs).float()
y_blobs_tensor = torch.from_numpy(y_blobs).long()


# %%
# Define a deep residual network and wrap it with SNGP
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

base_model_moons = ResFFN()

sngp_model_moons = sngp(
    base_model_moons,
    num_random_features=128,
    ridge_penalty=0.01,
    norm_multiplier=0.9,
    n_power_iterations=1,
)
opt = torch.optim.Adam(sngp_model_moons.parameters(), lr=1e-3)


base_model_blobs = ResFFN()

sngp_model_blobs = sngp(
    base_model_blobs,
    num_random_features=128,
    ridge_penalty=0.01,
    norm_multiplier=0.9,
    n_power_iterations=1,
)
opt_blobs = torch.optim.Adam(sngp_model_blobs.parameters(), lr=1e-3)

# %%
# Train the SNGP model
sngp_model_moons.train()

for epoch in range(300):
    reset_precision_matrix(sngp_model_moons)
    out = sngp_model_moons(X_moons_tensor)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
    loss = nn.functional.cross_entropy(logits, y_moons_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()


sngp_model_blobs.train()

for epoch in range(300):
    reset_precision_matrix(sngp_model_blobs)
    out = sngp_model_blobs(X_blobs_tensor)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
    loss = nn.functional.cross_entropy(logits, y_blobs_tensor)

    opt_blobs.zero_grad()
    loss.backward()
    opt_blobs.step()

# %%
# Evaluate Epistemic Uncertainty over a 2D Grid

sngp_model_moons.eval()
rep_moons = representer(sngp_model_moons, num_samples=800)

plot_moons = plot_example_uncertainty(X_moons, y_moons, rep_moons, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0), title="SNGP Predictive Uncertainty")
plot_moons.show()


sngp_model_blobs.eval()
rep_blobs = representer(sngp_model_blobs, num_samples=800)

plot_blobs = plot_example_uncertainty(
    X_blobs,
    y_blobs,
    rep_blobs,
    title="SNGP Predictive Uncertainty (Blobs)",
    xlim=(-5.0, 5.0),
    ylim=(-5.0, 5.0),
)
plot_blobs.show()
