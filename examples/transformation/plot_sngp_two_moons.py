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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from probly.method.sngp import sngp
from probly.quantification import quantify
from probly.representer import representer

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
for epoch in range(1500):
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

# 1. INCREASE RESOLUTION: 300x300 instead of 100x100 for perfectly smooth boundaries
grid_res = 200
xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, grid_res), np.linspace(-3.0, 3.0, grid_res))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid).float()

with torch.no_grad():
    # Convert from nats (default) to bits to get a [0, 1] scale like the paper
    epistemic_unc = quantify(rep.represent(grid_tensor)).epistemic.numpy() / np.log(2)

epistemic_unc = epistemic_unc.reshape(xx.shape)

# %%
# 5. Plot the epistemic uncertainty surface


fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

levels = np.linspace(0.0, 1.0, 100)
contour = ax.contourf(xx, yy, epistemic_unc, levels=levels, cmap="viridis", antialiased=True)

cbar = plt.colorbar(contour, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_label("Epistemic Uncertainty (bits)", fontsize=12, fontweight='bold')

ax.scatter(X[y == 0, 0], X[y == 0, 1], color="#ff7f0e", edgecolor="white", linewidths=0.5, s=25, zorder=3, label="Class 0")
ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#1f77b4", edgecolor="white", linewidths=0.5, s=25, zorder=3, label="Class 1")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# %%
# 6. Quantitative OOD Evaluation

rng = np.random.default_rng(42)
angles = rng.uniform(0, 2 * np.pi, 150)
radii = np.sqrt(rng.uniform(0, 1, 150))
X_ood = np.column_stack([
    1.5 + radii * 0.8 * np.cos(angles),
    -2.2 + radii * 0.3 * np.sin(angles)
])

with torch.no_grad():
    epistemic_id = quantify(rep.represent(X_tensor)).epistemic.numpy() / np.log(2)
    epistemic_ood = quantify(rep.represent(torch.from_numpy(X_ood).float())).epistemic.numpy() / np.log(2)

labels = np.concatenate([np.zeros(len(X)), np.ones(len(X_ood))])
scores = np.concatenate([epistemic_id, epistemic_ood])
auroc = roc_auc_score(labels, scores)

ax.scatter(X_ood[:, 0], X_ood[:, 1], color="#d62728", marker="x", s=30, alpha=1.0, linewidths=1.5, zorder=4, label="OOD Data")

ax.set_title(f"SNGP Distance-Aware Epistemic Uncertainty\n(OOD AUROC: {auroc:.3f})", fontsize=14, fontweight='bold')
ax.legend(loc="upper right", framealpha=0.95, edgecolor="black")

fig.tight_layout()
plt.show()
