"""==========================
BatchEnsemble on Two Moons
==========================

BatchEnsemble approximates a deep ensemble efficiently by applying
per-member rank-1 perturbations to shared weight matrices.
Training requires tiling the batch once per ensemble member.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import batchensemble

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Wrap the base model as a BatchEnsemble

base_model = SequentialModel()

num_members = 3
batchensemble_model = batchensemble(base_model, num_members=num_members)

# %%
# 3. Train — tile the batch once per ensemble member (shape: [E*B, ...])

batchensemble_model.train()
opt = torch.optim.Adam(batchensemble_model.parameters(), lr=1e-3)

X_tiled = X_tensor.repeat(num_members, 1)
y_tiled = y_tensor.repeat(num_members)

for epoch in range(300):
    opt.zero_grad()
    out = batchensemble_model(X_tiled)
    loss = nn.functional.cross_entropy(out, y_tiled)
    loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

batchensemble_model.eval()
rep = representer(batchensemble_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="BatchEnsemble Predictive Uncertainty")
plot.show()
