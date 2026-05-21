"""==========================
BatchEnsemble on Two Moons
==========================

BatchEnsemble approximates a deep ensemble efficiently by applying
per-member rank-1 perturbations to shared weight matrices. It achieves
near-ensemble uncertainty at a fraction of the memory cost, since weights
are shared across members.
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
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Pre-train the shared backbone

base_model = SequentialModel()
opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)

base_model.train()
for epoch in range(300):
    opt.zero_grad()
    out = base_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    loss.backward()
    opt.step()

# %%
# Wrap the pre-trained model as a BatchEnsemble

num_members = 3
batchensemble_model = batchensemble(
    base_model,
    num_members=num_members,
    use_base_weights=True,
    r_mean=1.0,
    r_std=0.5,
    s_mean=1.0,
    s_std=0.5,
    predictor_type="logit_classifier",
)

# %%
# Fine-tune — tile the batch once per ensemble member (shape: [E*B, ...])

batchensemble_model.train()
opt = torch.optim.Adam(batchensemble_model.parameters(), lr=1e-3)

X_tiled = X_tensor.repeat(num_members, 1)
y_tiled = y_tensor.repeat(num_members)

for epoch in range(200):
    opt.zero_grad()
    out = batchensemble_model(X_tiled)
    loss = nn.functional.cross_entropy(out, y_tiled)
    loss.backward()
    opt.step()

# %%
# Evaluate predictive uncertainty

batchensemble_model.eval()
rep = representer(batchensemble_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="BatchEnsemble Predictive Uncertainty")
plot.show()
