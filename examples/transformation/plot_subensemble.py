"""=======================================
Sub-Ensemble on Two Moons
=======================================

A Sub-Ensemble freezes a shared pre-trained backbone and trains multiple
lightweight heads with bootstrap aggregation, reducing the cost of full
ensembles. Uncertainty is higher in out-of-distribution regions due to
head diversity in the shared feature space.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import subensemble

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Pre-train the backbone before freezing it

base_model = SequentialModel()

base_model.train()
pretrain_opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
for epoch in range(200):
    pretrain_opt.zero_grad()
    loss = nn.functional.cross_entropy(base_model(X_tensor), y_tensor)
    loss.backward()
    pretrain_opt.step()

# %%
# 3. Create the sub-ensemble and train heads with bootstrap aggregation

subensemble_model = subensemble(
    base_model,
    num_heads=5,
    reset_params=True,
)

subensemble_model.train()
opt = torch.optim.Adam(subensemble_model.parameters(), lr=1e-3)
for epoch in range(500):
    opt.zero_grad()
    total_loss = 0.0
    for member in subensemble_model:
        idx = torch.randint(0, len(X_tensor), (len(X_tensor),))
        total_loss = total_loss + nn.functional.cross_entropy(member(X_tensor[idx]), y_tensor[idx])
    total_loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

subensemble_model.eval()
rep = representer(subensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Sub-Ensemble Predictive Uncertainty")
plot.show()
