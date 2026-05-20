"""=========================
Sub-Ensemble on Two Moons
=========================

A Sub-Ensemble freezes a shared backbone and trains multiple independent
heads, reducing the cost of full ensembles. Uncertainty is higher in
out-of-distribution regions due to head diversity in the shared feature space.
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
# 2. Pre-train the shared backbone

base_model = SequentialModel()
base_model.train()
opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
for epoch in range(250):
    opt.zero_grad()
    out = base_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    loss.backward()
    opt.step()

print(f"backbone loss: {loss:.4f}")

# %%
# 3. Create the sub-ensemble from the pre-trained backbone

subensemble_model = subensemble(
    base_model,
    num_heads=3,
    reset_params=True,
    head_layer=2,
    predictor_type="logit_classifier",
)

# %%
# 4. Train each head independently

subensemble_model.train()
for member in subensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        opt.zero_grad()
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)
        loss.backward()
        opt.step()

# %%
# 5. Evaluate predictive uncertainty

subensemble_model.eval()
rep = representer(subensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Sub-Ensemble Predictive Uncertainty", vmin=None, vmax=None)
plot.show()
