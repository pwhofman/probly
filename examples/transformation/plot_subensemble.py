"""=========================
Sub-Ensemble on Two Moons
=========================

A Sub-Ensemble freezes a shared backbone and trains multiple independent
heads, giving ensemble-style uncertainty at a fraction of the cost. Because
the backbone is shared, diversity -- and thus uncertainty -- comes solely
from head disagreement rather than full model disagreement.
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
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Pre-train the shared backbone

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
# Create the sub-ensemble from the pre-trained backbone

subensemble_model = subensemble(
    base_model,
    num_heads=3,
    reset_params=True,
    head_layer=2,
    predictor_type="logit_classifier",
)

# %%
# Train each head independently

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
# Evaluate predictive uncertainty

subensemble_model.eval()
rep = representer(subensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Sub-Ensemble Predictive Uncertainty", vmin=None, vmax=None)
plot.show()
