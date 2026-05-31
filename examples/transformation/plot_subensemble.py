"""=========================
Sub-Ensemble on Two Moons
=========================

Share a frozen pre-trained backbone across several independent classification heads.
Useful when an expensive backbone is already trained and only lightweight heads should be replicated to obtain uncertainty.
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
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Backbone Pre-training
# ---------------------

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
# Model
# -----
# ``subensemble`` requires an nn.Sequential for head_layer slicing.

subensemble_model = subensemble(
    base_model,
    num_heads=3,
    reset_params=True,
    head_layer=2,  # split point: lower = more diversity, higher = more sharing
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Only head parameters have requires_grad=True; the frozen backbone is skipped by the optimizer.

subensemble_model.train()
for member in subensemble_model:
    trainable = [p for p in member.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-3)
    for epoch in range(250):
        opt.zero_grad()
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)
        loss.backward()
        opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

subensemble_model.eval()
rep = representer(subensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Sub-Ensemble Predictive Uncertainty", vmin=None, vmax=None)
plot.show()
