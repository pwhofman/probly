"""==========================
BatchEnsemble on Two Moons
==========================

Replace a full ensemble with per-member rank-1 multiplicative factors on top of a shared backbone.
Training is two-phase: the backbone is pre-trained first, then per-member factors are fine-tuned on a tiled batch.
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
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Backbone Pre-training
# ---------------------
#
# Train the shared backbone with standard cross-entropy before wrapping it
# as a BatchEnsemble so the shared weights start in a sensible region.

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
# Model
# -----
#
# Per-member rank-1 factor vectors ``r`` and ``s`` rescale the shared weight matrix:
# the effective weight for member ``i`` is ``diag(s_i) * W * diag(r_i)``.
# Initializing both near 1.0 keeps members close to the pre-trained backbone at the
# start; the std controls how much they diverge.

num_members = 3
batchensemble_model = batchensemble(
    base_model,
    num_members=num_members,
    use_base_weights=True,  # seed the shared backbone with the pre-trained weights
    r_mean=1.0,             # input-scale factor, identity at 1.0
    r_std=0.5,              # controls input-scale diversity across members
    s_mean=1.0,             # output-scale factor, identity at 1.0
    s_std=0.5,              # controls output-scale diversity across members
    predictor_type="logit_classifier",
)

# %%
# Fine-tuning
# --------
#
# Fine-tune the rank-1 factors (and shared weights) with standard cross-entropy
# on a tiled batch.  The batch must be repeated ``num_members`` times so that
# member ``i`` processes samples ``[i*B : (i+1)*B]`` in a single forward pass.

batchensemble_model.train()
opt = torch.optim.Adam(batchensemble_model.parameters(), lr=1e-3)

# Shape [E*B, ...]: all members process the same data in one forward pass.
X_tiled = X_tensor.repeat(num_members, 1)
y_tiled = y_tensor.repeat(num_members)

for epoch in range(200):
    opt.zero_grad()
    out = batchensemble_model(X_tiled)
    loss = nn.functional.cross_entropy(out, y_tiled)
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

batchensemble_model.eval()
rep = representer(batchensemble_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="BatchEnsemble Predictive Uncertainty")
plot.show()
