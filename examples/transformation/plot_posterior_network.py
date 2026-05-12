"""=======================================
Posterior Network on Two Moons
=======================================

A Posterior Network estimates class-conditional feature densities via
normalizing flows to produce Dirichlet-based uncertainty estimates.
Unlike discriminative methods, uncertainty increases with distance from
the training data manifold rather than at the decision boundary.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import posterior_network

from examples.utils.model import SequentialModel
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

torch.manual_seed(42)

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Build a 64D backbone and wrap it with Posterior Network
#
# The final classification layer is stripped so the normalizing flow
# operates on rich 64D features rather than 2D logits.

backbone = nn.Sequential(*list(SequentialModel())[:-1])

posterior_network_model = posterior_network(backbone, latent_dim=8, num_classes=2)

# %%
# 3. Train with the UCE loss (expected log-likelihood under the Dirichlet)

opt = torch.optim.Adam(posterior_network_model.parameters(), lr=1e-3)

posterior_network_model.train()
for epoch in range(1000):
    out = posterior_network_model(X_tensor)
    log_probs = torch.digamma(out) - torch.digamma(out.sum(dim=-1, keepdim=True))
    loss = nn.functional.nll_loss(log_probs, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

posterior_network_model.eval()
rep = representer(posterior_network_model)

# Dirichlet total uncertainty is not bounded by 1 bit, so use auto-scaling.
plot = plot_example_uncertainty(X, X_tensor, y, rep, title="Posterior Network Predictive Uncertainty", vmin=None, vmax=None)
plot.show()
