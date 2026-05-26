"""==============================
Posterior Network on Two Moons
==============================

Unlike discriminative methods, a Posterior Network increases uncertainty
with distance from the training data rather than only at decision boundaries.
It achieves this by fitting per-class density models (normalizing flows) on
the feature space, producing Dirichlet-parameterized predictions.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import posterior_network

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Wrap the base model with Posterior Network
#
# The final classification layer is stripped so the normalizing flow
# operates on 64D features rather than 2D logits.

base_model = MLPClassifier()
backbone = nn.Sequential(*list(base_model.net)[:-1])

posterior_network_model = posterior_network(
    backbone,
    latent_dim=8,
    num_classes=2,
    predictor_type="logit_classifier",
)

# %%
# Train with the UCE loss (expected log-likelihood under the Dirichlet)

opt = torch.optim.Adam(posterior_network_model.parameters(), lr=1e-3)

posterior_network_model.train()
for epoch in range(750):
    out = posterior_network_model(X_tensor)
    log_probs = torch.digamma(out) - torch.digamma(out.sum(dim=-1, keepdim=True))
    loss = nn.functional.nll_loss(log_probs, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# Evaluate predictive uncertainty

posterior_network_model.eval()
rep = representer(posterior_network_model)

plot = plot_example_uncertainty(X, y, rep, title="Posterior Network Predictive Uncertainty")
plot.show()
