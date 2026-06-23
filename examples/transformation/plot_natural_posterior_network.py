"""======================================
Natural Posterior Network on Two Moons
======================================

Natural Posterior Network (NatPN, :cite:`charpentierNaturalPosteriorNetwork2022`)
extends Posterior Network with a Bayesian update over a single density model
and a configurable certainty budget, removing the need for per-class sample
counts.  Uncertainty grows with distance from training data, not just near
decision boundaries.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.natural_posterior_network import natural_posterior_network
from probly.representer import representer
from probly.train.evidential.torch import postnet_loss

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----
#
# Strip the final classification layer so the normalizing flow receives feature
# vectors instead of class logits.  Unlike Posterior Network, NatPN performs a
# Bayesian update against an ``alpha_prior`` and a ``certainty_budget`` instead
# of per-class sample counts.

base_model = MLPClassifier()
backbone = nn.Sequential(*list(base_model.net)[:-1])

natpn_model = natural_posterior_network(
    backbone,
    latent_dim=2,           # dimension of the normalizing-flow latent space
    num_classes=2,
    num_flows=8,            # number of radial flow steps; more = more expressive density model
    certainty_budget="normal",
    alpha_prior=1.0,
)

# %%
# Training
# --------
#
# The model outputs Dirichlet concentration parameters (alpha), not logits.
# postnet_loss computes the UCE: expected log-likelihood under the Dirichlet;
# the NatPN paper applies it with mean reduction.
# entropy_weight adds a small Dirichlet-entropy term that prevents concentration
# parameters from collapsing to near-zero early in training.

opt = torch.optim.Adam(natpn_model.parameters(), lr=1e-3)

natpn_model.train()
for _epoch in range(1000):
    opt.zero_grad()
    alpha = natpn_model(X_tensor)
    loss = postnet_loss(alpha, y_tensor, entropy_weight=1e-5, reduction="mean")
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

natpn_model.eval()
rep = representer(natpn_model)

plot = plot_example_uncertainty(
    X, y, rep,
    title="Natural Posterior Network Predictive Uncertainty",
    notion="total",
)
plot.show()
