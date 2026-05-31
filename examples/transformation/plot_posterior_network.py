"""==============================
Posterior Network on Two Moons
==============================

Output Dirichlet concentration parameters whose evidence comes from normalizing-flow density estimates in a learned latent space.
Uncertainty grows with distance from training data, not just near decision boundaries.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import posterior_network
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
# Strip the final classification layer so the normalizing flows receive feature
# vectors instead of class logits.
#
# ``class_counts`` scales flow densities into Dirichlet pseudo-counts:
# ``alpha = 1 + exp(log_density) * class_count``. Passing 1 (the default) keeps
# all alphas near 1 regardless of density, making uncertainty meaningless.

base_model = MLPClassifier()
backbone = nn.Sequential(*list(base_model.net)[:-1])
class_counts = [int((y == c).sum()) for c in range(2)]

posterior_network_model = posterior_network(
    backbone,
    latent_dim=6,    # dimension of the normalizing-flow latent space
    num_classes=2,
    num_flows=6,     # number of flow steps per class; more = more expressive density model
    class_counts=class_counts,
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# The model outputs Dirichlet concentration parameters (alpha), not logits.
# postnet_loss computes the UCE: expected log-likelihood under the Dirichlet.
# entropy_weight adds a small Dirichlet-entropy term that prevents concentration
# parameters from collapsing to near-zero early in training.

opt = torch.optim.Adam(posterior_network_model.parameters(), lr=1e-3)

posterior_network_model.train()
for epoch in range(1000):
    opt.zero_grad()
    alpha = posterior_network_model(X_tensor)
    loss = postnet_loss(alpha, y_tensor, entropy_weight=1e-5)
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

posterior_network_model.eval()
rep = representer(posterior_network_model)

plot = plot_example_uncertainty(X, y, rep, title="Posterior Network Predictive Uncertainty")
plot.show()
