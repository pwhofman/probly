"""====================================
Bayesian Neural Network on Two Moons
====================================

Replace point-estimate weights with distributions and train them with the ELBO loss.
Every forward pass samples new weights, so predictions are inherently stochastic.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch

from probly.representer import representer
from probly.transformation import bayesian
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence

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

base_model = MLPClassifier()

bayesian_model = bayesian(
    base_model,
    use_base_weights=False,  # initialize posterior means randomly rather than from base_model
    posterior_std=0.05,      # initial posterior std; small = near-deterministic start
    prior_mean=0.0,
    prior_std=1.0,           # smaller = stronger regularization toward zero
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# ELBOLoss(beta) computes: cross_entropy(out, y) + beta * kl.
# beta = 1/N scales the KL so its magnitude is independent of dataset size.
# collect_kl_divergence walks the model and sums the KL from every
# BayesianLinear layer, which must be called after each forward pass because
# each forward pass draws new weight samples.

opt = torch.optim.Adam(bayesian_model.parameters(), lr=1e-3)
criterion = ELBOLoss(1.0 / len(X_tensor))

bayesian_model.train()
for epoch in range(300):
    opt.zero_grad()
    out = bayesian_model(X_tensor)
    kl = collect_kl_divergence(bayesian_model)
    loss = criterion(out, y_tensor, kl)
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

bayesian_model.eval()
rep = representer(bayesian_model, num_samples=200)

plot = plot_example_uncertainty(X, y, rep, title="Bayesian Predictive Uncertainty")
plot.show()
