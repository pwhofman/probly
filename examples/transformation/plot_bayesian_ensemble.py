"""==============================
Bayesian Ensemble on Two Moons
==============================

A BNN replaces point-weight values with distributions; a Bayesian Ensemble trains several BNNs independently with the ELBO loss.
Predictions combine within-model uncertainty (weight sampling) and between-model uncertainty (initialization).
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch

from probly.representer import representer
from probly.transformation import bayesian_ensemble
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

bayesian_ensemble_model = bayesian_ensemble(
    base_model,
    num_members=5,
    use_base_weights=True,   # seed each member's posterior mean from base_model
    posterior_std=0.05,      # initial posterior std; small = near-deterministic start
    prior_mean=0.0,
    prior_std=1.0,           # smaller = stronger regularization toward zero
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Train each member independently with the ELBO loss.
# collect_kl_divergence is called on each member individually because the KL
# divergence is accumulated per-member during the forward pass.

criterion = ELBOLoss(1.0 / len(X_tensor))

for member in bayesian_ensemble_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(300):
        opt.zero_grad()
        out = member(X_tensor)
        kl = collect_kl_divergence(member)
        loss = criterion(out, y_tensor, kl)
        loss.backward()
        opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

for member in bayesian_ensemble_model:
    member.eval()
rep = representer(bayesian_ensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Bayesian Ensemble Predictive Uncertainty")
plot.show()
