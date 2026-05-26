"""==============================
Bayesian Ensemble on Two Moons
==============================

A Bayesian Ensemble combines multiple Bayesian Neural Networks, capturing
both within-model and between-model uncertainty.
Uncertainty concentrates along the decision boundary between classes.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import bayesian_ensemble
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Wrap the base model as a Bayesian Ensemble

base_model = MLPClassifier()

bayesian_ensemble_model = bayesian_ensemble(
    base_model,
    num_members=5,
    use_base_weights=True,
    posterior_std=0.05,
    prior_mean=0.0,
    prior_std=1.0,
    predictor_type="logit_classifier",
)

# %%
# Train each member independently

for member in bayesian_ensemble_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(300):
        criterion = ELBOLoss(1e-5)
        opt.zero_grad()
        out = member(X_tensor)
        kl = collect_kl_divergence(bayesian_ensemble_model)
        loss = criterion(out, y_tensor, kl)
        loss.backward()
        opt.step()

# %%
# Evaluate predictive uncertainty

for member in bayesian_ensemble_model:
    member.eval()
rep = representer(bayesian_ensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Bayesian Ensemble Predictive Uncertainty")
plot.show()
