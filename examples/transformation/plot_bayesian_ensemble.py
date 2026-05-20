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

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Wrap the base model as a Bayesian Ensemble

base_model = MLPClassifier()

bayesian_ensemble_model = bayesian_ensemble(base_model, num_members=5)

# %%
# 3. Train each member independently

for member in bayesian_ensemble_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(300):
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)

        opt.zero_grad()
        loss.backward()
        opt.step()

# %%
# 4. Evaluate predictive uncertainty

for member in bayesian_ensemble_model:
    member.eval()
rep = representer(bayesian_ensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Bayesian Ensemble Predictive Uncertainty")
plot.show()
