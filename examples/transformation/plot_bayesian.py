"""=======================================
Bayesian Neural Network on Two Moons
=======================================

A Bayesian Neural Network replaces point-weight estimates with weight
distributions, enabling uncertainty estimates via posterior sampling.
Uncertainty concentrates along the decision boundary between classes.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import bayesian

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Wrap the base model as a Bayesian Neural Network

base_model = MLPClassifier()

bayesian_model = bayesian(base_model)

# %%
# 3. Train

opt = torch.optim.Adam(bayesian_model.parameters(), lr=1e-3)

bayesian_model.train()
for epoch in range(300):
    out = bayesian_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

bayesian_model.eval()
rep = representer(bayesian_model, num_samples=200)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="Bayesian Predictive Uncertainty")
plot.show()
