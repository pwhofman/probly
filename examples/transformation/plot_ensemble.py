"""=======================================
Deep Ensemble on Two Moons
=======================================

A Deep Ensemble trains multiple independent models from different random
initializations, using prediction disagreement as an uncertainty signal.
Uncertainty concentrates at the decision boundary between classes.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import ensemble

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Create the ensemble from a base model

base_model = MLPClassifier()

ensemble_model = ensemble(
    base_model,
    num_members=3,
    reset_params=True,
)

# %%
# 3. Train each member independently

ensemble_model.train()
for member in ensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)

        opt.zero_grad()
        loss.backward()
        opt.step()

# %%
# 4. Evaluate predictive uncertainty

ensemble_model.eval()
rep = representer(ensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Ensemble Predictive Uncertainty")
plot.show()
